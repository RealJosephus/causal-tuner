import copy
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from causal_tuner.data.segments import canonical_segments, position_ids, prediction_targets, runs
from causal_tuner.modeling.rotary import rotary_values, rotate
from causal_tuner.modeling.model import CausalLM, Runtime, MLP
from causal_tuner.kernels.loss import projection_loss
from conftest import batch


def test_fixed_parameter_counts():
    from causal_tuner.modeling.config import ModelConfig
    assert ModelConfig().parameter_counts() == (9_574_004_736, 796_917_760)
    model = CausalLM(ModelConfig(), Runtime(), device="meta")
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad) == 9_574_004_736
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 796_917_760


@pytest.mark.parametrize("max_positions", [32768, 4194304])
def test_checkpoint_config_controls_model_and_position_range(max_positions):
    from causal_tuner.modeling.config import ModelConfig
    expected = replace(ModelConfig.tiny(), max_position_embeddings=max_positions,
                       rope_theta=10000., partial_rotary_factor=1., rms_norm_eps=1e-5)
    config = ModelConfig.from_dict({**expected.to_dict(), "model_type": "fixture"})
    assert config == expected
    model = CausalLM(config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32,
                                    attention="vanilla", checkpoint=False))
    data = batch([[2, 3, 4]])
    data["position_ids"] = torch.tensor([[0, max_positions-2, max_positions-1]])
    output = model(data)
    assert torch.isfinite(output).all()
    F.linear(output, model.lm_head.weight).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    data["position_ids"][0, -1] = max_positions
    with pytest.raises(ValueError, match="logical position range"):
        model(data)


def test_contiguous_segments_and_shifted_weights():
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1]])
    labels = torch.tensor([[0, 0, 5, 0, 0, 8, 0, 0]])
    segments = canonical_segments(mask, labels)
    assert segments.tolist() == [[1, 1, 0, 2, 2, 3, 4, 4]]
    assert position_ids(segments).tolist() == [[0, 1, 0, 0, 1, 0, 0, 1]]
    data = batch([[2, 3, 4, 5, 6, 7, 8, 9]], [[1, .2, 1, 1, 2, 1, 1, 3]], labels)
    data["attention_mask"] = mask
    data["labels"] = torch.tensor([[2, 3, 4, 5, -100, 7, 8, 9]])
    targets, weights = prediction_targets(data)
    assert weights.tolist() == [[pytest.approx(.2), 0, 0, 0, 0, 0, 3, 0]]
    assert targets.tolist() == [[3, 0, 0, 0, 0, 0, 9, 0]]


def test_partial_rotary_at_large_positions(tiny_config):
    q = torch.randn(1, 3, 4, 8, requires_grad=True)
    positions = torch.tensor([[0, 1999999, 2000000]])
    cosine, sine = rotary_values(positions, tiny_config, torch.float32)
    actual = rotate(q, cosine, sine)
    expected = q.clone()
    for token, position in enumerate(positions[0]):
        for channel in range(0, 4, 2):
            angle = position.float()/(tiny_config.rope_theta**(channel/4))
            expected[0, token, :, channel] = q[0, token, :, channel]*angle.cos()-q[0, token, :, channel+1]*angle.sin()
            expected[0, token, :, channel+1] = q[0, token, :, channel+1]*angle.cos()+q[0, token, :, channel]*angle.sin()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[..., 4:], q[..., 4:])


@pytest.mark.parametrize("train_head", [False, True])
@pytest.mark.parametrize("weight", [0.0, .2, 1.0])
def test_projection_loss_against_full_logits(train_head, weight):
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    head = torch.randn(13, 7, requires_grad=train_head)
    targets = torch.randint(0, 13, (2, 5))
    weights = torch.zeros(2, 5)
    weights[0, 1] = weight
    weights[1, 3] = weight*2
    actual = projection_loss(hidden, head, targets, weights, token_budget=1, compute_dtype=torch.float32)
    expected = (F.cross_entropy(F.linear(hidden, head).flatten(0, 1), targets.flatten(), reduction="none")*weights.flatten()).sum()
    params = [hidden]+([head] if train_head else [])
    expected_grads = torch.autograd.grad(expected, params)
    actual_grads = torch.autograd.grad(actual, params)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_head_gradient_uses_fp32_accumulation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    hidden = torch.randn(2, 7, 8, dtype=torch.bfloat16, device=device, requires_grad=True)
    head = torch.randn(19, 8, dtype=torch.float32, device=device, requires_grad=True)
    targets = torch.randint(0, 19, (2, 7), device=device)
    weights = torch.rand(2, 7, device=device)
    actual = projection_loss(hidden, head, targets, weights, token_budget=2, compute_dtype=torch.bfloat16)
    actual.backward()
    x = hidden.detach().flatten(0, 1)
    logits = F.linear(x, head.detach().bfloat16()).float()
    derivative = logits.softmax(-1)
    derivative[torch.arange(len(x), device=device), targets.flatten()] -= 1
    derivative = (derivative*weights.flatten()[:, None]).bfloat16()
    expected = derivative.float().T @ x.float()
    torch.testing.assert_close(head.grad, expected, rtol=1e-6, atol=1e-6)
    assert not torch.equal(head.grad, expected.bfloat16().float())


@pytest.mark.parametrize("rank", [0, 4])
def test_tiled_mlp_forward_and_all_gradients(tiny_config, rank):
    runtime = Runtime(rank=rank, base_dtype=torch.float32, train_dtype=torch.float32,
                      compute_dtype=torch.float32, mlp_tile=3)
    module = MLP(tiny_config, runtime, "cpu")
    for name, p in module.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=.03)
    control = copy.deepcopy(module)
    control.tile = 0
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    actual, expected = module(x), control(y)
    torch.testing.assert_close(actual, expected)
    signal = torch.randn_like(actual)
    actual.backward(signal)
    expected.backward(signal)
    torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)
    for (_, a), (_, b) in zip(module.named_parameters(), control.named_parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("group", [1, 2])
def test_checkpoint_retains_frozen_input_adapter_gradients(tiny_config, group):
    runtime = Runtime(rank=4, base_dtype=torch.float32, compute_dtype=torch.float32,
                      attention="vanilla", group_size=group, mlp_tile=3, projection_tile=4)
    model = CausalLM(tiny_config, runtime)
    data = batch([[2, 3, 4, 5, 6]])
    model(data).square().sum().backward()
    for name, p in model.named_parameters():
        if ".lora_B" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        elif ".lora_A" in name:
            assert p.grad is not None and not p.grad.any(), name
        else:
            assert p.grad is None, name


def test_flash_mqa_and_bidirectional_packed_groups():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("flash_attn")
    from causal_tuner.kernels.attention import flash_dense
    from causal_tuner.kernels.reference import dense_attention as dense_reference
    q = torch.randn(2, 65, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 65, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    segments = torch.tensor([[1]*65, [0]*3+[1]*30+[2]*32], device="cuda")
    spans = ((0, 0, 65), (1, 3, 33), (1, 33, 65))
    ends = torch.arange(65, device="cuda").expand(2, -1).clone()
    ends[0, 20:40] = 39
    for attention_end in (None, ends):
        actual = flash_dense(q, k, v, spans, attention_end)
        expected = dense_reference(q, k, v, segments, attention_end)
        torch.testing.assert_close(actual, expected, rtol=.03, atol=.03)
        go = torch.randn_like(q)
        for a, e in zip(torch.autograd.grad(actual, (q, k, v), go), torch.autograd.grad(expected, (q, k, v), go)):
            torch.testing.assert_close(a, e, rtol=.05, atol=.05)


def test_optional_cce_frozen_head_gradient():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("cut_cross_entropy")
    x = torch.randn(1, 13, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(257, 128, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 257, (1, 13), device="cuda")
    weights = torch.rand(1, 13, device="cuda")
    weights[:, ::3] = 0
    actual = projection_loss(x, w, targets, weights, implementation="cut", cut_chunk_size=4)
    oracle = (F.cross_entropy(F.linear(x.float(), w.float()).flatten(0, 1), targets.flatten(), reduction="none")*weights.flatten()).sum()
    torch.testing.assert_close(actual, oracle, rtol=.03, atol=.03)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(oracle, x)[0], rtol=.05, atol=.05)
