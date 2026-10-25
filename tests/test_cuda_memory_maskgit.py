import copy
import gc
import pytest
import torch
from causal_tuner.kernels.reference import dense_attention as dense_reference
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.training.memory import ActivationOffload, GIB
from causal_tuner.data.maskgit import MaskGIT
from conftest import batch


def test_dense_all_padding_has_finite_zero_gradients():
    q = torch.randn(1, 4, 3, 8, requires_grad=True)
    k, v = (torch.randn(1, 4, 1, 8, requires_grad=True) for _ in range(2))
    result = dense_reference(q, k, v, torch.zeros(1, 4))
    gradients = torch.autograd.grad(result.sum(), (q, k, v))
    assert not result.any()
    assert all(torch.isfinite(g).all() and not g.any() for g in gradients)


class ImageTokenizer:
    def get_vocab(self):
        return {"[MASK]": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<|image_0|>": 3, "<|image_1|>": 4}
    def encode(self, value):
        return [self.get_vocab()[value]]
    def decode(self, ids):
        return "".join({0: "[MASK]", 1: "<|im_start|>", 2: "<|im_end|>", 3: "<|image_0|>",
                        4: "<|image_1|>", 5: "assistant\n", 6: "\n", 7: "user\n"}[int(i)] for i in ids)


def test_maskgit_grid_and_dense_visibility():
    transform = MaskGIT(ImageTokenizer(), probability=1, seed=1)
    data = batch([[1, 5, 3, 4, 6, 4, 3, 2, 1, 7, 3, 2]])
    result = transform(data, 0)
    assert result["attention_end"].tolist()[0][2:7] == [6]*5
    assert result["input_ids"][0, 10] == 3
    selected = result["input_ids"] == 0
    assert selected.any()
    torch.testing.assert_close(result["prediction_targets"][selected], data["input_ids"][selected])
    assert not result["prediction_weights"][0, 1]
    assert result["prediction_weights"][0, 2:7].sum() == selected.sum()
    q = torch.zeros(1, 12, 1, 2)
    k = q.clone()
    v = torch.arange(12).reshape(1, 12, 1, 1).expand(-1, -1, -1, 2).float()
    output = dense_reference(q, k, v, torch.ones(1, 12), result["attention_end"])
    assert output[0, 2, 0, 0] == 3
    assert output[0, 0, 0, 0] == 0


def test_maskgit_sparse_grid_always_retains_supervision_and_replays():
    transform = MaskGIT(ImageTokenizer(), probability=1, seed=1)
    data = batch([[1, 5, 3, 4, 4, 3, 2]], [[0., 0., .25, 0., 0., 0., 0.]])
    original = {key: value.clone() for key, value in data.items()}
    for step in range(100):
        result = transform(data, step)
        assert result["prediction_weights"].sum() == .25
        assert result["input_ids"][0, 2] == transform.mask
        replayed = transform(data, step)
        for key in result:
            torch.testing.assert_close(result[key], replayed[key], rtol=0, atol=0)
    for key in data:
        torch.testing.assert_close(data[key], original[key], rtol=0, atol=0)
    data["loss_weights"].zero_()
    torch.testing.assert_close(transform(data, 0)["input_ids"], data["input_ids"], rtol=0, atol=0)


def test_accumulated_maskgit_packs_keep_sampling_and_attention_boundaries():
    from causal_tuner.data.microbatch import repack_batches
    from causal_tuner.data.segments import canonical_segments, prediction_targets, runs
    transform = MaskGIT(ImageTokenizer(), probability=1, seed=7)
    data = batch([[1, 5, 3, 4, 6, 4, 3, 2]])
    data['_spans'], data['_source_count'] = ((0, 0, 8),), 1
    sampled = [transform(data, step) for step in (4, 5, 6)]
    packed, = repack_batches(iter(sampled), 24, 1024**2)
    for key in ('input_ids', 'prediction_targets', 'prediction_weights'):
        torch.testing.assert_close(packed[key], torch.cat([part[key] for part in sampled], dim=1), rtol=0, atol=0)
    assert packed['_spans'] == ((0, 0, 8), (0, 8, 16), (0, 16, 24))
    assert runs(canonical_segments(packed['attention_mask'], packed['segment_ids'])) == packed['_spans']
    for index, part in enumerate(sampled):
        torch.testing.assert_close(packed['attention_end'][:, 8*index:8*(index+1)], part['attention_end']+8*index)
    assert prediction_targets(packed)[1].sum() == sum(prediction_targets(part)[1].sum() for part in sampled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_cpu_offload_forward_backward_and_budget(tiny_config, monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", True)
    runtime = Runtime(rank=4, base_dtype=torch.bfloat16, compute_dtype=torch.bfloat16,
                      attention="vanilla", mlp_tile=3, checkpoint=True, group_size=2)
    model = CausalLM(tiny_config, runtime, device="cuda")
    control = copy.deepcopy(model)
    model.offload = ActivationOffload(72*GIB, 1024**2, 4096)
    data = {k: v.cuda() for k, v in batch([[2, 3, 4, 5, 6]]).items()}
    output, expected = model(data), control(data)
    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    for p, q in zip(model.parameters(), control.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    del output
    gc.collect()
    assert model.offload.peak > 0 and model.offload.pinned_peak == 4096
    assert model.offload.peak == 1*5*32*2
    assert model.offload.used == 0
    tiny = ActivationOffload(72*GIB, 1, 4096)
    with pytest.raises(MemoryError, match="checkpoint budget"):
        tiny.pack(torch.ones(10, device="cuda"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.randn(129, 97, device="cuda", dtype=torch.bfloat16).T
        saved = model.offload.pack(source)
        restored = model.offload.unpack(saved)
        torch.testing.assert_close(restored, source, rtol=0, atol=0)


def test_nf4_input_gradient_matches_dequantized_base():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("bitsandbytes")
    from causal_tuner.modeling.lora import Projection
    module = Projection(64, 64, rank=4, base_dtype=torch.bfloat16, compute_dtype=torch.bfloat16, device="cuda")
    base = module.weight.detach().clone()
    module.install_nf4(base, "cuda")
    x = torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    result = module(x)
    expected = torch.nn.functional.linear(x, module.base_weight().bfloat16())
    torch.testing.assert_close(result, expected, rtol=.03, atol=.02)
    grad = torch.randn_like(result)
    actual_dx = torch.autograd.grad(result, x, grad)[0]
    expected_dx = torch.autograd.grad(expected, x, grad)[0]
    torch.testing.assert_close(actual_dx, expected_dx, rtol=.03, atol=.02)


def test_cuda_dense_training_with_nested_checkpoint(args, tiny_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    pytest.importorskip("flash_attn")
    from causal_tuner.train import runtime_from_args
    from causal_tuner.training.engine import Trainer, make_optimizer
    args.dtype = args.param_dtype = "bfloat16"
    args.attn_mechanism = "flash_attention_2"
    args.remat_group_size = 2
    args.mlp_chunk_size = 7
    args.lora_train_lm_head = True
    model = CausalLM(tiny_config, runtime_from_args(args), device="cuda")
    model.offload = ActivationOffload(72*GIB, 1024**2, 4096)
    optimizer = make_optimizer(model, args)
    trainer = Trainer(model, optimizer, args, device="cuda")
    data = batch([list(range(2, 67))])
    for step in range(2):
        metrics = trainer.update([data, data], step=step, total_steps=2)
        assert metrics["effective_weight"] == 128 and metrics["loss"] > 0
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
