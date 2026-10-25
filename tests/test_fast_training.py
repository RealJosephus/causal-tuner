"""Independent oracles for the CUDA training paths and bounded recomputation."""

import copy
import pytest
import torch
from torch.nn import functional as F
from causal_tuner.modeling.model import MLP, Runtime
from causal_tuner.modeling.lora import Projection
from causal_tuner.kernels.loss import projection_loss


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")


def plain_linear(module, x):
    dtype = module.compute_dtype
    x = x.to(dtype)
    weight = module.base_weight().to(dtype)
    bias = None if module.bias is None else module.bias.to(dtype)
    return F.linear(x, weight, bias)+F.linear(F.linear(x, module.lora_A.to(dtype)), module.lora_B.to(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_input", [False, True])
@pytest.mark.parametrize("mixed_ranks", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_fused_tiled_mlp_independent_gradients(tiny_config, dtype, train_input, mixed_ranks, cache):
    require_cuda()
    module = MLP(tiny_config, Runtime(rank=4, base_dtype=dtype, compute_dtype=dtype, mlp_tile=5), "cuda")
    module.cache_intermediates = cache
    if mixed_ranks:
        for name, rank in (("gate_proj", 3), ("up_proj", 5), ("down_proj", 7)):
            original = getattr(module, name)
            setattr(module, name, Projection(original.weight.shape[1], original.weight.shape[0], rank=rank,
                                             base_dtype=dtype, compute_dtype=dtype, device="cuda"))
    for name, parameter in module.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=.1)
    reference = copy.deepcopy(module)
    x = torch.randn(2, 7, 32, device="cuda", dtype=dtype, requires_grad=train_input)
    y = x.detach().clone().requires_grad_(train_input)
    actual = module(x)
    expected = plain_linear(reference.down_proj, F.silu(plain_linear(reference.gate_proj, y))*plain_linear(reference.up_proj, y))
    tolerance = dict(rtol=.04, atol=.003) if dtype == torch.bfloat16 else dict(rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    signal = torch.randn_like(actual)
    actual.backward(signal)
    expected.backward(signal)
    if train_input:
        torch.testing.assert_close(x.grad, y.grad, **tolerance)
    for (name, a), (_, b) in zip(module.named_parameters(), reference.named_parameters()):
        if a.requires_grad:
            assert a.grad.dtype == torch.float32, name
            torch.testing.assert_close(a.grad, b.grad, **tolerance)


@pytest.mark.parametrize("kind", ["mlp", "projection"])
def test_full_width_bf16_tiles_against_independent_autograd(kind):
    require_cuda()
    from causal_tuner.modeling.config import ModelConfig
    from causal_tuner.kernels.linear import project_group
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    module = (MLP(ModelConfig(), Runtime(rank=256, mlp_tile=4096), "cuda") if kind == "mlp"
              else Projection(4096, 4096, rank=256, device="cuda"))
    for name, parameter in module.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=.005)
    x = torch.randn(1, 4099, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    saved = x.detach().clone()
    y = x.detach().clone().requires_grad_(True)
    actual = module(x) if kind == "mlp" else project_group((module,), x, 0)[0]
    expected = (plain_linear(module.down_proj, F.silu(plain_linear(module.gate_proj, y))*plain_linear(module.up_proj, y))
                if kind == "mlp" else plain_linear(module, y))
    parameters = tuple(p for p in module.parameters() if p.requires_grad)
    signal = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (x, *parameters), signal, retain_graph=True)
    repeated = torch.autograd.grad(actual, (x, *parameters), signal)
    reference = torch.autograd.grad(expected, (y, *parameters), signal)
    # Different GEMM shapes and fused additions move BF16 rounding points.
    # Bound the whole output/gradient error rather than relative error at zeros.
    for left, right in zip((actual, *gradients), (expected, *reference)):
        left, right = left.float(), right.float()
        assert torch.isfinite(left).all()
        assert (left-right).norm()/right.norm().clamp_min(1e-20) < .02
    for left, right in zip(gradients, repeated):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(gradient.dtype == torch.float32 for gradient in gradients[1:])
    torch.testing.assert_close(x, saved, rtol=0, atol=0)


@pytest.mark.parametrize("ranks", [(4, 4, 4), (3, 5, 7)])
def test_shared_qkv_with_bias_and_fp32_adapter_accumulation(ranks):
    require_cuda()
    from causal_tuner.kernels.linear import project_group
    modules = [Projection(32, out, bias=True, rank=rank, compute_dtype=torch.bfloat16, device="cuda")
               for out, rank in zip((32, 8, 8), ranks)]
    for module in modules:
        torch.nn.init.normal_(module.lora_B, std=.1)
        torch.nn.init.normal_(module.bias, std=.1)
    reference = copy.deepcopy(modules)
    x = torch.randn(2, 9, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    outputs = project_group(modules, x, tile=4)
    targets = [plain_linear(module, y) for module in reference]
    signals = [torch.randn_like(out) for out in outputs]
    torch.autograd.backward(outputs, signals)
    torch.autograd.backward(targets, signals)
    torch.testing.assert_close(x.grad, y.grad, rtol=.05, atol=.02)
    for index, (module, oracle) in enumerate(zip(modules, reference)):
        torch.testing.assert_close(outputs[index], targets[index], rtol=.03, atol=.02)
        dy = signals[index].flatten(0, 1)
        expected_b = dy.float().T @ (x.detach().flatten(0, 1) @ module.lora_A.detach().bfloat16().T).float()
        torch.testing.assert_close(module.lora_B.grad, expected_b, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(module.lora_A.grad, oracle.lora_A.grad, rtol=.05, atol=.02)
        assert module.weight.grad is module.bias.grad is None


@pytest.mark.parametrize("vocab", [97, 185600])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_loss_large_vocabulary_and_scaled_gradient(vocab, dtype):
    require_cuda()
    x = torch.randn(1, 7, 32, device="cuda", dtype=dtype, requires_grad=True)
    head = torch.randn(vocab, 32, device="cuda", dtype=torch.float32, requires_grad=True)*.1
    head.retain_grad()
    labels = torch.randint(vocab, (1, 7), device="cuda")
    weights = torch.tensor([[0, .1, 1, .5, 0, 2, .2]], device="cuda")
    actual = projection_loss(x, head, labels, weights, token_budget=3, compute_dtype=dtype)
    logits = F.linear(x.detach(), head.detach().to(dtype)).float().flatten(0, 1)
    expected = (F.cross_entropy(logits, labels.flatten(), reduction="none")*weights.flatten()).sum()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-5)
    actual.mul(2.5).backward()
    derivative = logits.softmax(-1)
    derivative[torch.arange(7, device="cuda"), labels.flatten()] -= 1
    derivative = (derivative*weights.flatten()[:, None]).to(dtype)
    expected_head = (derivative.float().T @ x.detach().flatten(0, 1).float())*2.5
    torch.testing.assert_close(head.grad, expected_head, rtol=.01, atol=1e-4)


def test_fused_mlp_saves_only_boundary_and_weights(tiny_config, monkeypatch):
    require_cuda()
    module = MLP(tiny_config, Runtime(rank=4, mlp_tile=5), "cuda")
    x = torch.randn(2, 13, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    saved = []
    def pack(tensor):
        saved.append(tensor)
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = module(x)
    allowed = {x.data_ptr(), *(parameter.data_ptr() for parameter in module.parameters())}
    assert all(tensor.data_ptr() in allowed for tensor in saved)
    def forbidden(*args, **kwargs):
        raise AssertionError("The fused MLP backward must not build per-tile autograd graphs")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    output.sum().backward()


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inner", [True, False])
def test_async_offload_prefetch_and_nested_gradients(tiny_config, group, dtype, inner):
    require_cuda()
    from dataclasses import replace
    from causal_tuner.modeling.model import CausalLM
    from causal_tuner.training.memory import ActivationOffload, GIB
    from conftest import batch
    runtime = Runtime(rank=4, attention="vanilla", group_size=group, mlp_tile=5, compute_dtype=dtype,
                      inner_checkpoint=inner)
    model = CausalLM(replace(tiny_config, num_hidden_layers=4), runtime, "cuda")
    reference = copy.deepcopy(model)
    model.offload = ActivationOffload(72*GIB, 1024**2, 4096)
    data = {key: value.cuda() for key, value in batch([[2, 3, 4, 5, 6, 7, 8]]).items()}
    try:
        actual, expected = model(data), reference(data)
        signal = torch.randn_like(actual)
        actual.backward(signal)
        expected.backward(signal)
        model.offload.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for left, right in zip(model.parameters(), reference.parameters()):
            if left.requires_grad:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        assert model.offload.stats["prefetch_hits"] == 4//group-1
        assert model.offload.pinned_peak <= 4096
        assert model.offload.peak == (4//group)*7*32*torch.empty((), dtype=dtype).element_size()
    finally:
        model.offload.close()


def test_offload_pack_returns_before_copy_and_retains_source(monkeypatch):
    require_cuda()
    import gc
    import threading
    from causal_tuner.training.memory import ActivationOffload, GIB
    offload = ActivationOffload(72*GIB, 1024**2, 4096, gpu_budget=0)
    recorded = []
    original_record = torch.cuda.Event.record
    def record(event, stream=None):
        if stream is offload.stream:
            recorded.append(event)
        return original_record(event, stream)
    monkeypatch.setattr(torch.cuda.Event, "record", record)
    release = threading.Event()
    original = offload._transfer
    def delayed(*args, **kwargs):
        assert release.wait(timeout=10), "Test did not release the background copy"
        return original(*args, **kwargs)
    monkeypatch.setattr(offload, "_transfer", delayed)
    source = torch.arange(35*37, device="cuda", dtype=torch.float32).reshape(35, 37).T
    expected = source.cpu()
    try:
        saved = offload.pack(source)
        assert not saved.ready.done()
        del source
        gc.collect()
        release.set()
        torch.testing.assert_close(offload.unpack(saved).cpu(), expected, rtol=0, atol=0)
        second = offload.pack(expected.cuda())
        third = offload.pack(expected.cuda())
        offload.unpack(third)
        assert offload.stats["prefetch_skipped_budget"] == 1
        offload.unpack(second)
        assert offload.stats["prefetch_hits"] == 0
        del saved, second, third
        offload.synchronize()
        gc.collect()
        assert offload.used == 0
        assert len(recorded) > 2 and {id(event) for event in recorded} == {id(event) for event in offload.copy_events}
    finally:
        release.set()
        offload.close()


def test_nf4_fused_tiled_mlp_gradients(tiny_config):
    require_cuda()
    pytest.importorskip("bitsandbytes")
    module = MLP(tiny_config, Runtime(rank=4, mlp_tile=5), "cuda")
    for projection in (module.gate_proj, module.up_proj, module.down_proj):
        projection.install_nf4(projection.weight.detach().clone(), "cuda")
        torch.nn.init.normal_(projection.lora_B, std=.1)
    x = torch.randn(1, 13, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = module(x)
    expected = plain_linear(module.down_proj, F.silu(plain_linear(module.gate_proj, x))*plain_linear(module.up_proj, x))
    signal = torch.randn_like(actual)
    parameters = (x, *(p for p in module.parameters() if p.requires_grad))
    ag = torch.autograd.grad(actual, parameters, signal)
    eg = torch.autograd.grad(expected, parameters, signal)
    torch.testing.assert_close(actual, expected, rtol=.04, atol=.003)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, rtol=.05, atol=.003)


@pytest.mark.parametrize("supervised", [False, True])
def test_fused_loss_cached_gradients_can_be_reused(supervised):
    require_cuda()
    x = torch.randn(2, 5, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    head = torch.randn(97, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(97, (2, 5), device="cuda")
    weights = torch.full((2, 5), .1 if supervised else 0, device="cuda")
    loss = projection_loss(x, head, labels, weights, token_budget=3)
    first = torch.autograd.grad(loss, (x, head), retain_graph=True)
    second = torch.autograd.grad(loss*4, (x, head))
    for initial, repeated in zip(first, second):
        torch.testing.assert_close(repeated, initial*4, rtol=0, atol=0)
        if not supervised:
            assert not initial.any()


def test_fused_adamw_reference_bf16_state_and_resume(tmp_path, args, tiny_config):
    require_cuda()
    from causal_tuner.modeling.model import CausalLM
    from causal_tuner.train import runtime_from_args
    from causal_tuner.training.engine import Trainer, make_optimizer
    from causal_tuner.checkpointing.io import save_checkpoint, latest_checkpoint, restore_checkpoint
    from conftest import batch
    args.precision_recipe = "reference_bf16"
    args.dtype = args.param_dtype = "bfloat16"
    args.lora_train_embed = args.lora_train_lm_head = True
    model = CausalLM(tiny_config, runtime_from_args(args), "cuda")
    control = copy.deepcopy(model)
    optimizer = make_optimizer(model, args)
    trainer = Trainer(model, optimizer, args, device="cuda")
    data = batch([[2, 3, 4, 5, 6]])
    trainer.update([data], step=0, total_steps=2)
    save_checkpoint(tmp_path, model, optimizer, step=1, cursor={}, signature={}, scheduler={}, save_optimizer=True)
    trainer.update([data], step=1, total_steps=2)
    resumed = make_optimizer(control, args)
    restore_checkpoint(*latest_checkpoint(tmp_path, {}), control, resumed)
    Trainer(control, resumed, args, device="cuda").update([data], step=1, total_steps=2)
    assert optimizer.defaults["fused"]
    for left, right in zip(model.parameters(), control.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if left.requires_grad:
            assert left.grad.dtype == optimizer.state[left]["exp_avg"].dtype == torch.bfloat16
            assert optimizer.state[left]["step"].is_cuda
