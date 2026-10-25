"""Checkpoint work elimination, owned residuals, and attention layout dispatch."""

import copy
import sys
import types
import pytest
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from causal_tuner.modeling.model import MLP, CausalLM, Runtime
from causal_tuner.modeling.lora import Projection


@pytest.fixture(autouse=True)
def cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")


@pytest.mark.parametrize("downstream,early_stop,expected_calls", [(False, True, 1), (True, True, 2), (False, False, 2)])
def test_checkpoint_skips_only_unneeded_mlp_forward(tiny_config, monkeypatch, downstream, early_stop, expected_calls):
    import causal_tuner.kernels.mlp_cuda as kernel
    module = MLP(tiny_config, Runtime(rank=4, mlp_tile=5), "cuda")
    for name, parameter in module.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=.1)
    reference = copy.deepcopy(module)
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    calls = 0
    original = kernel._forward_into
    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(kernel, "_forward_into", counted)
    def function(value):
        output = module(value, residual=value)
        return output.sin() if downstream else output
    with set_checkpoint_early_stop(early_stop):
        actual = checkpoint(function, x, use_reentrant=False, preserve_rng_state=False)
    signal = torch.randn_like(actual)
    actual.backward(signal)
    assert calls == expected_calls
    expected = reference(y)+y
    if downstream:
        expected = expected.sin()
    expected.backward(signal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for left, right in zip(module.parameters(), reference.parameters()):
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("group,inner,expected_calls,attention_calls", [(1, True, 4, 8), (2, True, 6, 10), (2, False, 6, 8)])
def test_nested_model_recompute_count(tiny_config, monkeypatch, group, inner, expected_calls, attention_calls):
    from dataclasses import replace
    from conftest import batch
    import causal_tuner.kernels.mlp_cuda as kernel
    from causal_tuner.modeling.model import Attention
    model = CausalLM(replace(tiny_config, num_hidden_layers=4),
                     Runtime(rank=4, attention="vanilla", mlp_tile=5, group_size=group, inner_checkpoint=inner), "cuda")
    reference = copy.deepcopy(model)
    reference.runtime.checkpoint = False
    calls = 0
    attention_count = 0
    original_attention = Attention.forward
    def counted_attention(*args, **kwargs):
        nonlocal attention_count
        attention_count += 1
        return original_attention(*args, **kwargs)
    monkeypatch.setattr(Attention, "forward", counted_attention)
    original = kernel._forward_into
    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(kernel, "_forward_into", counted)
    data = {k: v.cuda() for k, v in batch([[2, 3, 4, 5, 6, 7, 8]]).items()}
    actual = model(data)
    signal = torch.randn_like(actual)
    actual.backward(signal)
    assert calls == expected_calls
    assert attention_count == attention_calls
    expected = reference(data)
    expected.backward(signal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(model.parameters(), reference.parameters()):
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_training_norms_preserve_nested_offload_gradients(tiny_config, args, dtype):
    from conftest import batch
    from causal_tuner.training.engine import Trainer, make_optimizer
    from causal_tuner.training.memory import ActivationOffload
    runtime = Runtime(rank=0, attention="vanilla", base_dtype=dtype, compute_dtype=dtype, group_size=2, mlp_tile=5)
    model = CausalLM(tiny_config, runtime, "cuda")
    reference = copy.deepcopy(model)
    reference.runtime.checkpoint = False
    model.offload = ActivationOffload(4*1024**3, 1024**3, 1024**2, gpu_budget=4*1024**3)
    data = [batch([[2, 3, 4, 5]], [[0., .25, .5, 1.]]), batch([[6, 7, 8, 9]], [[0., 1., 0., .3]])]
    try:
        actual = Trainer(model, make_optimizer(model, args), args, device="cuda").update(data, step=0, total_steps=2)
        expected = Trainer(reference, make_optimizer(reference, args), args, device="cuda").update(data, step=0, total_steps=2)
        assert actual["loss"] == expected["loss"]
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        model.offload.close()


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("alias", [False, True])
def test_owned_residual_preserves_inputs_and_repeated_gradients(tiny_config, kind, alias):
    from causal_tuner.kernels.linear import project_group
    module = Projection(32, 32, rank=4, device="cuda") if kind == "linear" else MLP(tiny_config, Runtime(rank=4, mlp_tile=5), "cuda")
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = x if alias else torch.randn_like(x, requires_grad=True)
    saved_x, saved_residual = x.detach().clone(), residual.detach().clone()
    def forward(residual):
        return project_group((module,), x, 5, residual=residual)[0] if kind == "linear" else module(x, residual=residual)
    actual, expected = forward(residual), forward(None)+residual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    signal = torch.randn_like(actual)
    original_signal = signal.clone()
    parameters = (x, *(() if alias else (residual,)), *(p for p in module.parameters() if p.requires_grad))
    first = torch.autograd.grad(actual, parameters, signal, retain_graph=True)
    second = torch.autograd.grad(actual, parameters, signal)
    oracle = torch.autograd.grad(expected, parameters, signal)
    for a, b, c in zip(first, second, oracle):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, saved_residual, rtol=0, atol=0)
    torch.testing.assert_close(signal, original_signal, rtol=0, atol=0)


def test_projection_backward_omits_unused_nf4_base(monkeypatch):
    pytest.importorskip("bitsandbytes")
    import causal_tuner.kernels.linear as kernel
    module = Projection(32, 32, rank=4, device="cuda")
    module.install_nf4(module.weight.detach().clone(), "cuda")
    torch.nn.init.normal_(module.lora_B, std=.1)
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16)
    actual = kernel.project_group((module,), x, 5)[0]
    def forbidden(*args):
        raise AssertionError("Adapter-only backward must not dequantize the frozen base")
    monkeypatch.setattr(kernel, "materialize", forbidden)
    actual.sum().backward()
    assert module.lora_A.grad.abs().sum() > 0 and module.lora_B.grad.abs().sum() > 0


@pytest.mark.parametrize("layout", ["full", "packed", "gaps"])
def test_flash_layout_dispatch_with_differentiable_stub(monkeypatch, layout):
    from causal_tuner.kernels.attention import flash_dense
    from causal_tuner.kernels.reference import dense_attention as dense_reference
    q = torch.randn(2, 7, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 7, 1, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    spans = {"full": ((0, 0, 7), (1, 0, 7)),
             "packed": ((0, 0, 3), (0, 3, 7), (1, 0, 5), (1, 5, 7)),
             "gaps": ((0, 1, 3), (0, 4, 7), (1, 0, 5))}[layout]
    segments = torch.zeros(2, 7, device="cuda", dtype=torch.long)
    for index, (row, start, end) in enumerate(spans, 1):
        segments[row, start:end] = index
    recorded = []
    def fixed(qi, ki, vi, **kwargs):
        assert layout == "full" and kwargs == {"dropout_p": 0., "causal": True}
        assert all(left.data_ptr() == right.data_ptr() for left, right in zip((qi, ki, vi), (q, k, v)))
        output = dense_reference(qi, ki, vi, torch.ones(2, 7, device="cuda"))
        recorded.append(output)
        return output
    def varlen(qi, ki, vi, cq, ck, maxq, maxk, **kwargs):
        assert layout != "full"
        assert torch.equal(cq, ck) and maxq == maxk == max(end-start for _, start, end in spans)
        if layout == "packed":
            assert all(left.data_ptr() == right.data_ptr() for left, right in zip((qi, ki, vi), (q, k, v)))
        boundaries = cq.tolist()
        outputs = [dense_reference(qi[a:b][None], ki[a:b][None], vi[a:b][None], torch.ones(1, b-a, device="cuda"))[0]
                   for a, b in zip(boundaries, boundaries[1:])]
        output = torch.cat(outputs)
        recorded.append(output)
        return output
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_func=fixed, flash_attn_varlen_func=varlen))
    actual = flash_dense(q, k, v, spans)
    if layout != "gaps":
        assert actual.data_ptr() == recorded[0].data_ptr()
    expected = dense_reference(q, k, v, segments)
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    signal = torch.randn_like(actual)
    for a, e in zip(torch.autograd.grad(actual, (q, k, v), signal), torch.autograd.grad(expected, (q, k, v), signal)):
        torch.testing.assert_close(a, e, rtol=.03, atol=.03)


@pytest.mark.parametrize("mode", ["packed", "groups"])
@pytest.mark.parametrize("group_size", [1, 2])
def test_flash_metadata_survives_layer_and_checkpoint_reuse(tiny_config, monkeypatch, mode, group_size):
    from dataclasses import replace
    from conftest import batch
    import causal_tuner.modeling.model as model_code
    from causal_tuner.kernels.reference import _masked_attention
    model = CausalLM(replace(tiny_config, num_hidden_layers=2),
                     Runtime(rank=4, mlp_tile=3, group_size=group_size), "cuda")
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=.1)
    reference = copy.deepcopy(model)
    reference.runtime.attention = "vanilla"
    data = {k: v.cuda() for k, v in batch([[2, 3, 4, 5, 6, 7, 8, 9, 10]],
                                        segments=[[1, 1, 1, 2, 2, 2, 2, 3, 3]]).items()}
    if mode == "groups":
        data["attention_end"] = torch.tensor([[0, 2, 2, 3, 5, 5, 6, 7, 8]], device="cuda")
    prepared, used, offsets = [], [], []
    original_prepare, original_flash = model_code.prepare_dense_layout, model_code.flash_dense
    def prepare(*args):
        result = original_prepare(*args)
        prepared.append(result)
        return result
    def flash(*args, **kwargs):
        used.append(kwargs["layout"])
        return original_flash(*args, **kwargs)
    def fixed(q, k, v, *, dropout_p, causal):
        assert dropout_p == 0
        queries, keys = q.shape[1], k.shape[1]
        allowed = (torch.arange(keys, device=q.device)[None, :] <=
                   torch.arange(queries, device=q.device)[:, None] + keys-queries)
        if not causal:
            allowed.fill_(True)
        return _masked_attention(q, k, v, allowed[None, None])
    def varlen(q, k, v, cq, ck, maxq, maxk, **kwargs):
        assert cq is ck and maxq == maxk == 4
        offsets.append(cq.data_ptr())
        boundaries = cq.tolist()
        return torch.cat([fixed(q[a:b][None], k[a:b][None], v[a:b][None], **kwargs)[0]
                          for a, b in zip(boundaries, boundaries[1:])])
    monkeypatch.setattr(model_code, "prepare_dense_layout", prepare)
    monkeypatch.setattr(model_code, "flash_dense", flash)
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_func=fixed, flash_attn_varlen_func=varlen))
    actual = model(data)
    signal = torch.randn_like(actual)
    actual.backward(signal)
    expected = reference(data)
    expected.backward(signal)
    assert len(prepared) == 1 and prepared[0].mode == mode
    assert len(used) >= 4 and all(item is prepared[0] for item in used)
    if mode == "packed":
        assert offsets and set(offsets) == {prepared[0].offsets.data_ptr()}
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    for left, right in zip(model.parameters(), reference.parameters()):
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=.03, atol=.03)


def test_cce_filter_against_unfiltered_oracle(monkeypatch):
    cce = pytest.importorskip("cut_cross_entropy")
    from causal_tuner.kernels.loss import projection_loss
    x = torch.randn(1, 9, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8192, 128, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(8192, (1, 9), device="cuda")
    weights = torch.tensor([[0., .1, .5, 1., 2., .2, 0., .7, 1.]], device="cuda")
    modes = []
    original = cce.linear_cross_entropy
    def record(*args, **kwargs):
        modes.append(kwargs["filter_eps"])
        return original(*args, **kwargs)
    monkeypatch.setattr(cce, "linear_cross_entropy", record)
    approximate = projection_loss(x, weight, targets, weights, implementation="cut", cut_chunk_size=4)
    assert modes == ["auto", "auto"]
    per_token = original(x.flatten(0, 1), weight, targets.flatten(), reduction="none", filter_eps=None, impl="cce")
    exact = (per_token*weights.flatten()).sum()
    torch.testing.assert_close(approximate, exact, rtol=2e-5, atol=2e-5)
    ag, eg = torch.autograd.grad(approximate, x)[0].float(), torch.autograd.grad(exact, x)[0].float()
    assert (ag-eg).norm()/eg.norm() < .02
    zero = projection_loss(x, weight, targets, torch.zeros_like(weights), implementation="cut")
    assert modes == ["auto", "auto"]
    assert zero.item() == 0 and not torch.autograd.grad(zero, x)[0].any()
