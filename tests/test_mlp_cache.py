import pytest
import torch

from causal_tuner.modeling.model import CausalLM, MLP, Runtime
from conftest import batch


@pytest.fixture(autouse=True)
def cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tile", [0, 5])
def test_saved_mlp_matches_recompute_and_preserves_repeated_backward(tiny_config, monkeypatch, dtype, tile):
    from causal_tuner.kernels import mlp_cuda
    module = MLP(tiny_config, Runtime(rank=4, base_dtype=dtype, compute_dtype=dtype, mlp_tile=tile), "cuda")
    for name, parameter in module.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=.05)
    x = torch.randn(2, 7, 32, device="cuda", dtype=dtype, requires_grad=True)
    original = x.detach().clone()
    signal = torch.randn_like(x)
    params = (x, *(p for p in module.parameters() if p.requires_grad))
    expected = module(x, residual=x)
    reference = torch.autograd.grad(expected, params, signal)
    module.cache_intermediates = True
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        actual = module(x, residual=x)
    allowed = {x.data_ptr(), *(p.data_ptr() for p in module.parameters())}
    retained = [t for t in saved if t.data_ptr() not in allowed]
    assert sum(t.numel()*t.element_size() for t in retained) == module.cache_bytes(14)
    snapshots = [t.clone() for t in retained]
    def forbidden(*a, **kw):
        raise AssertionError("Cached backward must not recompute gate/up projections")
    monkeypatch.setattr(mlp_cuda, "forward_linear", forbidden)
    gradients = torch.autograd.grad(actual, params, signal, retain_graph=True)
    repeated = torch.autograd.grad(actual, params, signal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, expected_grad, repeat in zip(gradients, reference, repeated):
        torch.testing.assert_close(value, expected_grad, rtol=0, atol=0)
        torch.testing.assert_close(value, repeat, rtol=0, atol=0)
    assert all(g.dtype == torch.float32 for g in gradients[1:])
    for value, snapshot in zip(retained, snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_model_cache_selection_follows_actual_tokens_and_budget(tiny_config):
    runtime = Runtime(rank=4, attention="vanilla", checkpoint=False)
    model = CausalLM(tiny_config, runtime, "cuda")
    runtime.mlp_cache_budget_bytes = model.model.layers[0].mlp.cache_bytes(7)
    data = {key: value.cuda() for key, value in batch([[2]*7]).items()}
    model(data).sum().backward()
    assert [layer.mlp.cache_intermediates for layer in model.model.layers] == [False, True]
    longer = {key: value.cuda() for key, value in batch([[2]*14]).items()}
    model(longer).sum().backward()
    assert not any(layer.mlp.cache_intermediates for layer in model.model.layers)
    with torch.no_grad():
        model(data)
    assert not any(layer.mlp.cache_intermediates for layer in model.model.layers)
