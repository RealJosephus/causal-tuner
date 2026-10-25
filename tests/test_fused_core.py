import pytest
import torch
from causal_tuner.modeling.model import RMSNorm
from causal_tuner.modeling.rotary import rotary_values, rotate


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [32, 4096])
def test_fused_rms_matches_torch_forward_backward(dtype, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    x = torch.randn(2, 17, width, device="cuda", dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    norm = RMSNorm(width, 1e-6, dtype, "cuda")
    torch.nn.init.uniform_(norm.weight, .5, 1.5)
    norm.weight.requires_grad_(False)
    expected = ((reference.float()*torch.rsqrt(reference.float().square().mean(-1, keepdim=True)+1e-6)).to(dtype)*norm.weight)
    actual = norm(x)
    tolerance = .02 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    go = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(actual, x, go)[0], torch.autograd.grad(expected, reference, go)[0], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype,weight_dtype", [(torch.float32, torch.float32), (torch.bfloat16, torch.float32),
                                               (torch.bfloat16, torch.bfloat16)])
@pytest.mark.parametrize("width", [33, 4096])
@pytest.mark.parametrize("train_input", [False, True])
def test_trainable_rms_bounded_fp32_weight_reduction(dtype, weight_dtype, width, train_input, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    x = torch.randn(257, width, device="cuda", dtype=dtype, requires_grad=train_input)
    oracle_x = x.detach().clone().requires_grad_(train_input)
    norm = RMSNorm(width, 1e-6, weight_dtype, "cuda")
    with torch.no_grad():
        norm.weight.uniform_(.5, 1.5)
        norm.weight[:3] = torch.tensor([0., -1., 2.], device="cuda")
    oracle_w = norm.weight.detach().float().requires_grad_(True)
    rounded_w = oracle_w+(oracle_w.to(dtype).float()-oracle_w).detach()
    normalized = (oracle_x.float()*torch.rsqrt(oracle_x.float().square().mean(-1, keepdim=True)+1e-6)).to(dtype)
    expected = (normalized.float()*rounded_w).to(dtype)
    actual = norm(x, reuse_grad=True)
    upstream = torch.randn_like(actual.T).contiguous().T
    original = upstream.clone()
    tolerance = dict(rtol=.02, atol=.04) if dtype == torch.bfloat16 else dict(rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(actual, expected, **tolerance)
    empty, allocations = torch.empty, []
    def checked(shape, *args, **kwargs):
        if kwargs.get("dtype") == torch.float32:
            allocations.append(shape)
        return empty(shape, *args, **kwargs)
    monkeypatch.setattr(torch, "empty", checked)
    inputs = (x, norm.weight) if train_input else (norm.weight,)
    reference = (oracle_x, oracle_w) if train_input else (oracle_w,)
    gradients = torch.autograd.grad(actual, inputs, upstream, retain_graph=True)
    oracles = torch.autograd.grad(expected, reference, upstream)
    for result, oracle in zip(gradients, oracles):
        torch.testing.assert_close(result, oracle.to(result.dtype), **tolerance)
    for result, repeat in zip(gradients, torch.autograd.grad(actual, inputs, upstream)):
        torch.testing.assert_close(result, repeat, rtol=0, atol=0)
    torch.testing.assert_close(upstream, original, rtol=0, atol=0)
    maximum = 4*torch.cuda.get_device_properties(x.device).multi_processor_count
    assert allocations and all(shape[0] <= maximum and shape[1] == width for shape in allocations)


def test_trainable_rms_empty_input_has_zero_weight_gradient():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    norm = RMSNorm(33, 1e-6, torch.float32, "cuda")
    x = torch.empty(0, 33, device="cuda", requires_grad=True)
    dx, dw = torch.autograd.grad(norm(x).sum(), (x, norm.weight))
    assert dx.shape == x.shape and torch.count_nonzero(dw) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads", [1, 32])
def test_fused_partial_adjacent_rotary_large_positions(dtype, heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from causal_tuner.modeling.config import ModelConfig
    positions = torch.arange(34, device="cuda").reshape(2, 17)*60000
    cosine, sine = rotary_values(positions, ModelConfig(), dtype)
    x = torch.randn(2, 17, heads, 128, device="cuda", dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    even, odd = reference[..., :64:2], reference[..., 1:64:2]
    c, s = cosine.unsqueeze(-2), sine.unsqueeze(-2)
    expected = torch.cat((torch.stack((even*c-odd*s, odd*c+even*s), -1).flatten(-2), reference[..., 64:]), -1)
    actual = rotate(x, cosine, sine)
    tolerance = 0 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual[..., 64:], x[..., 64:], rtol=0, atol=0)
    go = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(actual, x, go)[0], torch.autograd.grad(expected, reference, go)[0], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_extreme_inputs_and_tail_tiles(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from causal_tuner.kernels.mlp_cuda import silu_multiply
    gate = torch.linspace(-30, 30, 7*13569, device="cuda", dtype=dtype).reshape(7, -1).requires_grad_(True)
    up = torch.randn_like(gate, requires_grad=True)
    oracle_gate, oracle_up = (value.detach().clone().requires_grad_(True) for value in (gate, up))
    expected = torch.nn.functional.silu(oracle_gate)*oracle_up
    actual = silu_multiply(gate, up)
    signal = torch.randn_like(actual.T).contiguous().T
    original_signal = signal.clone()
    tolerance = dict(rtol=.02, atol=.002) if dtype == torch.bfloat16 else dict(rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    gradients = torch.autograd.grad(actual, (gate, up), signal, retain_graph=True)
    for result, oracle in zip(gradients,
                              torch.autograd.grad(expected, (oracle_gate, oracle_up), signal)):
        torch.testing.assert_close(result, oracle, **tolerance)
    for repeat, original in zip(torch.autograd.grad(actual, (gate, up), signal), gradients):
        torch.testing.assert_close(repeat, original, rtol=0, atol=0)
    torch.testing.assert_close(signal, original_signal, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_rotary_values_preserve_full_range_argument_reduction(dtype, strided):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from causal_tuner.modeling.config import ModelConfig
    config = ModelConfig()
    positions = torch.randint(config.max_position_embeddings, (3, 1031), device="cuda")
    positions[0, :4] = torch.tensor([0, 1, 131072, config.max_position_embeddings-1], device="cuda")
    if strided:
        positions = positions[:, ::2]
    channel = torch.arange(0, config.rotary_dim, 2, device="cuda", dtype=torch.float32)
    angle = positions.float()[..., None]*(1.0/(config.rope_theta**(channel/config.rotary_dim)))
    for actual, expected in zip(rotary_values(positions, config, dtype), (angle.cos().to(dtype), angle.sin().to(dtype))):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7 if dtype == torch.float32 else 0)
