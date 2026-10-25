"""Aliasing, tail tiles, and repeated backward for reusable CUDA workspaces."""

import pytest
import torch
from causal_tuner.modeling.lora import Projection
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.kernels.loss import projection_loss


@pytest.fixture(autouse=True)
def cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tile", [0, 5])
@pytest.mark.parametrize("head_dim,rotary_dim", [(8, 4), (128, 64)])
def test_projection_owned_rotary_preserves_shared_gradients(dtype, tile, head_dim, rotary_dim, monkeypatch):
    import causal_tuner.kernels.core_cuda as core
    from causal_tuner.kernels.linear import project_group
    from causal_tuner.modeling.rotary import rotate
    width = head_dim*4
    modules = [Projection(width, out, bias=True, rank=rank, base_dtype=dtype, compute_dtype=dtype, device="cuda")
               for out, rank in zip((width, head_dim, head_dim), (3, 5, 7))]
    for module in modules:
        torch.nn.init.normal_(module.lora_B, std=.03)
        torch.nn.init.normal_(module.bias, std=.1)
    x = torch.randn(2, 7, width, device="cuda", dtype=dtype, requires_grad=True)
    original_input = x.detach().clone()
    angles = torch.arange(14, device="cuda").reshape(2, 7, 1)*60000*torch.rand(rotary_dim//2, device="cuda")
    cosine, sine = angles.cos().to(dtype), angles.sin().to(dtype)
    actual = project_group(modules, x, tile, rotary=(cosine, sine, head_dim))
    raw = project_group(modules, x, tile)
    expected = tuple(rotate(t.reshape(2, 7, -1, head_dim), cosine, sine).reshape_as(t) if i < 2 else t
                     for i, t in enumerate(raw))
    tolerance = dict(rtol=0, atol=0) if dtype == torch.bfloat16 else dict(rtol=2e-6, atol=2e-6)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, **tolerance)
    for a, r in zip(actual[:2], raw[:2]):
        torch.testing.assert_close(a.reshape(2, 7, -1, head_dim)[..., rotary_dim:],
                                   r.reshape(2, 7, -1, head_dim)[..., rotary_dim:], rtol=0, atol=0)
    shared = torch.randn_like(actual[0])
    signals = (shared, shared[..., :head_dim], shared[..., :head_dim])
    saved_signal = shared.clone()
    parameters = (x, *(p for module in modules for p in module.parameters() if p.requires_grad))
    largest_tile = 0
    original_inverse = core.inverse_rope_tile
    def bounded_inverse(grad, *args):
        nonlocal largest_tile
        largest_tile = max(largest_tile, len(grad))
        return original_inverse(grad, *args)
    monkeypatch.setattr(core, "inverse_rope_tile", bounded_inverse)
    ag = torch.autograd.grad(actual, parameters, signals, retain_graph=True)
    eg = torch.autograd.grad(expected, parameters, signals)
    repeated = torch.autograd.grad(actual, parameters, signals)
    assert 0 < largest_tile <= (tile or 4096)
    for a, e, r in zip(ag, eg, repeated):
        torch.testing.assert_close(a, e, **tolerance)
        torch.testing.assert_close(a, r, rtol=0, atol=0)
    torch.testing.assert_close(shared, saved_signal, rtol=0, atol=0)
    torch.testing.assert_close(x, original_input, rtol=0, atol=0)


def test_owned_rotary_beyond_512k():
    from causal_tuner.kernels.core_cuda import rotate_owned_qk
    # One extra token also exercises the masked tail after 65,536 position tiles.
    rows, head_dim, rotary_dim = 524289, 128, 64
    query = torch.randn(rows, 3*head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(rows, head_dim, device="cuda", dtype=torch.bfloat16)
    original = (query.clone(), key.clone())
    angles = torch.randn(rows, rotary_dim//2, device="cuda")
    cosine, sine = angles.cos().bfloat16(), angles.sin().bfloat16()
    rotate_owned_qk(query, key, cosine, sine, head_dim)
    for actual, raw in zip((query, key), original):
        # An independent, bounded PyTorch oracle checks every token and head.
        for start in range(0, rows, 4096):
            source = raw[start:start+4096].reshape(-1, raw.shape[-1]//head_dim, head_dim)
            expected = source.clone()
            c, s = cosine[start:start+4096, None], sine[start:start+4096, None]
            even, odd = source[..., :rotary_dim:2], source[..., 1:rotary_dim:2]
            expected[..., :rotary_dim:2] = even*c-odd*s
            expected[..., 1:rotary_dim:2] = odd*c+even*s
            torch.testing.assert_close(actual[start:start+4096].reshape_as(expected), expected, rtol=0, atol=0)


def test_nf4_projection_rotary_input_gradient():
    pytest.importorskip("bitsandbytes")
    from causal_tuner.kernels.linear import project_group
    from causal_tuner.modeling.rotary import rotate
    modules = [Projection(32, out, bias=True, rank=4, device="cuda") for out in (32, 8, 8)]
    for module in modules:
        module.install_nf4(module.weight.detach().clone(), "cuda")
        torch.nn.init.normal_(module.lora_B, std=.1)
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    angles = torch.randn(2, 7, 2, device="cuda")
    cosine, sine = angles.cos().bfloat16(), angles.sin().bfloat16()
    actual = project_group(modules, x, 5, rotary=(cosine, sine, 8))
    raw = project_group(modules, x, 5)
    expected = tuple(rotate(t.reshape(2, 7, -1, 8), cosine, sine).reshape_as(t) if i < 2 else t
                     for i, t in enumerate(raw))
    signal = tuple(torch.randn_like(t) for t in actual)
    parameters = (x, *(p for m in modules for p in m.parameters() if p.requires_grad))
    for a, e in zip(torch.autograd.grad(actual, parameters, signal), torch.autograd.grad(expected, parameters, signal)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("reuse", [False, True])
def test_rms_gradient_ownership_and_zero_weights(reuse):
    from causal_tuner.kernels.core_cuda import RMS
    x = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    weight[:64] = 0
    signal = torch.randn_like(x)
    original = signal.clone()
    expected = torch.autograd.grad(RMS.apply(x, weight, 1e-6), x, original)[0]
    result = torch.autograd.grad(RMS.apply(x, weight, 1e-6, reuse), x, signal)[0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert (result.data_ptr() == signal.data_ptr()) == reuse
    if not reuse:
        torch.testing.assert_close(signal, original, rtol=0, atol=0)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_loss_reuses_logits_and_skips_evaluation_gradients(grad_enabled, monkeypatch):
    import causal_tuner.kernels.loss_cuda as loss_kernel
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    head = torch.randn(4099, 32, device="cuda", requires_grad=True)
    targets = torch.randint(4099, (2, 7), device="cuda")
    weights = torch.rand(2, 7, device="cuda")
    weights[0, 1] = 0
    pointers = []
    statistics, labels, factors = [], [], []
    original = loss_kernel.weighted_ce
    def checked(logits, targets, weights, *, gradient, **workspace):
        assert gradient == grad_enabled
        pointers.append(logits.data_ptr())
        statistics.append(workspace["partials"].data_ptr())
        labels.append(targets.data_ptr())
        factors.append(weights.data_ptr())
        return original(logits, targets, weights, gradient=gradient, **workspace)
    monkeypatch.setattr(loss_kernel, "weighted_ce", checked)
    with torch.set_grad_enabled(grad_enabled):
        output = projection_loss(x, head, targets, weights, token_budget=5)
    assert len(pointers) == 3 and len(set(pointers)) == 1
    assert all(len(set(values)) == 1 for values in (statistics, labels, factors))
    assert output.requires_grad == grad_enabled
    logits = torch.nn.functional.linear(x.detach(), head.detach().bfloat16()).float()
    expected = (torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")*weights.flatten()).sum()
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("reuse_hidden", [False, True])
def test_prenorm_reuse_preserves_final_norm_cached_loss_gradient(tiny_config, checkpoint, reuse_hidden):
    from conftest import batch
    model = CausalLM(tiny_config, Runtime(rank=4, attention="vanilla", mlp_tile=5,
                                        checkpoint=checkpoint, group_size=2, train_head=True), "cuda")
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=.03)
    data = {key: value.cuda() for key, value in batch([[2, 3, 4, 5, 6, 7, 8]]).items()}
    hidden = model(data)
    loss = projection_loss(hidden, model.lm_head.weight, data["input_ids"], data["loss_weights"], token_budget=3,
                           reuse_hidden=reuse_hidden)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    first = torch.autograd.grad(loss, parameters, retain_graph=True)
    second = torch.autograd.grad(loss*2, parameters)
    for initial, repeated in zip(first, second):
        torch.testing.assert_close(repeated, initial*2, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_head", [False, True])
@pytest.mark.parametrize("supervision", ["all", "mixed", "none"])
def test_loss_consumes_only_owned_final_norm_output(dtype, train_head, supervision):
    from causal_tuner.modeling.model import RMSNorm
    norm = RMSNorm(32, 1e-6, dtype, "cuda")
    norm.weight.requires_grad_(False)
    x = torch.randn(2, 7, 32, device="cuda", dtype=dtype, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    head = torch.randn(4099, 32, device="cuda", requires_grad=train_head)
    other_head = head.detach().clone().requires_grad_(train_head)
    targets = torch.randint(4099, (2, 7), device="cuda")
    weights = torch.rand(2, 7, device="cuda")
    if supervision == "mixed":
        weights[:, ::2] = 0
    elif supervision == "none":
        weights.zero_()
    hidden = norm(x)
    original = hidden.detach().clone()
    actual = projection_loss(hidden, head, targets, weights, token_budget=3, compute_dtype=dtype, reuse_hidden=True)
    cached = actual.grad_fn.saved_tensors[0]
    assert cached.data_ptr() == hidden.data_ptr()
    assert not cached[weights == 0].any()
    expected = projection_loss(norm(other), other_head, targets, weights, token_budget=3, compute_dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cached_before_backward = cached.clone()
    parameters = (x, head) if train_head else (x,)
    other_parameters = (other, other_head) if train_head else (other,)
    for a, e in zip(torch.autograd.grad(actual, parameters, retain_graph=True), torch.autograd.grad(expected, other_parameters)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    torch.testing.assert_close(cached, cached_before_backward, rtol=0, atol=0)
    scaled = torch.autograd.grad(actual*2, x)[0]
    torch.testing.assert_close(scaled, torch.autograd.grad(norm(x), x, cached_before_backward*2)[0], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(norm(x), original, rtol=0, atol=0)


def test_loss_rejects_consuming_leaf_or_view():
    x = torch.randn(2, 7, 32, device="cuda", requires_grad=True)
    head = torch.randn(17, 32, device="cuda")
    targets = torch.randint(17, (2, 7), device="cuda")
    weights = torch.ones(2, 7, device="cuda")
    for value in (x, (x*2).view_as(x), (x*2).transpose(0, 1)):
        with pytest.raises(ValueError, match="exclusive contiguous non-leaf"):
            projection_loss(value, head, targets, weights, reuse_hidden=True)


@pytest.mark.parametrize("width", [32, 4096, 4100])
def test_reconstructed_rms_shares_output_and_bounds_gradient_error(width):
    from causal_tuner.modeling.model import RMSNorm
    norm = RMSNorm(width, 1e-6, torch.bfloat16, "cuda")
    norm.weight.requires_grad_(False)
    torch.nn.init.uniform_(norm.weight, .5, 1.5)
    x = torch.randn(2, 7, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        output = norm(x, reuse_grad=True)
    assert saved[0].data_ptr() == output.data_ptr() and saved[0].data_ptr() != x.data_ptr()
    exact = norm(x)
    torch.testing.assert_close(output, exact, rtol=0, atol=0)
    signal = torch.randn_like(output)
    expected = torch.autograd.grad(exact, x, signal)[0].float()
    actual = torch.autograd.grad(output, x, signal.clone())[0].float()
    assert (actual-expected).norm()/expected.norm() < .003
    assert (actual-expected).abs().max() < .05


@pytest.mark.parametrize("bad_weight", [0., 1e-9, 1e9, float("inf"), float("nan")])
def test_reconstructed_rms_invalidates_cache_and_falls_back(bad_weight):
    from causal_tuner.modeling.model import RMSNorm
    norm = RMSNorm(32, 1e-6, torch.bfloat16, "cuda")
    norm.weight.requires_grad_(False)
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert norm._can_reconstruct(x.dtype)
    norm.weight[0] = bad_weight
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        norm(x, reuse_grad=True)
    assert not norm._can_reconstruct(x.dtype)
    assert saved[0].data_ptr() == x.data_ptr()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("blocksize", [64, 128, 4096])
def test_single_pass_nf4_matches_native_dequantization(nested, source_dtype, blocksize):
    bnb = pytest.importorskip("bitsandbytes")
    from causal_tuner.kernels.linear import materialize
    values = torch.randn(71, 96, device="cuda", dtype=source_dtype)
    packed, state = bnb.functional.quantize_4bit(values, blocksize=blocksize, compress_statistics=nested, quant_type="nf4")
    native = bnb.functional.dequantize_4bit(packed, state)
    for dtype in (torch.bfloat16, torch.float32):
        torch.testing.assert_close(materialize(packed, state, dtype), native.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gradient_first_write_ignores_uninitialized_storage(dtype):
    from causal_tuner.kernels.linear import accumulate
    left = torch.randn(19, 37, device="cuda", dtype=dtype)
    right = torch.randn(37, 23, device="cuda", dtype=dtype)
    output = torch.full((19, 23), float("nan"), device="cuda")
    expected = left.float() @ right.float()
    accumulate(output, left, right, add=False)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    accumulate(output, left, right, add=True)
    torch.testing.assert_close(output, expected*2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("vocab", [97, 4096, 4097, 185600])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_ce_target_boundaries_and_logit_ownership(vocab, dtype):
    from causal_tuner.kernels.loss_cuda import weighted_ce
    logits = (torch.randn(5, vocab, device="cuda")*4).to(dtype)
    targets = torch.tensor([0, min(4095, vocab-1), min(4096, vocab-1), vocab//2, vocab-1], device="cuda")
    weights = torch.tensor([0, .125, 1, 2, .5], device="cuda")
    reference = logits.float().requires_grad_(True)
    loss = (torch.nn.functional.cross_entropy(reference, targets, reduction="none")*weights).sum()
    expected_grad, = torch.autograd.grad(loss, reference)
    for gradient in (False, True):
        workspace = logits.clone()
        actual = weighted_ce(workspace, targets, weights, gradient=gradient).sum()
        torch.testing.assert_close(actual, loss, rtol=2e-6, atol=1e-5)
        if gradient:
            tolerance = .008 if dtype == torch.bfloat16 else 2e-6
            torch.testing.assert_close(workspace, expected_grad.to(dtype), rtol=tolerance, atol=1e-7)
        else:
            torch.testing.assert_close(workspace, logits, rtol=0, atol=0)


def test_cached_kernel_specialization_streams_and_launch_hooks(monkeypatch):
    import importlib
    import weakref
    import triton
    from causal_tuner.kernels.mlp_cuda import _activation, activation
    dispatch = importlib.import_module("causal_tuner.kernels.launch")
    monkeypatch.setattr(dispatch, "_CACHE", {})
    calls = []
    original = _activation.run
    def traced(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(_activation, "run", traced)
    stream = torch.cuda.Stream()
    for dtype in (torch.float32, torch.bfloat16):
        for offset in (0, 1):
            for size in (257, 259):
                gate = torch.randn(size+offset, device="cuda", dtype=dtype)[offset:]
                up = torch.randn_like(gate)
                activation(gate, up)
                count = len(calls)
                gate = gate.clone() if offset == 0 else torch.cat((gate[:1], gate))[1:]
                up = up.neg()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    output = activation(gate, up)
                    expected = torch.nn.functional.silu(gate)*up
                torch.cuda.current_stream().wait_stream(stream)
                torch.testing.assert_close(output, expected, rtol=.008 if dtype == torch.bfloat16 else 2e-6, atol=1e-6)
                assert len(calls) == count
    # Enabling a profiler after the cache is warm must still observe the active stream.
    streams = []
    def observed(metadata):
        streams.append(metadata.get()["stream"])
    with monkeypatch.context() as hooks:
        hooks.setattr(triton.knobs.runtime, "launch_enter_hook", observed)
        with torch.cuda.stream(stream):
            activation(gate, up)
        assert streams == [stream.cuda_stream]
    before = len(calls)
    pre_runs = []
    monkeypatch.setattr(_activation, "pre_run_hooks", [lambda *args, **kwargs: pre_runs.append(1)])
    activation(gate, up)
    assert pre_runs == [1] and len(calls) == before+1
    references = weakref.ref(gate), weakref.ref(up)
    del gate, up
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("kind", ["mlp", "projection"])
def test_empty_tiled_inputs_return_zero_parameter_gradients(tiny_config, kind):
    from causal_tuner.modeling.model import MLP
    from causal_tuner.kernels.linear import project_group
    module = MLP(tiny_config, Runtime(rank=4, mlp_tile=5), "cuda") if kind == "mlp" else Projection(32, 32, rank=4, device="cuda")
    x = torch.empty(2, 0, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = module(x) if kind == "mlp" else project_group((module,), x, 5)[0]
    gradients = torch.autograd.grad(output.sum(), (x, *(p for p in module.parameters() if p.requires_grad)))
    assert gradients[0].shape == x.shape
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


def test_zero_supervision_skips_head_cast_and_projection(monkeypatch):
    hidden = torch.randn(2, 7, 32, device="cuda", requires_grad=True)
    head = torch.randn(97, 32, device="cuda", requires_grad=True)
    original_to = torch.Tensor.to
    def checked_to(tensor, *args, **kwargs):
        assert tensor is not head, "An unsupervised batch must not materialize the head"
        return original_to(tensor, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("An unsupervised batch must not project logits")
    monkeypatch.setattr(torch.Tensor, "to", checked_to)
    monkeypatch.setattr(torch, "mm", forbidden)
    loss = projection_loss(hidden, head, torch.full((2, 7), -100, device="cuda"), torch.zeros(2, 7, device="cuda"))
    gradients = torch.autograd.grad(loss, (hidden, head))
    assert loss == 0 and all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("layout", ["all", "interval", "gaps", "empty"])
def test_host_loss_rows_preserve_weighted_gradients_and_avoid_dense_gathers(dtype, reuse, layout, monkeypatch):
    from causal_tuner.kernels.loss import active_rows
    values = torch.randn(17, 32, device="cuda", dtype=dtype)
    leaf = values.clone().requires_grad_(True)
    hidden = leaf+0 if reuse else leaf
    head = torch.randn(259, 32, device="cuda", dtype=torch.float32, requires_grad=True)
    weights = torch.rand(17)
    if layout == "interval":
        weights[:3] = weights[-2:] = 0
    elif layout == "gaps":
        weights[::3] = 0
    elif layout == "empty":
        weights.zero_()
    selection = active_rows(weights, "cuda")
    assert isinstance(selection, slice) == (layout != "gaps")
    targets = torch.randint(259, (17,), device="cuda")
    weights = weights.cuda()
    oracle_input = values.clone().requires_grad_(True)
    oracle_head = head.detach().clone().requires_grad_(True)
    logits = torch.nn.functional.linear(oracle_input, oracle_head.to(dtype)).float()
    expected = (torch.nn.functional.cross_entropy(logits, targets, reduction="none")*weights).sum()
    def forbidden(*a, **kw):
        raise AssertionError("Host selection and contiguous views must avoid CUDA selection/gather")
    monkeypatch.setattr(torch, "nonzero", forbidden)
    if layout != "gaps":
        monkeypatch.setattr(torch, "index_select", forbidden)
        monkeypatch.setattr(torch.Tensor, "index_copy_", forbidden)
    actual = projection_loss(hidden, head, targets, weights, token_budget=5, compute_dtype=dtype,
                             reuse_hidden=reuse, rows=selection)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    gradients = torch.autograd.grad(actual, (leaf, head), retain_graph=True)
    for result, oracle in zip(gradients, torch.autograd.grad(expected, (oracle_input, oracle_head))):
        torch.testing.assert_close(result, oracle, rtol=.03 if dtype == torch.bfloat16 else 3e-5,
                                   atol=.03 if dtype == torch.bfloat16 else 3e-5)
    for result, repeat in zip(gradients, torch.autograd.grad(actual, (leaf, head))):
        torch.testing.assert_close(result, repeat, rtol=0, atol=0)
    torch.testing.assert_close(leaf, values, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_nf4_odd_tail_groups(dtype):
    bnb = pytest.importorskip("bitsandbytes")
    from causal_tuner.kernels.linear import materialize
    source = torch.randn(17, 19, device="cuda", dtype=dtype)
    packed, state = bnb.functional.quantize_4bit(source, blocksize=64, compress_statistics=True, quant_type="nf4")
    native = bnb.functional.dequantize_4bit(packed, state)
    for compute in (torch.bfloat16, torch.float32):
        torch.testing.assert_close(materialize(packed, state, compute), native.to(compute), rtol=0, atol=0)
