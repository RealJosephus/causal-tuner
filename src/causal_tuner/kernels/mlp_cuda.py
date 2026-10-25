"""Tiled LoRA MLP with fused SiLU products and a graph-free analytic backward."""

import torch
import triton
import triton.language as tl
from .linear import materialize, forward_linear, input_gradient, parameter_gradients, projection_parameters
from .launch import launch, element_indices


@triton.jit
def _activation(G, U, H, DG, DU, N: tl.constexpr, BACKWARD: tl.constexpr, RECONSTRUCT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = element_indices(N, BLOCK)
    raw = tl.load(G+offsets, offsets < N, 0)
    gate = raw.to(tl.float32)
    up = tl.load(U+offsets, offsets < N, 0).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    activated = (gate*sigmoid).to(raw.dtype).to(tl.float32)
    if BACKWARD:
        grad = tl.load(H+offsets, offsets < N, 0).to(tl.float32)
        product_grad = (grad*up).to(raw.dtype).to(tl.float32)
        dgate = product_grad*(sigmoid*(1+gate*(1-sigmoid)))
        tl.store(DG+offsets, dgate, offsets < N)
        tl.store(DU+offsets, grad*activated, offsets < N)
    if RECONSTRUCT:
        tl.store(H+offsets, activated*up, offsets < N)


def activation(gate, up, gradient=None, *, out=None):
    if gradient is not None:
        out = gradient
    elif out is None:
        out = torch.empty_like(gate)
    launch(_activation, (triton.cdiv(gate.numel(), 1024),),
        gate, up, out, gate, up, N=gate.numel(), BACKWARD=gradient is not None, RECONSTRUCT=True, BLOCK=1024)
    return out


class _SiLUMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate, up = gate.contiguous(), up.contiguous()
        ctx.save_for_backward(gate, up)
        return activation(gate, up)

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        launch(_activation, (triton.cdiv(gate.numel(), 1024),), gate, up, grad.contiguous(), dg, du,
               N=gate.numel(), BACKWARD=True, RECONSTRUCT=False, BLOCK=1024)
        return dg, du


def silu_multiply(gate, up):
    return _SiLUMultiply.apply(gate, up)


@torch.no_grad()
def _forward_into(output, x, dtype, tile, states, parameters, residual, *, save=False):
    flat = x.reshape(-1, x.shape[-1]).to(dtype)
    matrices = [materialize(parameters[4*i], state, dtype) for i, state in enumerate(states)]
    adapters = [(parameters[4*i+1].to(dtype), parameters[4*i+2].to(dtype)) for i in range(3)]
    workspace = torch.empty((1 if save else 2, min(tile, len(flat)), len(matrices[0])), device=x.device, dtype=dtype)
    low = torch.empty((min(tile, len(flat)), max(len(a) for a, _ in adapters)), device=x.device, dtype=dtype)
    saved = (() if not save else (
        *(torch.empty((len(flat), len(matrices[0])), device=x.device, dtype=dtype) for _ in range(2)),
        *(torch.empty((len(flat), len(a)), device=x.device, dtype=dtype) for a, _ in adapters[:2])))
    destination = output.view_as(flat)
    for start in range(0, len(flat), tile):
        part = flat[start:start+tile]
        gate, up = (buffer[start:start+len(part)] for buffer in saved[:2]) if save else (buffer[:len(part)] for buffer in workspace)
        xa = [low[:len(part), :len(a)] for a, _ in adapters]
        if save:
            xa[:2] = [buffer[start:start+len(part)] for buffer in saved[2:]]
        forward_linear(part, matrices[0], *adapters[0], out=gate, xa=xa[0])
        forward_linear(part, matrices[1], *adapters[1], out=up, xa=xa[1])
        hidden = activation(gate, up, out=workspace[0, :len(part)] if save else gate)
        forward_linear(hidden, matrices[2], *adapters[2], out=destination[start:start+tile], xa=xa[2])
    if residual is not None:
        output.add_(residual)
    return saved


class _TiledLoRAMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype, tile, states, residual, cache, *parameters):
        ctx.dtype, ctx.tile, ctx.states = dtype, tile, states
        ctx.cache = cache
        if cache:
            output = torch.empty(x.shape, device=x.device, dtype=dtype)
            saved = _forward_into(output, x, dtype, tile, states, parameters, residual, save=True)
            ctx.save_for_backward(x, *saved, *parameters)
            return output
        # Register saved inputs before the wrapper fills this private output.
        # Checkpoint replay can then stop before redundant GEMMs.
        ctx.save_for_backward(x, *parameters)
        return torch.empty(x.shape, device=x.device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        x, *parameters = ctx.saved_tensors
        saved = parameters[:4] if ctx.cache else ()
        if ctx.cache:
            parameters = parameters[4:]
        residual_grad = grad if ctx.needs_input_grad[4] else None
        flat = x.reshape(-1, x.shape[-1]).to(ctx.dtype)
        grad = grad.reshape_as(flat).to(ctx.dtype)
        dx = torch.empty_like(flat) if ctx.needs_input_grad[0] else None
        matrices = [materialize(parameters[4*i], state, ctx.dtype) for i, state in enumerate(ctx.states)]
        adapters = [(parameters[4*i+1].to(ctx.dtype), parameters[4*i+2].to(ctx.dtype)) for i in range(3)]
        allocate = torch.empty_like if len(flat) else torch.zeros_like
        gradients = [(allocate(parameters[4*i+1]), allocate(parameters[4*i+2])) for i in range(3)]
        rows = min(ctx.tile, len(flat))
        workspace = torch.empty((3, rows, len(matrices[0])), device=x.device, dtype=ctx.dtype)
        low_inputs = [torch.empty((rows, len(a)), device=x.device, dtype=ctx.dtype) for a, _ in adapters]
        low_grad = torch.empty((rows, max(len(a) for a, _ in adapters)), device=x.device, dtype=ctx.dtype)
        for start in range(0, len(flat), ctx.tile):
            part, dy = flat[start:start+ctx.tile], grad[start:start+ctx.tile]
            gate, up, dh = (buffer[:len(part)] for buffer in workspace)
            xa = [buffer[:len(part)] for buffer in low_inputs]
            low = [low_grad[:len(part), :len(a)] for a, _ in adapters]
            if ctx.cache:
                # Activation backward overwrites its tile buffers. Copy only
                # the current tile so saved values survive repeated backward.
                gate.copy_(saved[0][start:start+len(part)])
                up.copy_(saved[1][start:start+len(part)])
                xa[:2] = [value[start:start+len(part)] for value in saved[2:]]
            else:
                forward_linear(part, matrices[0], *adapters[0], out=gate, xa=xa[0])
                forward_linear(part, matrices[1], *adapters[1], out=up, xa=xa[1])
            input_gradient(dy, matrices[2], *adapters[2], dh, low[2])
            # Reuse all three wide tile buffers: (gate, up, dh) -> (dgate, dup, hidden).
            hidden = activation(gate, up, dh)
            torch.mm(hidden, adapters[2][0].T, out=xa[2])
            parameter_gradients(hidden, dy, *gradients[2], low[2], xa[2], add=start > 0)
            for index, signal in enumerate((gate, up)):
                if dx is not None:
                    input_gradient(signal, matrices[index], *adapters[index], dx[start:start+ctx.tile], low[index], add=index > 0)
                else:
                    torch.mm(signal, adapters[index][1], out=low[index])
                parameter_gradients(part, signal, *gradients[index], low[index], xa[index], add=start > 0)
        result = [value for da, db in gradients for value in (None, da, db, None)]
        return (None if dx is None else dx.reshape_as(x).to(x.dtype), None, None, None, residual_grad, None, *result)


def tiled_lora_mlp(module, x, residual=None, *, tile=None, cache=None):
    modules = (module.gate_proj, module.up_proj, module.down_proj)
    dtype = modules[0].compute_dtype
    if residual is not None and (residual.shape != x.shape or residual.dtype != dtype):
        return tiled_lora_mlp(module, x, tile=tile, cache=cache)+residual
    parameters, states = [], []
    for projection in modules:
        values, state = projection_parameters(projection)
        parameters.extend(values)
        states.append(state)
    tile = (module.tile if tile is None else tile) or max(1, x.numel()//x.shape[-1])
    cache = (module.cache_intermediates if cache is None else cache) and torch.is_grad_enabled()
    output = _TiledLoRAMLP.apply(x, dtype, tile, tuple(states), residual, cache, *parameters)
    if not cache:
        _forward_into(output, x, dtype, tile, states, parameters, residual)
    return output
