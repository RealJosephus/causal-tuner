"""Shared-input LoRA backward with bounded workspaces and FP32 master gradients."""

import torch


def materialize(weight, state, dtype):
    if state is not None:
        if weight.is_cuda and weight.dtype == torch.uint8 and weight.is_contiguous() and state.quant_type == "nf4":
            from .nf4_cuda import dequantize_nf4
            return dequantize_nf4(weight, state, dtype)
        from bitsandbytes.functional import dequantize_4bit
        weight = dequantize_4bit(weight, state)
    return weight.to(dtype)


def accumulate(output, left, right, *, add=True):
    beta = 1 if add else 0
    if output.dtype == torch.float32 and left.dtype == torch.bfloat16 and left.is_cuda:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.addmm(output, left, right, beta=beta, out_dtype=torch.float32, out=output)
    elif output.dtype != left.dtype:
        output.addmm_(left.to(output.dtype), right.to(output.dtype), beta=beta)
    else:
        output.addmm_(left, right, beta=beta)


def forward_linear(x, weight, a, b, *, out, xa, bias=None):
    if bias is None:
        torch.mm(x, weight.T, out=out)
    else:
        torch.addmm(bias, x, weight.T, out=out)
    torch.mm(x, a.T, out=xa)
    out.addmm_(xa, b.T)


def input_gradient(dy, weight, a, b, dx, low, *, add=False):
    torch.mm(dy, b, out=low)
    torch.addmm(dx, dy, weight, beta=1 if add else 0, out=dx)
    dx.addmm_(low, a)


def parameter_gradients(x, dy, da, db, low, xa, *, add):
    accumulate(da, low.T, x, add=add)
    accumulate(db, dy.T, xa, add=add)


def projection_parameters(module):
    weight = module.weight if module.quantized is None else module.quantized.weight
    state = None if module.quantized is None else weight.quant_state
    return (weight, module.lora_A, module.lora_B, module.bias), state


def can_fuse(module, x):
    return (x.is_cuda and module.rank and not (module.bias is not None and module.bias.requires_grad)
            and (module.weight is None or not module.weight.requires_grad)
            and module.lora_A.requires_grad and module.lora_B.requires_grad)


class _LoRAGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype, tile, states, cosine, sine, head_dim, residual, *parameters):
        ctx.save_for_backward(x, cosine, sine, *parameters)
        ctx.dtype, ctx.tile, ctx.states = dtype, tile or 4096, states
        ctx.head_dim = head_dim
        flat = x.reshape(-1, x.shape[-1]).to(dtype)
        outputs = []
        workspace = torch.empty((min(ctx.tile, len(flat)), max(len(a) for a in parameters[1::4])), device=x.device, dtype=dtype)
        for index, state in enumerate(states):
            weight, a, b, bias = parameters[index*4:index*4+4]
            matrix, ac, bc = materialize(weight, state, dtype), a.to(dtype), b.to(dtype)
            bias = None if bias is None else bias.to(dtype)
            low = workspace[:, :len(a)]
            if tile:
                out = torch.empty((len(flat), len(matrix)), device=x.device, dtype=dtype)
                for start in range(0, len(flat), tile):
                    part = flat[start:start+tile]
                    forward_linear(part, matrix, ac, bc, bias=bias, out=out[start:start+tile], xa=low[:len(part)])
            else:
                # An untiled base GEMM still uses bounded low-rank epilogues.
                out = torch.nn.functional.linear(flat, matrix, bias)
                for start in range(0, len(flat), ctx.tile):
                    part = flat[start:start+ctx.tile]
                    torch.mm(part, ac.T, out=low[:len(part)])
                    out[start:start+ctx.tile].addmm_(low[:len(part)], bc.T)
            outputs.append(out.reshape(*x.shape[:-1], len(matrix)))
            del matrix, ac, bc, low
        if head_dim:
            from .core_cuda import rotate_owned_qk
            rotate_owned_qk(outputs[0], outputs[1], cosine, sine, head_dim)
        if residual is not None:
            outputs[0].add_(residual)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, cosine, sine, *parameters = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1]).to(ctx.dtype)
        dx = torch.empty_like(flat) if ctx.needs_input_grad[0] else None
        if ctx.head_dim:
            cosine, sine = (value.reshape(-1, value.shape[-1]) for value in (cosine, sine))
        rows = min(ctx.tile, len(flat))
        workspace = torch.empty((2, rows, max(len(a) for a in parameters[1::4])), device=x.device, dtype=ctx.dtype)
        rotary = torch.empty(rows*max(len(b) for b in parameters[2:8:4]), device=x.device, dtype=ctx.dtype) if ctx.head_dim else None
        results = []
        for index, (state, grad) in enumerate(zip(ctx.states, grad_outputs)):
            weight, a, b, _ = parameters[index*4:index*4+4]
            matrix = materialize(weight, state, ctx.dtype) if dx is not None else None
            ac, bc = a.to(ctx.dtype), b.to(ctx.dtype)
            dy = grad.reshape(-1, grad.shape[-1]).to(ctx.dtype)
            allocate = torch.empty_like if len(flat) else torch.zeros_like
            da, db = allocate(a), allocate(b)
            rotate = ctx.head_dim and index < 2
            for start in range(0, len(flat), ctx.tile):
                stop = start+ctx.tile
                signal = dy[start:stop]
                if rotate:
                    from .core_cuda import inverse_rope_tile
                    target = rotary[:signal.numel()].view_as(signal)
                    signal = inverse_rope_tile(signal, cosine[start:stop], sine[start:stop], ctx.head_dim, target)
                low, xa = (buffer[:len(signal), :len(a)] for buffer in workspace)
                if dx is not None:
                    input_gradient(signal, matrix, ac, bc, dx[start:stop], low, add=index > 0)
                else:
                    torch.mm(signal, bc, out=low)
                torch.mm(flat[start:stop], ac.T, out=xa)
                parameter_gradients(flat[start:stop], signal, da, db, low, xa, add=start > 0)
            results.extend((None, da, db, None))
            del matrix, ac, bc, dy
        residual_grad = grad_outputs[0] if ctx.needs_input_grad[7] else None
        return (None if dx is None else dx.reshape_as(x).to(x.dtype), None, None, None, None, None, None, residual_grad, *results)


def project_group(modules, x, tile=0, *, rotary=None, residual=None):
    if residual is not None:
        if len(modules) != 1:
            raise ValueError("Residual fusion requires one output projection")
        shape = (*x.shape[:-1], len(modules[0].lora_B))
        if residual.shape != shape or residual.dtype != modules[0].compute_dtype:
            return (project_group(modules, x, tile, rotary=rotary)[0]+residual,)
    parameters, states = [], []
    for module in modules:
        values, state = projection_parameters(module)
        parameters.extend(values)
        states.append(state)
    cosine, sine, head_dim = (None, None, 0) if rotary is None else rotary
    if rotary is not None:
        cosine, sine = cosine.contiguous(), sine.contiguous()
    return _LoRAGroup.apply(x, modules[0].compute_dtype, tile, tuple(states), cosine, sine, head_dim, residual, *parameters)
