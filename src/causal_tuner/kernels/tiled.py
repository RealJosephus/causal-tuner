"""One tile graph at a time, with FP32 master-gradient accumulation."""

import torch


class _Tiled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, function, tile, *parameters):
        ctx.function, ctx.tile = function, tile
        ctx.save_for_backward(x, *parameters)
        flat = x.reshape(-1, x.shape[-1])
        output = None
        for start in range(0, len(flat), tile):
            value = function(flat[start:start+tile])
            if output is None:
                output = value.new_empty((len(flat), value.shape[-1]))
            output[start:start+tile].copy_(value)
        return output.reshape(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x, *parameters = ctx.saved_tensors
        flat, grad = x.reshape(-1, x.shape[-1]), grad_output.reshape(-1, grad_output.shape[-1])
        dx = torch.empty_like(flat) if ctx.needs_input_grad[0] else None
        buffers = [None] * len(parameters)
        active = [(i, p) for i, p in enumerate(parameters) if p.requires_grad]
        for start in range(0, len(flat), ctx.tile):
            with torch.enable_grad():
                part = flat[start:start+ctx.tile].detach().requires_grad_(dx is not None)
                value = ctx.function(part)
                derivatives = torch.autograd.grad(value, ([part] if dx is not None else []) + [p for _, p in active],
                                                  grad[start:start+ctx.tile], allow_unused=False)
            if dx is not None:
                dx[start:start+ctx.tile].copy_(derivatives[0])
            for (i, p), derivative in zip(active, derivatives[1:] if dx is not None else derivatives):
                if buffers[i] is None:
                    buffers[i] = derivative
                else:
                    buffers[i].add_(derivative)
        return (None if dx is None else dx.reshape_as(x), None, None, *buffers)


def tiled(function, x, size, parameters):
    if not size or x.numel() // x.shape[-1] <= size:
        return function(x)
    return _Tiled.apply(x, function, size, *tuple(parameters))
