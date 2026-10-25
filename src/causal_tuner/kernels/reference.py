"""Small-shape attention oracles and a portable CPU projection loss.

Production CUDA paths do not fall back to these operators.
"""

import torch
from torch.nn import functional as F
from .loss import row_count


def _masked_attention(q, k, v, mask):
    score = torch.einsum("bqhd,bkd->bhqk", q.float(), k[:, :, 0].float()) / q.shape[-1]**0.5
    score = score.masked_fill(~mask, -torch.inf)
    score = torch.where(mask.any(-1, keepdim=True), score, 0)
    probs = torch.softmax(score, -1) * mask
    return torch.einsum("bhqk,bkd->bqhd", probs, v[:, :, 0].float()).to(q.dtype)


def dense_attention(q, k, v, segments, attention_end=None):
    b, s, h, _ = q.shape
    if b*h*s*s > 8_388_608:
        raise ValueError("vanilla attention exceeds the small-shape reference limit")
    index = torch.arange(s, device=q.device)
    end = index.expand(b, s) if attention_end is None else attention_end
    mask = ((segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] > 0)
            & (index[None, None, :] <= end[:, :, None]))
    return _masked_attention(q, k, v, mask[:, None])


class ProjectionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head, targets, weights, tile, compute_dtype, active):
        ctx.tile, ctx.compute_dtype = tile, compute_dtype
        ctx.interval = active if isinstance(active, slice) else None
        ctx.save_for_backward(hidden, head, targets, weights, None if ctx.interval is not None else active)
        h, t, w = hidden.reshape(-1, hidden.shape[-1]), targets.flatten(), weights.flatten()
        matrix = head.to(compute_dtype)
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        count = row_count(active)
        for start in range(0, count, tile):
            rows = (slice(active.start+start, active.start+min(start+tile, count))
                    if ctx.interval is not None else active[start:start+tile])
            logits = F.linear(h[rows].to(compute_dtype), matrix).float()
            total.add_((F.cross_entropy(logits, t[rows], reduction="none") * w[rows]).sum())
        return total

    @staticmethod
    def backward(ctx, upstream):
        hidden, head, targets, weights, active = ctx.saved_tensors
        active = ctx.interval if ctx.interval is not None else active
        h, t, w = hidden.reshape(-1, hidden.shape[-1]), targets.flatten(), weights.flatten()
        matrix = head.to(ctx.compute_dtype)
        dh = torch.zeros_like(h) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(head) if ctx.needs_input_grad[1] else None
        count = row_count(active)
        for start in range(0, count, ctx.tile):
            size = min(ctx.tile, count-start)
            rows = slice(active.start+start, active.start+start+size) if ctx.interval is not None else active[start:start+size]
            x = h[rows].to(ctx.compute_dtype)
            logits = F.linear(x, matrix).float()
            derivative = logits.softmax(-1)
            derivative[torch.arange(size, device=h.device), t[rows]] -= 1
            derivative.mul_((w[rows] * upstream)[:, None])
            derivative = derivative.to(ctx.compute_dtype)
            if dh is not None:
                dh[rows] = (derivative @ matrix).to(dh.dtype)
            if dw is not None:
                if head.dtype == torch.float32:
                    # Both the GEMM output and every cross-tile sum stay FP32.
                    dw.addmm_(derivative.float().T, x.float())
                else:
                    dw.add_((derivative.T @ x).to(dw.dtype))
        return None if dh is None else dh.reshape_as(hidden), dw, None, None, None, None, None
