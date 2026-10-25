"""Projection and CE are jointly bounded; the full sequence logits never exist."""

import torch


def active_rows(weights, device=None):
    """Select supervision on the host before transfer; keep contiguous rows as a view."""
    indices = torch.nonzero(weights.flatten() > 0, as_tuple=False).flatten()
    count = len(indices)
    if not count or count == weights.numel():
        return slice(0, count)
    if weights.device.type == "cpu":
        first, last = int(indices[0]), int(indices[-1])
        if last-first+1 == count:
            return slice(first, last+1)
    return indices.to(device) if device is not None else indices


def row_count(rows):
    return rows.stop-rows.start if isinstance(rows, slice) else len(rows)


def projection_loss(hidden, head, targets, weights, *, token_budget=128,
                    compute_dtype=torch.bfloat16, implementation="fused", cut_chunk_size=0, reuse_hidden=False, rows=None):
    """reuse_hidden consumes an exclusive output whose producer does not save it."""
    if token_budget <= 0:
        raise ValueError("loss_token_budget must be positive")
    rows = active_rows(weights, hidden.device) if rows is None else rows
    if hidden.is_cuda and head.requires_grad and head.dtype == torch.float32:
        # FP32 output alone does not forbid reduced-precision GEMM reductions.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if implementation == "cut":
        if head.requires_grad:
            raise ValueError("cut cannot guarantee FP32 head-gradient accumulation; use fused")
        try:
            from cut_cross_entropy import linear_cross_entropy
        except ImportError as exc:
            raise RuntimeError("The cut backend requires cut-cross-entropy") from exc
        h, t, w = hidden.reshape(-1, hidden.shape[-1]), targets.flatten(), weights.flatten()
        count = row_count(rows)
        if not count:
            return h[:0].sum(dtype=torch.float32)
        matrix = head.to(compute_dtype)
        total = None
        size = cut_chunk_size or count
        for start in range(0, count, size):
            selected = (slice(rows.start+start, rows.start+min(start+size, count))
                        if isinstance(rows, slice) else rows[start:start+size])
            per_token = linear_cross_entropy(h[selected].to(compute_dtype), matrix,
                                              t[selected], reduction="none", filter_eps="auto", impl="cce")
            loss = (per_token * w[selected]).sum()
            total = loss if total is None else total+loss
        return total
    if implementation != "fused":
        raise ValueError("Unsupported projection-loss backend")
    if cut_chunk_size:
        raise ValueError("cut_chunk_size applies only to the cut backend")
    if hidden.is_cuda:
        from .loss_cuda import FusedProjectionLoss
        if not torch.is_grad_enabled():
            hidden, head = hidden.detach(), head.detach()
        if reuse_hidden and hidden.requires_grad and (hidden.is_leaf or hidden._base is not None or not hidden.is_contiguous()):
            raise ValueError("Hidden reuse requires an exclusive contiguous non-leaf tensor")
        return FusedProjectionLoss.apply(hidden, head, targets, weights, token_budget, compute_dtype, reuse_hidden, rows)[0]
    from .reference import ProjectionLoss
    return ProjectionLoss.apply(hidden, head, targets, weights, token_budget, compute_dtype, rows)
