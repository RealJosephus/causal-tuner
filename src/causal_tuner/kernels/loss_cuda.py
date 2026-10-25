"""Chunked projection and fused weighted CE with gradients computed once per tile."""

import torch
import triton
import triton.language as tl
from .linear import accumulate
from .loss import row_count
from .launch import launch, element_indices


@triton.jit
def _statistics(LOGITS, PARTS, TARGETS, WEIGHTS, LOSS,
                VOCAB: tl.constexpr, SPLITS: tl.constexpr, GRADIENT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    columns = split*BLOCK+tl.arange(0, BLOCK)
    values = tl.load(LOGITS+row*VOCAB+columns, columns < VOCAB, -float("inf")).to(tl.float32)
    maximum = tl.max(values, 0)
    normalizer = tl.sum(tl.exp(values-maximum), 0)
    lse = maximum+tl.log(normalizer)
    target = tl.load(TARGETS+row)
    if SPLITS == 1:
        selected = tl.sum(tl.where(columns == target, values, 0.), 0)
        weight = tl.load(WEIGHTS+row).to(tl.float32)
        tl.store(LOSS+row, weight*(lse-selected))
        if GRADIENT:
            derivative = (tl.exp(values-lse)-(columns == target).to(tl.float32))*weight
            tl.store(LOGITS+row*VOCAB+columns, derivative, columns < VOCAB)
    else:
        tl.store(PARTS+row*SPLITS+split, lse)
        if split == 0:
            # Capture the target before any program overwrites logits with gradients.
            tl.store(LOSS+row, tl.load(LOGITS+row*VOCAB+target).to(tl.float32))


@triton.jit
def _finish(LOGITS, PARTS, TARGETS, WEIGHTS, LOSS,
            VOCAB: tl.constexpr, SPLITS: tl.constexpr, PART_BLOCK: tl.constexpr,
            GRADIENT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    indices = tl.arange(0, PART_BLOCK)
    partial = tl.load(PARTS+row*SPLITS+indices, indices < SPLITS, -float("inf"))
    maximum = tl.max(partial, 0)
    lse = maximum+tl.log(tl.sum(tl.exp(partial-maximum), 0))
    weight = tl.load(WEIGHTS+row).to(tl.float32)
    if split == 0:
        selected = tl.load(LOSS+row)
        tl.store(LOSS+row, weight*(lse-selected))
    if GRADIENT:
        columns = split*BLOCK+tl.arange(0, BLOCK)
        raw = tl.load(LOGITS+row*VOCAB+columns, columns < VOCAB, -float("inf")).to(tl.float32)
        target = tl.load(TARGETS+row)
        derivative = (tl.exp(raw-lse)-(columns == target).to(tl.float32))*weight
        tl.store(LOGITS+row*VOCAB+columns, derivative, columns < VOCAB)


def weighted_ce(logits, targets, weights, *, gradient, loss=None, partials=None):
    """Write per-row weighted losses and optionally overwrite logits with derivatives."""
    rows, vocab = logits.shape
    splits = triton.cdiv(vocab, 4096)
    if loss is None:
        loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
    if partials is None:
        partials = torch.empty((rows, splits), device=logits.device, dtype=torch.float32) if splits > 1 else loss
    launch(_statistics, (rows, splits), logits, partials, targets, weights, loss, VOCAB=vocab, SPLITS=splits,
                                GRADIENT=gradient, BLOCK=min(4096, triton.next_power_of_2(vocab)))
    if splits > 1:
        launch(_finish, (rows, splits if gradient else 1), logits, partials, targets, weights, loss, VOCAB=vocab,
                     SPLITS=splits, PART_BLOCK=triton.next_power_of_2(splits), GRADIENT=gradient, BLOCK=4096)
    return loss


@triton.jit
def _clear_inactive(GRAD, WEIGHTS, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = element_indices(N, BLOCK)
    weight = tl.load(WEIGHTS+offsets//WIDTH, offsets < N, 0)
    tl.store(GRAD+offsets, 0., (offsets < N) & ~(weight > 0))


class FusedProjectionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head, targets, weights, tile, dtype, reuse_hidden, active):
        h = hidden.reshape(-1, hidden.shape[-1])
        t, w = targets.flatten(), weights.flatten()
        count = row_count(active)
        contiguous = isinstance(active, slice)
        reuse_hidden = reuse_hidden and ctx.needs_input_grad[0]
        allocate = torch.empty_like if contiguous else torch.zeros_like
        dh = (h if reuse_hidden else allocate(h)) if ctx.needs_input_grad[0] else None
        dw = (torch.empty_like(head) if count else torch.zeros_like(head)) if ctx.needs_input_grad[1] else None
        losses = torch.empty(count, device=hidden.device, dtype=torch.float32)
        rows = min(tile, count)
        direct = contiguous and dh is not None and dh.dtype == dtype
        if rows:
            matrix = head.to(dtype)
            workspace = torch.empty((rows, len(matrix)), device=hidden.device, dtype=dtype)
            if not contiguous:
                inputs = h.new_empty((rows, h.shape[-1]))
                labels, factors = t.new_empty(rows), w.new_empty(rows)
            signals = torch.empty((rows, h.shape[-1]), device=hidden.device, dtype=dtype) if dh is not None and not direct else None
            splits = triton.cdiv(len(matrix), 4096)
            partials = torch.empty((rows, splits), device=hidden.device, dtype=torch.float32) if splits > 1 else losses
        for start in range(0, count, tile):
            size = min(tile, count-start)
            if contiguous:
                indices = slice(active.start+start, active.start+start+size)
                x, target, weight = h[indices].to(dtype), t[indices], w[indices]
            else:
                indices = active[start:start+size]
                torch.index_select(h, 0, indices, out=inputs[:size])
                torch.index_select(t, 0, indices, out=labels[:size])
                torch.index_select(w, 0, indices, out=factors[:size])
                x, target, weight = inputs[:size].to(dtype), labels[:size], factors[:size]
            logits = workspace[:size]
            torch.mm(x, matrix.T, out=logits)
            weighted_ce(logits, target, weight, gradient=dh is not None or dw is not None,
                        loss=losses[start:start+size], partials=partials[:size])
            # A direct dHidden write can consume x; finish the head gradient first.
            if dw is not None:
                accumulate(dw, logits.T, x, add=start > 0)
            if dh is not None:
                if direct:
                    torch.mm(logits, matrix, out=dh[indices])
                else:
                    torch.mm(logits, matrix, out=signals[:size])
                    if contiguous:
                        dh[indices].copy_(signals[:size])
                    else:
                        dh.index_copy_(0, indices, signals[:size].to(dh.dtype))
        if dh is not None and contiguous:
            if active.start:
                dh[:active.start].zero_()
            if active.stop < len(h):
                dh[active.stop:].zero_()
        if reuse_hidden:
            # Active rows have already been gathered before their storage is reused.
            # Clear only inactive rows; their original hidden values are not gradients.
            if not contiguous and count != len(h):
                launch(_clear_inactive, (triton.cdiv(h.numel(), 1024),), dh, w,
                       N=h.numel(), WIDTH=h.shape[-1], BLOCK=1024)
            ctx.mark_dirty(hidden)
            dh = hidden
        elif dh is not None:
            dh = dh.reshape_as(hidden)
        if dh is not None:
            ctx.mark_non_differentiable(dh)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(dh, dw)
        # Returning the consumed input lets autograd track the in-place mutation.
        return losses.sum(), dh

    @staticmethod
    def backward(ctx, upstream, unused):
        dh, dw = ctx.saved_tensors
        # The normal training call has unit upstream. Preserve cached gradients for
        # repeated backward calls; scaling never modifies the saved tensors.
        scale = float(upstream)
        if scale != 1:
            dh = None if dh is None else dh*upstream
            dw = None if dw is None else dw*upstream
        return dh, dw, None, None, None, None, None, None
