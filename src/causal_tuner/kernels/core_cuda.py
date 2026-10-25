"""Row-wise normalization and adjacent-pair RoPE without full FP32 activations."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from .launch import launch, element_indices


@triton.jit
def _position_values(POSITIONS, INVERSE, COSINE, SINE, N: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    indices = element_indices(N, BLOCK)
    position = tl.load(POSITIONS+indices//HALF, indices < N, 0).to(tl.float32)
    inverse = tl.load(INVERSE+indices % HALF).to(tl.float32)
    angle = position*inverse
    # Full-range argument reduction is required for long logical positions.
    tl.store(COSINE+indices, libdevice.cos(angle), indices < N)
    tl.store(SINE+indices, libdevice.sin(angle), indices < N)


def position_values(positions, inverse, dtype):
    cosine = torch.empty((*positions.shape, len(inverse)), device=positions.device, dtype=dtype)
    sine = torch.empty_like(cosine)
    launch(_position_values, (triton.cdiv(cosine.numel(), 256),), positions.contiguous(), inverse, cosine, sine,
           N=cosine.numel(), HALF=len(inverse), BLOCK=256)
    return cosine, sine


@triton.jit
def _rms_forward(X, W, Y, RSTD, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    channel = tl.arange(0, BLOCK)
    raw = tl.load(X+row*WIDTH+channel, channel < WIDTH, 0)
    x = raw.to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x*x, 0)/WIDTH+EPS)
    normalized = (x*inverse).to(raw.dtype).to(tl.float32)
    weight = tl.load(W+channel, channel < WIDTH, 0).to(raw.dtype).to(tl.float32)
    tl.store(Y+row*WIDTH+channel, normalized*weight, channel < WIDTH)
    tl.store(RSTD+row, inverse)


@triton.jit
def _rms_gradient_row(X, W, RSTD, GO, GX, row, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                      FROM_OUTPUT: tl.constexpr, INPUT_GRAD: tl.constexpr, WEIGHT_GRAD: tl.constexpr):
    channel = tl.arange(0, BLOCK)
    raw = tl.load(X+row*WIDTH+channel, channel < WIDTH, 0)
    x = raw.to(tl.float32)
    weight = tl.load(W+channel, channel < WIDTH, 0).to(raw.dtype).to(tl.float32)
    go = tl.load(GO+row*WIDTH+channel, channel < WIDTH, 0).to(tl.float32)
    inverse = tl.load(RSTD+row)
    if INPUT_GRAD:
        gradient = (go*weight).to(raw.dtype).to(tl.float32)
        if FROM_OUTPUT:
            normalized = tl.where(channel < WIDTH, x/weight, 0.)
            projection = tl.sum(gradient*normalized, 0)/WIDTH
            gx = inverse*(gradient-normalized*projection)
        else:
            projection = tl.sum(gradient*x, 0)/WIDTH
            gx = inverse*(gradient-x*(inverse*inverse*projection))
        tl.store(GX+row*WIDTH+channel, gx, channel < WIDTH)
    contribution = tl.full((BLOCK,), 0., tl.float32)
    if WEIGHT_GRAD:
        contribution = go*(x*inverse).to(raw.dtype).to(tl.float32)
    return contribution


@triton.jit
def _rms_backward(X, W, RSTD, GO, GX, DW, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                  FROM_OUTPUT: tl.constexpr, INPUT_GRAD: tl.constexpr, WEIGHT_GRAD: tl.constexpr,
                  ROWS_PER_GROUP: tl.constexpr):
    index_type: tl.constexpr = tl.int64 if ROWS*WIDTH+BLOCK > 2**31 else tl.int32
    group = tl.program_id(0).to(index_type)
    if WEIGHT_GRAD:
        partial = tl.full((BLOCK,), 0., tl.float32)
        first = group*ROWS_PER_GROUP
        for row in range(first, tl.minimum(first+ROWS_PER_GROUP, ROWS)):
            partial += _rms_gradient_row(X, W, RSTD, GO, GX, row, WIDTH, BLOCK, False, INPUT_GRAD, True)
        channel = tl.arange(0, BLOCK)
        tl.store(DW+group*WIDTH+channel, partial, channel < WIDTH)
    else:
        _rms_gradient_row(X, W, RSTD, GO, GX, group, WIDTH, BLOCK, FROM_OUTPUT, True, False)


@triton.jit
def _rms_weight_reduce(PARTIALS, DW, WIDTH: tl.constexpr, GROUPS: tl.constexpr, REDUCE: tl.constexpr,
                       CHANNELS: tl.constexpr):
    group = tl.arange(0, REDUCE)
    channel = tl.program_id(0)*CHANNELS+tl.arange(0, CHANNELS)
    values = tl.load(PARTIALS+group[:, None]*WIDTH+channel[None, :],
                     (group[:, None] < GROUPS) & (channel[None, :] < WIDTH), 0.)
    tl.store(DW+channel, tl.sum(values, 0), channel < WIDTH)


class RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, reuse_grad=False, reconstruct=False):
        # Reconstruction and private-gradient reuse belong to frozen pre-norms.
        ctx.reuse_grad = reuse_grad and not weight.requires_grad
        ctx.reconstruct = reconstruct and not weight.requires_grad
        x = x.contiguous()
        output = torch.empty_like(x)
        width = x.shape[-1]
        inverse = torch.empty(x.numel()//width, device=x.device, dtype=torch.float32)
        launch(_rms_forward, (len(inverse),), x, weight, output, inverse, WIDTH=width,
                                     BLOCK=triton.next_power_of_2(width), EPS=eps)
        ctx.save_for_backward(output if ctx.reconstruct else x, weight, inverse)
        return output

    @staticmethod
    def backward(ctx, grad):
        x, weight, inverse = ctx.saved_tensors
        grad = grad.contiguous()
        # Only decoder pre-norms followed by our analytic projections opt in:
        # those projections return a private, freshly allocated input gradient.
        result = (grad if ctx.reuse_grad else torch.empty_like(x)) if ctx.needs_input_grad[0] else None
        dw = torch.empty_like(weight) if ctx.needs_input_grad[1] else None
        width, rows = x.shape[-1], len(inverse)
        groups = min(rows, 4*torch.cuda.get_device_properties(x.device).multi_processor_count) if dw is not None else rows
        rows_per_group = triton.cdiv(rows, groups) if groups else 1
        groups = triton.cdiv(rows, rows_per_group)
        partials = torch.empty((groups, width), device=x.device, dtype=torch.float32) if dw is not None else None
        if rows:
            launch(_rms_backward, (groups,), x, weight, inverse, grad,
                   result if result is not None else x, partials if partials is not None else weight,
                   ROWS=rows, WIDTH=width, BLOCK=triton.next_power_of_2(width), FROM_OUTPUT=ctx.reconstruct,
                   INPUT_GRAD=result is not None, WEIGHT_GRAD=dw is not None, ROWS_PER_GROUP=rows_per_group)
        if dw is not None:
            launch(_rms_weight_reduce, (triton.cdiv(width, 32),), partials, dw, WIDTH=width, GROUPS=groups,
                   REDUCE=triton.next_power_of_2(max(1, groups)), CHANNELS=32)
        return (result, dw, None, None, None)[:len(ctx.needs_input_grad)]


@triton.jit
def _rotate_pair(even, odd, cosine, sine, INVERSE: tl.constexpr):
    # Launches disable contraction so each activation-dtype product rounds before addition.
    if INVERSE:
        sine = -sine
    ec = (even.to(tl.float32)*cosine).to(even.dtype).to(tl.float32)
    os = (odd.to(tl.float32)*sine).to(even.dtype).to(tl.float32)
    oc = (odd.to(tl.float32)*cosine).to(even.dtype).to(tl.float32)
    es = (even.to(tl.float32)*sine).to(even.dtype).to(tl.float32)
    return ec-os, oc+es


@triton.jit
def _rope(X, C, S, Y, TOTAL: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
          ROTARY: tl.constexpr, INVERSE: tl.constexpr, BLOCK: tl.constexpr):
    coordinate_type: tl.constexpr = tl.int64 if TOTAL//2+BLOCK > 2**31 else tl.int32
    pairs = tl.program_id(0).to(coordinate_type)*BLOCK+tl.arange(0, BLOCK)
    channel = pairs % (DIM//2)
    token = pairs//(HEADS*DIM//2)
    offsets = pairs
    if TOTAL+2*BLOCK > 2**31:
        offsets = offsets.to(tl.int64)
    offsets *= 2
    valid = pairs < TOTAL//2
    rotated = channel < ROTARY//2
    even, odd = tl.load(X+offsets, valid, 0), tl.load(X+offsets+1, valid, 0)
    cosine = tl.load(C+token*(ROTARY//2)+channel, valid & rotated, 0).to(tl.float32)
    sine = tl.load(S+token*(ROTARY//2)+channel, valid & rotated, 0).to(tl.float32)
    first, second = _rotate_pair(even, odd, cosine, sine, INVERSE)
    tl.store(Y+offsets, tl.where(rotated, first, even.to(tl.float32)), valid)
    tl.store(Y+offsets+1, tl.where(rotated, second, odd.to(tl.float32)), valid)


@triton.jit
def _owned_qk_rope(Q, K, C, S, ROWS: tl.constexpr, Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr,
                   DIM: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    coordinate_type: tl.constexpr = tl.int64 if ROWS*HALF+BLOCK > 2**31 else tl.int32
    heads: tl.constexpr = Q_HEADS if Q_HEADS >= K_HEADS else K_HEADS
    pairs = (tl.program_id(0)//heads).to(coordinate_type)*BLOCK+tl.arange(0, BLOCK)
    head = tl.program_id(0) % heads
    token, channel = pairs//HALF, pairs % HALF
    valid = pairs < ROWS*HALF
    cosine = tl.load(C+pairs, valid, 0).to(tl.float32)
    sine = tl.load(S+pairs, valid, 0).to(tl.float32)
    # Keep logical divisions narrow; widen only the final storage addresses.
    address_type: tl.constexpr = tl.int64 if (ROWS+BLOCK//HALF)*heads*DIM > 2**31 else coordinate_type
    token = token.to(address_type)
    if head < Q_HEADS:
        offset = (token*Q_HEADS+head)*DIM+2*channel
        first, second = _rotate_pair(tl.load(Q+offset, valid, 0), tl.load(Q+offset+1, valid, 0), cosine, sine, False)
        tl.store(Q+offset, first, valid)
        tl.store(Q+offset+1, second, valid)
    if head < K_HEADS:
        offset = (token*K_HEADS+head)*DIM+2*channel
        first, second = _rotate_pair(tl.load(K+offset, valid, 0), tl.load(K+offset+1, valid, 0), cosine, sine, False)
        tl.store(K+offset, first, valid)
        tl.store(K+offset+1, second, valid)


def rotate_owned_qk(query, key, cosine, sine, head_dim):
    """Rotate newly allocated projection outputs before exposing them to autograd."""
    rows, half = query.numel()//query.shape[-1], cosine.shape[-1]
    q_heads, k_heads = query.shape[-1]//head_dim, key.shape[-1]//head_dim
    # Flatten the grid to avoid CUDA's 65,535 limit on its second dimension.
    # Adjacent heads still share positions and the small trigonometric tile.
    launch(_owned_qk_rope, (max(q_heads, k_heads)*triton.cdiv(rows*half, 256),),
        query, key, cosine, sine, ROWS=rows, Q_HEADS=q_heads, K_HEADS=k_heads,
        DIM=head_dim, HALF=half, BLOCK=256, enable_fp_fusion=False)


def inverse_rope_tile(grad, cosine, sine, head_dim, output):
    """Write an inverse rotation into a private tile, preserving caller gradients."""
    launch(_rope, (triton.cdiv(grad.numel()//2, 256),),
        grad.contiguous(), cosine, sine, output, TOTAL=grad.numel(), HEADS=grad.shape[-1]//head_dim,
        DIM=head_dim, ROTARY=cosine.shape[-1]*2, INVERSE=True, BLOCK=256, enable_fp_fusion=False)
    return output


class PartialRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cosine, sine):
        x = x.contiguous()
        output = torch.empty_like(x)
        ctx.shape = x.shape
        ctx.save_for_backward(cosine, sine)
        launch(_rope, (triton.cdiv(x.numel()//2, 256),), x, cosine, sine, output, TOTAL=x.numel(), HEADS=x.shape[-2],
                                            DIM=x.shape[-1], ROTARY=cosine.shape[-1]*2, INVERSE=False, BLOCK=256, enable_fp_fusion=False)
        return output

    @staticmethod
    def backward(ctx, grad):
        cosine, sine = ctx.saved_tensors
        grad = grad.contiguous()
        output = torch.empty_like(grad)
        launch(_rope, (triton.cdiv(grad.numel()//2, 256),), grad, cosine, sine, output, TOTAL=grad.numel(), HEADS=ctx.shape[-2],
                                               DIM=ctx.shape[-1], ROTARY=cosine.shape[-1]*2, INVERSE=True, BLOCK=256, enable_fp_fusion=False)
        return output, None, None
