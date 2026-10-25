"""Decode packed NF4 weights and nested scales in one CUDA launch."""

import torch
import triton
import triton.language as tl
from .launch import launch, element_indices


@triton.jit
def _decode(PACKED, SCALES, CODE, SCALE_CODE, SCALE_MAX, OFFSET, OUT,
            N: tl.constexpr, GROUP: tl.constexpr, SCALE_GROUP: tl.constexpr,
            NESTED: tl.constexpr, STORAGE_DTYPE: tl.constexpr, ROWS: tl.constexpr):
    group_count: tl.constexpr = (N+GROUP-1)//GROUP
    groups = element_indices(group_count, ROWS)
    valid = groups < group_count
    scale = tl.load(SCALES+groups, valid, 0)
    if NESTED:
        decoded_scale = tl.load(SCALE_CODE+scale.to(tl.int32), eviction_policy="evict_last").to(tl.float32)
        maximum = tl.load(SCALE_MAX+groups//SCALE_GROUP, valid, 0).to(tl.float32)
        scale = decoded_scale*maximum+tl.load(OFFSET).to(tl.float32)
    addresses = groups.to(tl.int64) if N+ROWS*GROUP > 2**31 else groups
    packed_offsets = addresses[:, None]*(GROUP//2)+tl.arange(0, GROUP//2)[None, :]
    packed = tl.load(PACKED+packed_offsets, packed_offsets < (N+1)//2, 0).to(tl.int32)
    high = tl.load(CODE+(packed >> 4), eviction_policy="evict_last").to(tl.float32)
    low = tl.load(CODE+(packed & 15), eviction_policy="evict_last").to(tl.float32)
    # Each packed byte is read once; the decoded scale is shared by its entire group.
    values = tl.interleave(high, low)*scale[:, None].to(tl.float32)
    indices = addresses[:, None]*GROUP+tl.arange(0, GROUP)[None, :]
    tl.store(OUT+indices, values.to(STORAGE_DTYPE), indices < N)


def dequantize_nf4(weight, state, dtype):
    output = torch.empty(state.shape, device=weight.device, dtype=dtype)
    nested = bool(state.nested)
    auxiliary = state.state2 if nested else state
    storage_dtype = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}[state.dtype]
    rows = max(1, 4096//state.blocksize)
    launch(_decode, (triton.cdiv(output.numel(), rows*state.blocksize),),
        weight, state.absmax, state.code, auxiliary.code, auxiliary.absmax,
        state.offset if nested else state.absmax, output, N=output.numel(), GROUP=state.blocksize,
        SCALE_GROUP=auxiliary.blocksize, NESTED=nested, STORAGE_DTYPE=storage_dtype,
        ROWS=rows, enable_fp_fusion=False)
    return output
