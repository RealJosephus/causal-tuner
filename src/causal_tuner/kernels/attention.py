"""Packed Dense attention without long-sequence square masks."""

import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class DenseLayout:
    spans: tuple
    mode: str
    offsets: torch.Tensor | None = None
    maximum: int = 0
    groups: tuple = ()


def prepare_dense_layout(shape, spans, device, attention_end=None):
    """Build immutable batch metadata once, including checkpoint replay consumers."""
    spans = tuple(spans)
    if not spans:
        return DenseLayout(spans, "empty")
    if attention_end is not None:
        if tuple(attention_end.shape) != tuple(shape):
            raise ValueError("attention_end must match the batch and sequence dimensions")
        ends = attention_end.cpu().tolist()
        groups = []
        for row, left, right in spans:
            start = left
            while start < right:
                stop = ends[row][start]+1
                if stop < start + 1 or stop > right:
                    raise ValueError("attention_end must stay inside its contiguous segment")
                if stop == start + 1:
                    while stop < right and ends[row][stop] == stop:
                        stop += 1
                    causal = True
                else:
                    if any(end != stop-1 for end in ends[row][start:stop]):
                        raise ValueError("Bidirectional groups must be complete contiguous intervals")
                    causal = False
                groups.append((row, left, start, stop, causal))
                start = stop
        return DenseLayout(spans, "groups", groups=tuple(groups))
    if spans == tuple((row, 0, shape[1]) for row in range(shape[0])):
        return DenseLayout(spans, "full")
    sizes = [end-start for _, start, end in spans]
    boundaries = [0]
    for size in sizes:
        boundaries.append(boundaries[-1]+size)
    offsets = torch.tensor(boundaries, dtype=torch.int32, device=device)
    cursor, contiguous = 0, True
    for row, start, end in spans:
        contiguous = contiguous and row*shape[1]+start == cursor
        cursor = row*shape[1]+end
    mode = "packed" if contiguous and cursor == shape[0]*shape[1] else "gaps"
    return DenseLayout(spans, mode, offsets, max(sizes))


def flash_dense(q, k, v, spans, attention_end=None, *, layout=None):
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("flash_attention_2 requires CUDA and BF16/FP16 inputs")
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_func
    except ImportError as exc:
        raise RuntimeError("flash_attention_2 is unavailable; install flash-attn (no automatic fallback)") from exc
    layout = layout or prepare_dense_layout(q.shape[:2], spans, q.device, attention_end)
    spans, offsets = layout.spans, layout.offsets
    if layout.mode == "empty":
        return q * 0 + (k.sum() + v.sum()) * 0
    if layout.mode == "groups":
        output = torch.zeros_like(q)
        for row, left, start, stop, causal in layout.groups:
            output[row:row+1, start:stop] = flash_attn_func(
                q[row:row+1, start:stop], k[row:row+1, left:stop],
                v[row:row+1, left:stop], dropout_p=0.0, causal=causal)
        return output
    if layout.mode == "full":
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    if layout.mode == "packed":
        # Packed segments already cover the flat storage in order. Preserve the
        # segment offsets without copying inputs or scattering the output.
        return flash_attn_varlen_func(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), offsets, offsets,
                                      layout.maximum, layout.maximum, dropout_p=0.0, causal=True).reshape_as(q)
    def compact(x):
        return torch.cat([x[row, start:end] for row, start, end in spans], 0)
    packed = flash_attn_varlen_func(compact(q), compact(k), compact(v), offsets, offsets,
                                    layout.maximum, layout.maximum, dropout_p=0.0, causal=True)
    output = torch.zeros_like(q)
    offset = 0
    for row, start, end in spans:
        length = end-start
        output[row, start:end] = packed[offset:offset+length]
        offset += length
    return output
