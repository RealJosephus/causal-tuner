"""Reuse compiled tensor-only kernels through Triton's standard launch interface."""

import torch
import triton
import triton.language as tl


_CACHE = {}


@triton.jit
def element_indices(N: tl.constexpr, BLOCK: tl.constexpr):
    # Include the masked tail in the bound; widen before multiplying the program ID.
    kind: tl.constexpr = tl.int64 if N+BLOCK > 2**31 else tl.int32
    return tl.program_id(0).to(kind)*BLOCK+tl.arange(0, BLOCK)


def launch(kernel, grid, *tensors, **meta):
    """All positional arguments are tensors; keyword arguments specialize the kernel."""
    device = tensors[0].device.index
    if device != torch.cuda.current_device():
        with torch.cuda.device(device):
            return launch(kernel, grid, *tensors, **meta)
    knobs = triton.knobs
    instrumented = (knobs.runtime.debug or getattr(knobs.runtime, "add_stages_inspection_hook", None) is not None
                    or getattr(knobs.compilation, "instrumentation_mode", "") != "")
    if torch.compiler.is_compiling() or getattr(kernel, "pre_run_hooks", None) or instrumented:
        return kernel[grid](*tensors, **meta)
    # Dtype and 16-byte alignment determine tensor specialization. Keep no tensor
    # references or stream handles: each invocation must use its current stream.
    key = (kernel, device, tuple((tensor.dtype, tensor.data_ptr() % 16 == 0) for tensor in tensors),
           tuple(meta.items()))
    entry = _CACHE.get(key)
    if entry is None:
        compiled = kernel[grid](*tensors, **meta)
        if compiled is not None:
            arguments = tuple(meta[name] for name in kernel.arg_names[len(tensors):])
            if len(_CACHE) >= 256:
                _CACHE.clear()
            _CACHE[key] = compiled, arguments
    else:
        compiled, arguments = entry
        compiled[tuple(grid)+(1,)*(3-len(grid))](*tensors, *arguments)
