"""Optional device limits and bounded pageable checkpoint storage."""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
import psutil
import torch

GIB = 1024**3


def host_peak_bytes():
    import sys
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss)


def check_host(limit):
    rss = psutil.Process().memory_info().rss
    if rss > limit:
        raise MemoryError(f"Host RSS {rss/GIB:.2f} GiB exceeds budget {limit/GIB:.2f} GiB")
    return rss


def check_gpu(budget):
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    if budget is not None and max(allocated, reserved) > budget:
        raise MemoryError("Measured GPU allocated/reserved memory exceeds the configured budget")
    return {"peak_allocated_gib": allocated/GIB, "peak_reserved_gib": reserved/GIB}


def guard_device(args, device):
    if torch.device(device).type != "cuda":
        raise RuntimeError("Training requires a CUDA device")
    return int(args.gpu_memory_budget_gib*GIB) if args.gpu_memory_budget_gib else None


class _Saved:
    def __init__(self, tensor, device, owner, future, index):
        self.tensor, self.device = tensor, device
        self.ready, self.index = future, index
        self.prefetched, self.unpacked = None, False
        self.ticket = {"future": future}
        self.finalizer = weakref.finalize(self, owner.retire, tensor.numel()*tensor.element_size(), self.ticket)


class ActivationOffload:
    """Bounded asynchronous pageable offload and one-boundary backward prefetch.

    One CPU worker pipelines two pinned staging slots. At most one forward
    boundary is in flight and one backward boundary is prefetched on the GPU.
    Host records stay pageable, and each slot is reused only after its event.
    """
    def __init__(self, host_budget, checkpoint_budget, pinned_budget, gpu_budget=None):
        self.host_budget, self.limit = int(host_budget), int(checkpoint_budget)
        self.staging_size = min(int(pinned_budget), 8*1024**2)
        self.gpu_budget = gpu_budget
        if self.staging_size < 4096:
            raise ValueError("CPU offload needs at least 4096 bytes of pinned staging")
        self.used = self.peak = self.pinned_peak = 0
        self.staging = None
        self.stream = None
        self.copy_events = None
        self.worker = None
        self.lock = threading.Lock()
        self.pending = None
        self.jobs, self.order = [], []
        self.prefetched = None
        self.stats = {"offloaded": 0, "prefetched": 0, "prefetch_hits": 0, "prefetch_skipped_budget": 0}

    def release(self, size):
        with self.lock:
            self.used -= size

    def retire(self, size, ticket):
        future = ticket["future"]
        if future.done():
            self.release(size)
        else:
            future.add_done_callback(lambda completed: self.release(size))

    def synchronize(self):
        if self.pending is not None:
            self.pending.result()
        for job in self.jobs:
            job.result()
        self.pending = None
        self.jobs.clear()

    def begin_forward(self):
        self.synchronize()
        self.order.clear()
        self.prefetched = None

    def close(self):
        try:
            self.synchronize()
        finally:
            if self.worker is not None:
                self.worker.shutdown(wait=True)
                self.worker = None

    def _submit(self, cpu, gpu, *, to_cpu):
        if self.worker is None:
            self.staging = torch.empty(self.staging_size, dtype=torch.uint8, pin_memory=True)
            self.pinned_peak = self.staging_size
            self.stream = torch.cuda.Stream(device=gpu.device)
            self.copy_events = (torch.cuda.Event(), torch.cuda.Event())
            self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="activation-copy")
        ready = torch.cuda.current_stream(gpu.device).record_event()
        return self.worker.submit(self._transfer, cpu, gpu, to_cpu=to_cpu, ready=ready)

    def pack(self, tensor):
        if not tensor.is_cuda or tensor.numel() == 0:
            return tensor
        # Backpressure bounds the extra live GPU source to one boundary.
        if self.pending is not None:
            self.pending.result()
        size = tensor.numel()*tensor.element_size()
        if self.used+size > self.limit:
            raise MemoryError("CPU activation checkpoint budget exceeded; increase remat_group_size or reduce context explicitly")
        if size+self.staging_size+512*1024**2 > psutil.virtual_memory().available:
            raise MemoryError("Insufficient available host RAM for the next checkpoint boundary")
        check_host(self.host_budget-size-self.staging_size)
        with self.lock:
            self.used += size
            self.peak = max(self.peak, self.used)
        try:
            cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            self.pending = self._submit(cpu, tensor.detach(), to_cpu=True)
        except BaseException:
            self.release(size)
            raise
        value = _Saved(cpu, tensor.device, self, self.pending, len(self.order))
        self.order.append(weakref.ref(value))
        self.stats["offloaded"] += 1
        return value

    def _prefetch(self, index):
        previous = None if self.prefetched is None else self.prefetched()
        if previous is not None and previous.prefetched is not None:
            return
        while index >= 0:
            value = self.order[index]()
            if value is not None and not value.unpacked:
                break
            index -= 1
        else:
            return
        size = value.tensor.numel()*value.tensor.element_size()
        free, _ = torch.cuda.mem_get_info(value.device)
        reserved = torch.cuda.memory_reserved(value.device)
        # Reserve recomputation workspace as well as the prefetched boundary;
        # allocator fragmentation must not let prefetch exceed the peak budget.
        margin = max(GIB, 8*size)
        if free < size+margin or (self.gpu_budget is not None and reserved+size+margin > self.gpu_budget):
            self.stats["prefetch_skipped_budget"] += 1
            return
        value.ready.result()
        gpu = torch.empty_like(value.tensor, device=value.device)
        future = self._submit(value.tensor, gpu, to_cpu=False)
        value.ticket["future"] = future
        value.prefetched = (gpu, future)
        self.prefetched = weakref.ref(value)
        self.jobs.append(future)
        self.stats["prefetched"] += 1

    def unpack(self, value):
        if isinstance(value, torch.Tensor):
            return value
        value.ready.result()
        if value.prefetched is None:
            gpu = torch.empty_like(value.tensor, device=value.device)
            future = self._submit(value.tensor, gpu, to_cpu=False)
            value.ticket["future"] = future
        else:
            gpu, future = value.prefetched
            value.prefetched = None
            self.stats["prefetch_hits"] += 1
        future.result()
        value.unpacked = True
        self._prefetch(value.index-1)
        return gpu

    def _transfer(self, cpu, gpu, *, to_cpu, ready):
        with torch.cuda.device(gpu.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            cpu_bytes = cpu.reshape(-1).view(torch.uint8)
            gpu_bytes = gpu.contiguous().reshape(-1).view(torch.uint8)
            slot_size = self.staging_size//2
            pending = []
            def finish(item):
                start, count, slot, event = item
                event.synchronize()
                if to_cpu:
                    cpu_bytes[start:start+count].copy_(slot[:count])
            try:
                for index, start in enumerate(range(0, len(cpu_bytes), slot_size)):
                    if len(pending) == 2:
                        finish(pending.pop(0))
                    slot = self.staging[(index % 2)*slot_size:(index % 2+1)*slot_size]
                    count = min(slot_size, len(cpu_bytes)-start)
                    if to_cpu:
                        slot[:count].copy_(gpu_bytes[start:start+count], non_blocking=True)
                    else:
                        slot[:count].copy_(cpu_bytes[start:start+count])
                        gpu_bytes[start:start+count].copy_(slot[:count], non_blocking=True)
                    event = self.copy_events[index % 2]
                    event.record(self.stream)
                    pending.append((start, count, slot, event))
                for item in pending:
                    finish(item)
                gpu.record_stream(self.stream)
            except BaseException:
                self.stream.synchronize()
                raise

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


def no_offload():
    return contextlib.nullcontext()
