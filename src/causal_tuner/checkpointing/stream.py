"""Standard safetensors with bounded staging, independent of file size."""

from bisect import bisect_right
import io
import json
import math
import os
from pathlib import Path
import struct

import torch

from ..training.memory import GIB, check_host


DTYPES = {torch.bool: "BOOL", torch.uint8: "U8", torch.int8: "I8", torch.int16: "I16",
          torch.int32: "I32", torch.int64: "I64", torch.float16: "F16",
          torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64"}
CHUNK_BYTES = 64*1024**2
MAX_HEADER_BYTES = 100_000_000  # safetensors' format reader limit


class TensorStream(io.BufferedIOBase):
    """Seekable serialization of stable tensors, suitable for CommitOperationAdd.

    Sized reads may be short, as with other binary streams. Large unbounded
    reads are rejected; Hub hashing and HTTP/multipart uploads use sized reads.
    The caller must keep tensors unchanged until upload and commit finish.
    """

    def __init__(self, tensors, *, chunk_bytes=CHUNK_BYTES, host_budget=72*GIB):
        super().__init__()
        if chunk_bytes < 8:
            raise ValueError("Tensor staging must allow at least eight bytes")
        self.limit, self.budget, self.position = chunk_bytes, host_budget, 0
        header, offset = {"__metadata__": {"format": "pt"}}, 0
        self.tensors, self.ends = [], []
        for name, tensor in sorted(tensors, key=lambda item: (-item[1].element_size(), item[0])):
            if name in header or tensor.dtype not in DTYPES or not tensor.is_contiguous():
                raise ValueError("Safetensors requires unique names and contiguous tensors with supported dtypes")
            length = tensor.numel()*tensor.element_size()
            header[name] = {"dtype": DTYPES[tensor.dtype], "shape": list(tensor.shape),
                            "data_offsets": [offset, offset+length]}
            offset += length
            if length:
                self.tensors.append(tensor.detach().view(-1).view(torch.uint8))
                self.ends.append(offset)
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" "*((-len(encoded)) % 8)
        if len(encoded) > MAX_HEADER_BYTES:
            raise ValueError("Safetensors header exceeds the format limit")
        self.header = struct.pack("<Q", len(encoded))+encoded
        self.size = len(self.header)+offset

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        self._checkClosed()
        if whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            raise ValueError("Invalid seek origin")
        position = offset + (0 if whence == os.SEEK_SET else self.position if whence == os.SEEK_CUR else self.size)
        if position < 0:
            raise ValueError("Cannot seek before the start of a tensor stream")
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        remaining = max(0, self.size-self.position)
        if size is None or size < 0:
            if remaining > self.limit:
                raise ValueError("Large tensor streams require bounded reads")
            size = remaining
        size = min(size, remaining, self.limit)
        check_host(self.budget-3*size)
        buffer = bytearray(size)
        self.readinto(buffer)
        return bytes(buffer)

    def readinto(self, buffer):
        self._checkClosed()
        output = memoryview(buffer).cast("B")
        count = min(len(output), max(0, self.size-self.position), self.limit)
        check_host(self.budget-count)
        written = 0
        if self.position < len(self.header):
            length = min(count, len(self.header)-self.position)
            output[:length] = self.header[self.position:self.position+length]
            written += length
            self.position += length
        while written < count:
            offset = self.position-len(self.header)
            index = bisect_right(self.ends, offset)
            start = self.ends[index-1] if index else 0
            length = min(count-written, self.ends[index]-offset)
            value = self.tensors[index][offset-start:offset-start+length].to("cpu", copy=True).numpy()
            output[written:written+length] = memoryview(value)
            del value
            written += length
            self.position += length
        return count

    def close(self):
        self.tensors = []
        super().close()


class TensorReader:
    """Read slices without mapping or copying an entire tensor/file into RAM."""

    def __init__(self, handle, *, chunk_bytes=CHUNK_BYTES, host_budget=72*GIB):
        if chunk_bytes < 8:
            raise ValueError("Tensor staging must allow at least eight bytes")
        self.handle, self.limit, self.budget = handle, chunk_bytes, host_budget
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= MAX_HEADER_BYTES:
            raise ValueError("Invalid safetensors header length")
        check_host(host_budget-4*length)
        self.header = json.loads(handle.read(length))
        self.start = 8+length
        offset = 0
        types = {code: dtype for dtype, code in DTYPES.items()}
        for name, spec in sorted(((n, s) for n, s in self.header.items() if n != "__metadata__"),
                                 key=lambda item: tuple(item[1]["data_offsets"])):
            shape, bounds = spec["shape"], spec["data_offsets"]
            if spec["dtype"] not in types or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("Invalid safetensors shape or dtype")
            size = math.prod(shape)*torch.empty((), dtype=types[spec["dtype"]]).element_size()
            if bounds != [offset, offset+size]:
                raise ValueError("Invalid safetensors offsets")
            offset += size
        if handle.seek(0, os.SEEK_END) != self.start+offset:
            raise ValueError("Incomplete safetensors data")

    def load(self, name, entry, target):
        spec = self.header.get(name)
        if (spec is None or list(target.shape) != entry["shape"] or str(target.dtype) != entry["dtype"]
                or spec["shape"] != entry["shape"] or spec["dtype"] != DTYPES.get(target.dtype)
                or not target.is_contiguous()):
            raise ValueError("Checkpoint tensor shape or dtype mismatch")
        self.handle.seek(self.start+spec["data_offsets"][0])
        flat = target.view(-1)
        elements = max(1, self.limit//target.element_size())
        with torch.no_grad():
            for start in range(0, flat.numel(), elements):
                count = min(elements, flat.numel()-start)
                length = count*target.element_size()
                check_host(self.budget-length)
                buffer = bytearray(length)
                if self.handle.readinto(buffer) != length:
                    raise ValueError("Checkpoint tensor is incomplete")
                value = torch.frombuffer(buffer, dtype=target.dtype)
                flat[start:start+count].copy_(value)
                del value, buffer


def write_safetensors(path, tensors, *, chunk_bytes=CHUNK_BYTES, host_budget=72*GIB):
    import hashlib
    digest = hashlib.sha256()
    with TensorStream(tensors, chunk_bytes=chunk_bytes, host_budget=host_budget) as source, Path(path).open("wb") as output:
        while block := source.read(chunk_bytes):
            output.write(block)
            digest.update(block)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest()


def write_generated_safetensors(path, specifications, materialize, *, chunk_bytes=CHUNK_BYTES, host_budget=72*GIB):
    """Write one file, materializing and releasing one tensor at a time.

    Specifications may be meta tensors: only their shapes and dtypes are used.
    """
    specifications = sorted(specifications, key=lambda item: (-item[1].element_size(), item[0]))
    options = dict(chunk_bytes=chunk_bytes, host_budget=host_budget)
    with TensorStream(specifications, **options) as layout, Path(path).open("wb") as output:
        output.write(layout.header)
        for name, spec in specifications:
            value = materialize(name)
            if value.shape != spec.shape or value.dtype != spec.dtype:
                raise ValueError(f"Generated tensor shape or dtype mismatch for {name}")
            with TensorStream([(name, value)], **options) as source:
                source.seek(len(source.header))
                while block := source.read(chunk_bytes):
                    output.write(block)
            del value
        output.flush()
        os.fsync(output.fileno())
