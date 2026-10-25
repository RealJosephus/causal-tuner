"""Real safetensors and official Hub file-object consumers, without network."""

import hashlib
import io
import json

import pytest
import torch
from safetensors.torch import load_file

from causal_tuner.checkpointing.io import TensorWriter, read_manifest, save_checkpoint, restore_checkpoint
from causal_tuner.checkpointing.stream import DTYPES, TensorReader, TensorStream, write_generated_safetensors


def test_many_weights_moments_and_scalar_steps_share_one_file(tmp_path):
    values = {}
    for i in range(560):
        for prefix in ("model", "exp_avg", "exp_avg_sq"):
            values[f"{prefix}/{i}"] = torch.tensor([float(i)])
        values[f"step/{i}"] = torch.tensor(3.)
    values["rng/torch"], values["rng/cuda/0"] = torch.arange(32, dtype=torch.uint8), torch.zeros(64, dtype=torch.uint8)
    with TensorWriter(tmp_path, chunk_bytes=4096) as writer:
        for name, value in values.items():
            writer.write(name, value)
        writer.finish()
    assert len(values) == 2242
    assert list(writer.files) == ["state.safetensors"]
    assert len(list(tmp_path.iterdir())) == 1
    actual = load_file(tmp_path/"state.safetensors")
    assert actual.keys() == values.keys()
    for name in values:
        torch.testing.assert_close(actual[name], values[name], rtol=0, atol=0)


def test_stream_dtypes_shapes_seek_hash_and_bounded_reads(tmp_path):
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.lfs import SliceFileObj
    values = {str(dtype): torch.ones((3, 5), dtype=dtype) for dtype in DTYPES}
    values.update(large=torch.arange(65536, dtype=torch.float32), scalar=torch.tensor(42.), empty=torch.empty(0, 3))
    with TensorWriter(tmp_path, chunk_bytes=1024) as writer:
        for name, value in values.items():
            writer.write(name, value)
        writer.finish()
    expected = (tmp_path/"state.safetensors").read_bytes()
    assert len(expected) > 100*1024
    with TensorStream(values.items(), chunk_bytes=1024) as stream:
        operation = CommitOperationAdd(path_in_repo="state.safetensors", path_or_fileobj=stream)
        assert operation.upload_info.size == len(expected)
        assert operation.upload_info.sha256.hex() == hashlib.sha256(expected).hexdigest()
        assert operation.upload_info.sample == expected[:512]
        assert stream.tell() == 0
        with pytest.raises(ValueError, match="bounded reads"):
            stream.read()
        # Hub multipart views seek over tensor and header boundaries and may
        # replay a slice after a failed request. Compare exact serialized bytes.
        for offset, count in [(0, 8000), (15, 5555), (len(stream.header)-3, 9000), (len(expected)-20, 100)]:
            with SliceFileObj(stream, offset, count) as part:
                for _ in range(2):
                    part.seek(0)
                    chunks = []
                    while block := part.read(65536):
                        assert len(block) <= 1024
                        chunks.append(block)
                    assert b"".join(chunks) == expected[offset:offset+count]
        assert stream.seek(7, io.SEEK_END) == len(expected)+7
        assert stream.read(12) == b""
        stream.seek(0)
        with pytest.raises(ValueError):
            stream.seek(-1)
    with (tmp_path/"state.safetensors").open("rb") as handle:
        reader = TensorReader(handle, chunk_bytes=1031)
        for name, value in values.items():
            target = torch.empty_like(value)
            reader.load(name, writer.entries[name], target)
            torch.testing.assert_close(target, value, rtol=0, atol=0)
    for name, value in load_file(tmp_path/"state.safetensors").items():
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)


def test_reader_never_requests_a_full_large_tensor(tmp_path):
    values = {"weight": torch.arange(65536, dtype=torch.float32)}
    with TensorWriter(tmp_path, chunk_bytes=4096) as writer:
        writer.write("weight", values["weight"])
        writer.finish()
    class Observed(io.BytesIO):
        def readinto(self, buffer):
            assert len(buffer) <= 4096
            return super().readinto(buffer)
    handle = Observed((tmp_path/"state.safetensors").read_bytes())
    target = torch.empty_like(values["weight"])
    TensorReader(handle, chunk_bytes=4096).load("weight", writer.entries["weight"], target)
    torch.testing.assert_close(target, values["weight"], rtol=0, atol=0)


def test_generated_tensors_share_one_file_with_bounded_staging(tmp_path, monkeypatch):
    import weakref
    specifications = {str(dtype): torch.empty((3, 5), dtype=dtype, device="meta") for dtype in DTYPES}
    specifications.update(large=torch.empty(65536, dtype=torch.float32, device="meta"),
                          scalar=torch.empty((), device="meta"), empty=torch.empty(0, 3, device="meta"))
    calls, previous = [], []
    def materialize(name):
        assert not previous or previous[-1]() is None
        calls.append(name)
        spec = specifications[name]
        value = torch.ones(spec.shape, dtype=spec.dtype)
        previous.append(weakref.ref(value))
        return value
    original = TensorStream.read
    def bounded_read(self, size=-1):
        assert 0 <= size <= 1024
        return original(self, size)
    monkeypatch.setattr(TensorStream, "read", bounded_read)
    path = tmp_path/"model.safetensors"
    write_generated_safetensors(path, specifications.items(), materialize, chunk_bytes=1024)
    assert len(calls) == len(set(calls)) == len(specifications)
    actual = load_file(path)
    assert actual.keys() == specifications.keys()
    for name, value in actual.items():
        spec = specifications[name]
        torch.testing.assert_close(value, torch.ones(spec.shape, dtype=spec.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["shape", "dtype"])
def test_generated_tensor_must_match_its_specification(tmp_path, problem):
    value = torch.ones(2) if problem == "shape" else torch.ones(3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shape or dtype mismatch"):
        write_generated_safetensors(tmp_path/"model.safetensors", [("weight", torch.empty(3, device="meta"))],
                                    lambda name: value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_preserves_all_adam_state_and_rng_with_bounded_load(tmp_path, dtype, device):
    import copy
    import random
    import numpy as np
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = torch.nn.Linear(9, 7, dtype=dtype, device=device)
    initial = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), fused=device == "cuda")
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    options = dict(step=1, cursor={"batch": 1}, scheduler={"complete": False}, signature={}, save_optimizer=True)
    directory = save_checkpoint(tmp_path, model, optimizer, **options)
    manifest = read_manifest(directory)
    assert manifest["version"] == 1
    expected = random.random(), np.random.rand(), torch.rand(4)
    expected_cuda = torch.rand(4, device="cuda") if device == "cuda" else None
    restored = torch.optim.AdamW(initial.parameters(), fused=device == "cuda")
    assert restore_checkpoint(directory, manifest, initial, restored, chunk_bytes=32) == (1, {"batch": 1})
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
    if expected_cuda is not None:
        torch.testing.assert_close(torch.rand(4, device="cuda"), expected_cuda, rtol=0, atol=0)
    assert len(manifest["files"]) == 1
    for left, right in zip(model.parameters(), initial.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert optimizer.state[left].keys() == restored.state[right].keys()
        for name, value in optimizer.state[left].items():
            torch.testing.assert_close(value, restored.state[right][name], rtol=0, atol=0)


def test_invalid_tensor_layout_and_tampered_manifest_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        TensorStream([("bad", torch.ones(3, 4).T)])
    with TensorWriter(tmp_path) as writer:
        writer.write("weight", torch.ones(3))
        writer.finish()
    with (tmp_path/"state.safetensors").open("rb") as handle:
        reader = TensorReader(handle)
        with pytest.raises(ValueError, match="shape or dtype"):
            reader.load("weight", {**writer.entries["weight"], "dtype": "torch.float16"}, torch.empty(3))


def test_official_hub_preupload_and_commit_with_stream_and_only_network_replaced(monkeypatch):
    import math
    from types import SimpleNamespace
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub import _commit_api, hf_api, lfs
    values = [("weight", torch.arange(12345, dtype=torch.float32)), ("step", torch.tensor(4.))]
    with TensorStream(values, chunk_bytes=1024) as source:
        expected = b"".join(iter(lambda: source.read(1024), b""))
        source.seek(0)
        operation = CommitOperationAdd(path_in_repo="checkpoint/state.safetensors", path_or_fileobj=source)
        parts, commits = {}, []
        def response(data=None, headers=None):
            return SimpleNamespace(status_code=200, headers=headers or {}, json=lambda: data,
                                   raise_for_status=lambda: None)
        def post(url, *, json=None, data=None, **kwargs):
            if "/preupload/" in url:
                return response({"files": [{"path": f["path"],
                                           "uploadMode": "regular" if f["path"].endswith(".json") else "lfs", "shouldIgnore": False}
                                           for f in json["files"]]})
            if url.endswith("/objects/batch"):
                assert "xet" not in json["transfers"]
                header = {"chunk_size": "4096", **{str(i+1): f"https://upload.invalid/{i}"
                           for i in range(math.ceil(len(expected)/4096))}}
                return response({"transfer": "multipart", "objects": [{**json["objects"][0],
                    "actions": {"upload": {"href": "https://upload.invalid/complete", "header": header}}}]})
            if url.endswith("/complete"):
                assert len(json["parts"]) == len(parts)
                return response()
            if "/commit/" in url:
                commits.append(data)
                return response({"commitOid": "a"*40, "commitUrl": "https://huggingface.co/owner/run/commit/"+"a"*40})
            raise AssertionError(f"Unexpected network request: {url}")
        def transport(method, url, *, data=None, **kwargs):
            if method == "POST":
                return post(url, data=data, **kwargs)
            assert method == "PUT"
            blocks = []
            while block := data.read(65536):
                assert len(block) <= 1024
                blocks.append(block)
            index = int(url.rsplit("/", 1)[1])
            parts[index] = b"".join(blocks)
            return response(headers={"etag": str(index)})
        session = SimpleNamespace(post=post)
        for module in (_commit_api, hf_api, lfs):
            monkeypatch.setattr(module, "get_session", lambda: session, raising=False)
            monkeypatch.setattr(module, "http_backoff", transport, raising=False)
        api = HfApi(endpoint="https://hub.invalid", token=False)
        api.preupload_lfs_files("owner/run", additions=[operation])
        assert operation._is_uploaded and operation.path_or_fileobj == b""
        assert b"".join(parts[i] for i in sorted(parts)) == expected
        committed = api.create_commit("owner/run", operations=[operation,
            CommitOperationAdd(path_in_repo="checkpoint/manifest.json", path_or_fileobj=b'{"committed":true}')],
            parent_commit="0"*40, commit_message="Complete checkpoint")
        assert committed.oid == "a"*40 and len(commits) == 1
