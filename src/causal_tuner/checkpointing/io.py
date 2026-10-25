"""Atomic training checkpoints and standard weight exports with bounded staging."""

import hashlib
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
from ..training.memory import GIB, check_host
from .stream import CHUNK_BYTES, TensorReader, TensorStream, write_generated_safetensors, write_safetensors


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def signatures_match(saved, current):
    """Match the current compatibility fields against saved metadata."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    return all(key in saved and saved[key] == value for key, value in current.items())


def json_write(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_path(root, name):
    path = (root/name).resolve()
    if path.parent != root.resolve():
        raise ValueError("Manifest contains a path outside its checkpoint directory")
    return path


class TensorWriter:
    def __init__(self, directory=None, *, chunk_bytes=CHUNK_BYTES, host_budget=72*GIB, put=None):
        self.directory = Path(directory) if directory is not None else None
        self.limit, self.budget, self.put = chunk_bytes, host_budget, put
        self.entries = {}
        self.files = {}
        self.tensors = {}
        self.stack = ExitStack()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def write(self, name, tensor):
        if name in self.tensors or self.files:
            raise ValueError("Cannot duplicate tensors or append to a finished checkpoint")
        self.tensors[name] = tensor.detach()
        self.entries[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "file": "state.safetensors"}

    def finish(self):
        filename = "state.safetensors"
        options = dict(chunk_bytes=self.limit, host_budget=self.budget)
        if self.put:
            source = self.stack.enter_context(TensorStream(self.tensors.items(), **options))
            # Keep the source open through commit, including small regular Git
            # files for which preupload_lfs_files does not consume the content.
            self.files[filename] = self.put(filename, source)
        else:
            self.files[filename] = write_safetensors(self.directory/filename, self.tensors.items(), **options)


def rng_save(writer):
    writer.write("rng/torch", torch.get_rng_state())
    for index, state in enumerate(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []):
        writer.write(f"rng/cuda/{index}", state)
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def _tuple(value):
    return tuple(map(_tuple, value)) if isinstance(value, list) else value


def checkpoint_manifest(writer, model, optimizer, *, step, cursor, signature, scheduler, save_optimizer=False, tracking=None,
                        export=None):
    names = {p: n for n, p in model.named_parameters() if p.requires_grad}
    for param, name in names.items():
        writer.write("model/"+name, param)
    state_metadata = {}
    if save_optimizer:
        for param, name in names.items():
            state = optimizer.state.get(param)
            if not state:
                raise ValueError(f"Missing optimizer state for trainable parameter {name}")
            state_metadata[name] = {}
            for key, value in state.items():
                if torch.is_tensor(value):
                    writer.write(f"optimizer/{name}/{key}", value)
                    state_metadata[name][key] = "tensor"
                else:
                    state_metadata[name][key] = value
    rng = rng_save(writer)
    writer.finish()
    return {"version": 1, "step": step, "cursor": cursor, "signature": signature, "export": export or {},
            "scheduler": scheduler, "optimizer_saved": save_optimizer, "tracking": tracking or {},
            "optimizer": state_metadata, "rng": rng, "tensors": writer.entries,
            "files": writer.files, "committed": True}


def save_checkpoint(output, model, optimizer, *, step, cursor, signature, scheduler,
                    save_optimizer=False, host_budget=72*GIB, keep=1, tracking=None, export=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output/f"checkpoint-{step:08d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    temporary = output/(".incomplete-"+uuid.uuid4().hex)
    temporary.mkdir()
    with TensorWriter(temporary, host_budget=host_budget) as writer:
        manifest = checkpoint_manifest(writer, model, optimizer, step=step, cursor=cursor, signature=signature,
                                       scheduler=scheduler, save_optimizer=save_optimizer, tracking=tracking, export=export)
    json_write(temporary/"manifest.json", manifest)
    temporary.replace(destination)
    if os.name != "nt":
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if keep:
        completed = sorted(p for p in output.glob("checkpoint-*") if (p/"manifest.json").is_file())
        for old in completed[:-keep]:
            # Only completed, direct checkpoint children created by this format.
            if old.resolve().parent == output.resolve() and read_manifest(old, verify=False)["version"] == 1:
                shutil.rmtree(old)
    return destination


def validate_manifest(manifest):
    if manifest.get("version") != 1 or manifest.get("committed") is not True:
        raise ValueError("Unsupported or uncommitted checkpoint")
    if any(not name or name != Path(name).name or "/" in name or "\\" in name or ":" in name or name in {".", ".."}
           for name in manifest["files"]):
        raise ValueError("Manifest contains a path outside its checkpoint directory")
    if any(entry["file"] not in manifest["files"] for entry in manifest["tensors"].values()):
        raise ValueError("Checkpoint tensor refers to an unsigned file")
    return manifest


def read_manifest(path, verify=True):
    manifest = validate_manifest(json.loads((Path(path)/"manifest.json").read_text()))
    if verify:
        for name, checksum in manifest["files"].items():
            if file_hash(safe_path(Path(path), name)) != checksum:
                raise ValueError("Checkpoint checksum mismatch")
    return manifest


def latest_checkpoint(output, signature):
    incompatible = False
    for path in sorted(Path(output).glob("checkpoint-*"), reverse=True):
        if not (path/"manifest.json").is_file():
            continue
        if not json.loads((path/"manifest.json").read_text()).get("committed"):
            continue
        manifest = read_manifest(path)
        if not signatures_match(manifest["signature"], signature):
            incompatible = True
            continue
        return path, manifest
    if incompatible:
        raise ValueError("No compatible checkpoint exists in output_dir; configuration, runtime, model, or data changed")
    return None


def restore_checkpoint(path, manifest, model, optimizer, *, require_optimizer=True, resolve=None, restore_rng=True,
                       host_budget=72*GIB, chunk_bytes=CHUNK_BYTES):
    if require_optimizer and (optimizer is None or not manifest["optimizer_saved"]):
        raise ValueError("Exact continuation requires saved optimizer state; this checkpoint cannot resume")
    path = Path(path) if path is not None else None
    entries = manifest["tensors"]
    expected = {"model/"+name for name, p in model.named_parameters() if p.requires_grad}
    if expected != {name for name in entries if name.startswith("model/")}:
        raise ValueError("Checkpoint trainable parameter set differs from the model")
    with ExitStack() as stack:
        readers = {}
        def load(name, target):
            entry = entries[name]
            filename = entry["file"]
            if filename not in readers:
                source = stack.enter_context(resolve(filename)) if resolve else safe_path(path, filename)
                handle = stack.enter_context(Path(source).open("rb"))
                readers[filename] = TensorReader(handle, chunk_bytes=chunk_bytes, host_budget=host_budget)
            readers[filename].load(name, entry, target)

        for name, param in model.named_parameters():
            if param.requires_grad:
                load("model/"+name, param)
                if optimizer is not None and manifest["optimizer_saved"]:
                    state = optimizer.state[param]
                    if set(manifest["optimizer"][name]) != {"step", "exp_avg", "exp_avg_sq"}:
                        raise ValueError("Checkpoint does not contain the required AdamW state")
                    for key, kind in manifest["optimizer"][name].items():
                        if kind == "tensor":
                            entry = entries[f"optimizer/{name}/{key}"]
                            device = "cpu" if key == "step" and not optimizer.defaults.get("fused") else param.device
                            expected_shape = [] if key == "step" else list(param.shape)
                            expected_dtype = "torch.float32" if key == "step" else str(param.dtype)
                            if entry["shape"] != expected_shape or entry["dtype"] != expected_dtype:
                                raise ValueError("Checkpoint AdamW state shape or precision does not match")
                            target = torch.empty(entry["shape"], dtype=getattr(torch, entry["dtype"].split(".")[-1]), device=device)
                            load(f"optimizer/{name}/{key}", target)
                            state[key] = target
                        else:
                            state[key] = kind
        if restore_rng:
            for key, entry in entries.items():
                if key.startswith("rng/"):
                    target = torch.empty(entry["shape"], dtype=torch.uint8)
                    load(key, target)
                    if key == "rng/torch":
                        torch.set_rng_state(target)
                    else:
                        if not torch.cuda.is_available():
                            raise ValueError("A CUDA RNG checkpoint cannot resume on CPU")
                        torch.cuda.set_rng_state(target, int(key.rsplit("/", 1)[1]))
            random.setstate(_tuple(manifest["rng"]["python"]))
            n = manifest["rng"]["numpy"]
            np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    return manifest["step"], manifest["cursor"]


def copy_tokenizer(source, destination):
    if source is not None:
        from ..data.tokenizer import tokenizer_files
        for path in tokenizer_files(source):
            target = destination/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def export_hf(model, destination, metadata, *, merge=True, host_budget=72*GIB, tokenizer_directory=None):
    """Export a single safetensors file, with no whole-model CPU copy."""
    from ..modeling.lora import Projection
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Export destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".export-"+uuid.uuid4().hex)
    temporary.mkdir()
    weights = {name+".weight": module for name, module in model.named_modules() if isinstance(module, Projection)}
    weights.update((name, value) for name, value in model.named_parameters()
                   if name not in weights and ".lora_" not in name and ".quantized." not in name)
    specifications = {}
    for name, value in weights.items():
        if isinstance(value, Projection):
            shape = (value.lora_B.shape[0], value.lora_A.shape[1]) if value.rank else value.weight.shape
        else:
            shape = value.shape
        specifications[name] = torch.empty(shape, dtype=model.runtime.base_dtype, device="meta")

    def materialize(name):
        value = weights[name]
        if isinstance(value, Projection):
            weight = value.base_weight()
            if merge and value.rank:
                # Only this projection's merged tensor exists at a time.
                weight = weight.float() + value.lora_B.float() @ value.lora_A.float()
        else:
            weight = value
        check_host(host_budget-weight.numel()*specifications[name].element_size())
        return weight.detach().to(model.runtime.base_dtype).contiguous()

    with torch.no_grad():
        write_generated_safetensors(temporary/"model.safetensors", specifications.items(), materialize, host_budget=host_budget)
    json_write(temporary/"config.json", {**metadata, **model.config.to_dict(), "torch_dtype": str(model.runtime.base_dtype).split(".")[-1]})
    json_write(temporary/"export_provenance.json", {"merged": bool(merge and model.runtime.rank),
               "base": "dequantized_nf4" if any(getattr(m, "quantized", None) is not None for m in model.modules()) else "unquantized"})
    copy_tokenizer(tokenizer_directory, temporary)
    temporary.replace(destination)


def export_adapter(model, destination, *, base_identity, host_budget=72*GIB, tokenizer_directory=None):
    if not model.runtime.rank:
        raise ValueError("Adapter export requires LoRA")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Adapter export destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".adapter-"+uuid.uuid4().hex)
    temporary.mkdir()
    tensors, tables = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(("lora_A", "lora_B")):
            key = "base_model.model."+name+".weight"
        elif name in {"model.embed_tokens.weight", "lm_head.weight"}:
            key = "base_model.model."+name
            tables.append("embed_tokens" if "embed_tokens" in name else "lm_head")
        else:
            raise ValueError("Unexpected trainable parameter in adapter export")
        tensors.append((key, param))
    write_safetensors(temporary/"adapter_model.safetensors", tensors, host_budget=host_budget)
    json_write(temporary/"adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_identity, "r": model.runtime.rank, "lora_alpha": model.runtime.rank,
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": tables or None, "inference_mode": True})
    json_write(temporary/"export_provenance.json", {"base": "nf4" if any(getattr(m, "quantized", None) is not None for m in model.modules()) else "unquantized"})
    copy_tokenizer(tokenizer_directory, temporary)
    temporary.replace(destination)
