"""Safe asset resolution and one-tensor-at-a-time standard HF loading."""

import json
import math
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from .io import file_hash
from ..modeling.lora import Projection
from ..training.memory import check_host, GIB
from ..hub import retry
from ..data.tokenizer import TOKENIZER_PATTERNS


def resolve_assets(repo, revision=None, token=None, *, weights=True, adapter=False, initial_delay=1, max_delay=60):
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import HfApi, snapshot_download
    options = dict(initial_delay=initial_delay, max_delay=max_delay)
    revision = retry(lambda: HfApi(token=token).model_info(repo, revision=revision).sha, **options)
    patterns = ["adapter_config.json" if adapter else "config.json", *TOKENIZER_PATTERNS]
    if weights:
        patterns += ["adapter_model.safetensors"] if adapter else ["*.safetensors", "*.safetensors.index.json"]
    return Path(retry(lambda: snapshot_download(repo_id=repo, revision=revision, token=token, allow_patterns=patterns), **options))


def weight_files(directory):
    directory = Path(directory)
    index = directory/"model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    elif (directory/"model.safetensors").exists():
        with safe_open(directory/"model.safetensors", framework="pt") as handle:
            mapping = {name: "model.safetensors" for name in handle.keys()}
    else:
        raise ValueError("Model assets must include standard safetensors weights")
    for filename in mapping.values():
        if (directory/filename).resolve().parent != directory.resolve():
            raise ValueError("Weight index contains an unsafe shard path")
    return mapping


def base_identity(directory):
    mapping = weight_files(directory)
    return {"config": file_hash(Path(directory)/"config.json"),
            "weights": {f: file_hash(Path(directory)/f) for f in sorted(set(mapping.values()))}}


def load_weights(model, directory, *, device, quantization="none", host_budget=72*GIB):
    directory = Path(directory)
    mapping = weight_files(directory)
    expected = {name for name, _ in model.named_parameters() if ".lora_" not in name}
    if set(mapping) != expected:
        raise ValueError(f"HF parameter keys mismatch: missing={sorted(expected-set(mapping))}, unexpected={sorted(set(mapping)-expected)}")
    specifications = list(model.named_parameters())
    for name, param in specifications:
        parent_name, field = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        if ".lora_" in name:
            value = torch.empty(param.shape, device=device, dtype=param.dtype)
            if field == "lora_A":
                nn.init.uniform_(value, -math.sqrt(6/value.shape[1]), math.sqrt(6/value.shape[1]))
            else:
                value.zero_()
            setattr(parent, field, nn.Parameter(value, requires_grad=True))
            continue
        check_host(host_budget-param.numel()*max(param.element_size(), 2))
        with safe_open(directory/mapping[name], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
            if value.shape != param.shape:
                raise ValueError(f"Invalid tensor shape for {name}")
            if quantization == "nf4" and field == "weight" and isinstance(parent, Projection):
                parent.install_nf4(value, device)
            else:
                setattr(parent, field, nn.Parameter(value.to(device=device, dtype=param.dtype, copy=True),
                                                   requires_grad=param.requires_grad))
    return model
