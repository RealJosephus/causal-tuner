"""Merge an exported LoRA adapter into a single model.safetensors file."""

import argparse
import json
import math
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(prog="causal-tune-merge", description=__doc__)
    result.add_argument("--adapter", required=True, help="Local adapter directory or Hub model repository")
    result.add_argument("--adapter-revision", help="Hub adapter revision; omission resolves the current default branch")
    result.add_argument("--repo-id", help="Base model directory or Hub repository; defaults to the adapter configuration")
    result.add_argument("--revision", help="Base model revision; defaults to the adapter configuration")
    result.add_argument("--output-dir", required=True, help="New local directory for the merged model and tokenizer")
    result.add_argument("--device", default="cpu", help="Device for loading and merging weights, e.g. cpu or cuda:0")
    result.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                        help="Output and base storage dtype; defaults to the base configuration, then bfloat16")
    result.add_argument("--token", help="Hugging Face token; otherwise use the normal Hub credentials")
    result.add_argument("--host-memory-budget-gib", type=float, default=72)
    result.add_argument("--hf-retry-initial-delay", type=float, default=1)
    result.add_argument("--hf-retry-max-delay", type=float, default=60)
    return result


def adapter_settings(config):
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if type(rank) is not int or rank <= 0:
        raise ValueError("Adapter rank must be a positive integer")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Adapter lora_alpha must be finite and non-negative")
    targets = config.get("target_modules")
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not isinstance(targets, list) or any(not isinstance(t, str) for t in targets) or set(targets) != projections:
        raise ValueError("Adapter must target the seven decoder projections exported by Causal Tuner")
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
            or config.get("bias", "none") != "none"
            or any(config.get(key) for key in ("fan_in_fan_out", "use_dora", "use_rslora", "lora_bias",
                       "rank_pattern", "alpha_pattern", "layer_replication", "layers_to_transform",
                       "layers_pattern", "target_parameters"))):
        raise ValueError("Only standard LoRA adapters exported by Causal Tuner are supported")
    tables = config.get("modules_to_save") or []
    if (not isinstance(tables, list) or any(not isinstance(t, str) for t in tables)
            or not set(tables) <= {"embed_tokens", "lm_head"}):
        raise ValueError("Adapter modules_to_save may only contain embed_tokens and lm_head")
    return rank, alpha/rank, set(tables)


def run(options):
    import torch
    from safetensors import safe_open
    from .checkpointing.huggingface import resolve_assets, load_weights
    from .checkpointing.io import export_hf
    from .modeling.config import ModelConfig
    from .modeling.model import CausalLM, Runtime
    from .training.memory import GIB, check_host

    if not (0 < options.hf_retry_initial_delay <= options.hf_retry_max_delay < math.inf):
        raise ValueError("HF retry delays must be finite, positive and ordered")
    if not math.isfinite(options.host_memory_budget_gib) or not 0 < options.host_memory_budget_gib <= 72:
        raise ValueError("Host memory budget must be in (0, 72] GiB")
    budget = options.host_memory_budget_gib*GIB
    destination = Path(options.output_dir)
    if destination.exists():
        raise FileExistsError("Merge destination already exists")
    if Path(options.adapter).is_dir() and options.adapter_revision is not None:
        raise ValueError("Adapter revisions apply only to Hub repositories")
    resolve_options = dict(token=options.token, initial_delay=options.hf_retry_initial_delay,
                           max_delay=options.hf_retry_max_delay)
    adapter = resolve_assets(options.adapter, options.adapter_revision, adapter=True, **resolve_options)
    config = json.loads((adapter/"adapter_config.json").read_text(encoding="utf-8"))
    rank, scale, tables = adapter_settings(config)
    base_repo = options.repo_id or config.get("base_model_name_or_path")
    if not isinstance(base_repo, str) or not base_repo:
        raise ValueError("Specify --repo-id or set base_model_name_or_path in the adapter configuration")
    assets = resolve_assets(base_repo, options.revision or config.get("revision"), **resolve_options)
    if any(destination.resolve().is_relative_to(source.resolve()) for source in (adapter, assets)):
        raise ValueError("Merge destination must be outside the source adapter and base model")
    metadata = json.loads((assets/"config.json").read_text(encoding="utf-8"))
    dtype_name = options.dtype or metadata.get("torch_dtype") or metadata.get("dtype") or "bfloat16"
    if dtype_name not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Base dtype must be bfloat16, float16 or float32; override it with --dtype")
    if "dtype" in metadata:
        metadata["dtype"] = dtype_name
    dtype = getattr(torch, dtype_name)
    runtime = Runtime(rank=rank, base_dtype=dtype, compute_dtype=dtype, attention="vanilla",
                      train_embed="embed_tokens" in tables, train_head="lm_head" in tables)
    model = CausalLM(ModelConfig.from_dict(metadata), runtime, device="meta")
    expected = {"base_model.model."+name+(".weight" if name.endswith(("lora_A", "lora_B")) else ""): name
                for name, param in model.named_parameters() if param.requires_grad}
    with safe_open(adapter/"adapter_model.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError(f"Adapter parameter keys mismatch: missing={sorted(set(expected)-set(handle.keys()))}, "
                             f"unexpected={sorted(set(handle.keys())-set(expected))}")
        for key, name in expected.items():
            spec = handle.get_slice(key)
            if list(model.get_parameter(name).shape) != spec.get_shape() or spec.get_dtype() not in {"BF16", "F16", "F32"}:
                raise ValueError(f"Invalid adapter tensor shape or dtype for {key}")
        # Validate the complete adapter before allocating base weights. Load
        # adapter tensors individually, without restoring any training state.
        load_weights(model, assets, device=torch.device(options.device), host_budget=budget)
        with torch.no_grad():
            for key, name in expected.items():
                param = model.get_parameter(name)
                check_host(budget-param.numel()*4)
                value = handle.get_tensor(key)
                param.copy_(value)
                if name.endswith("lora_B"):
                    param.mul_(scale)
                del value
    model.eval().requires_grad_(False)
    tokenizer = adapter if (adapter/"tokenizer.json").is_file() else assets
    export_hf(model, destination, metadata, host_budget=budget, tokenizer_directory=tokenizer)
    return destination


def main(argv=None):
    print(run(parser().parse_args(argv)))


if __name__ == "__main__":
    main()
