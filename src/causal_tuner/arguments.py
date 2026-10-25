"""CUDA-free CLI parsing and executable option validation."""

import argparse
import json
import math
from pathlib import Path


PREPARE_ARGUMENTS = {
    "repo_id", "revision", "token", "output_dir", "max_sequence_length", "seed",
    "packing", "packing_batch_size", "assistant_only_loss", "last_assistant_only_loss", "data_buffer_budget_mib",
    "hub_dataset_repo_id", "hub_private", "hub_gated",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "y", "on", "1"}:
        return True
    if normalized in {"false", "no", "n", "off", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def parser(prepare=False):
    result = argparse.ArgumentParser(prog="causal-tune-prepare" if prepare else "causal-tune",
                                     description="Prepare signed training data." if prepare else "Single-device decoder training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    schema = json.loads(Path(__file__).with_name("arguments.json").read_text())
    result.set_defaults(**{name: entry["default"] for name, entry in schema.items()})
    for name, entry in schema.items():
        if not prepare and name == "hub_dataset_repo_id":
            continue
        if prepare and name not in PREPARE_ARGUMENTS and not name.startswith(("dataset_", "processor_", "hf_retry_", "endprompt_")):
            continue
        if prepare and name == "dataset_format":
            continue
        flags = list(dict.fromkeys(["--"+name, "--"+name.replace("_", "-")]))
        options = dict(dest=name, help=entry.get("help", name.replace("_", " ").capitalize()))
        if entry["type"] == "bool":
            options.update(type=parse_bool, nargs="?", const=True)
        else:
            options["type"] = {"int": int, "float": float, "str": str}[entry["type"]]
        for key in ("choices", "required"):
            if key in entry:
                options[key] = entry[key]
        result.add_argument(*flags, **options)
        if entry["type"] == "bool":
            result.add_argument("--no-"+name.replace("_", "-"), dest=name, action="store_false")
    if not prepare:
        result.add_argument("--stop-after-updates", "--stop_after_updates", type=int, default=None,
                            help="Save and stop after this many updates in this invocation, keeping the original schedule")
    return result


def validate(a, *, prepare=False):
    for name, value in vars(a).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 2 <= a.max_sequence_length <= 524288:
        raise ValueError("max_sequence_length must be in [2, 524288]")
    if not 0 <= a.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if a.assistant_only_loss and a.last_assistant_only_loss:
        raise ValueError("Choose one assistant loss mode")
    if a.dataset_format == "raw" and (a.assistant_only_loss or a.last_assistant_only_loss) and a.dataset_text_field != "messages":
        raise ValueError("Assistant loss modes require dataset_text_field=messages")
    if a.hf_retry_initial_delay > a.hf_retry_max_delay:
        raise ValueError("HF initial retry delay must not exceed the maximum")
    if a.endprompt_enable:
        if a.endprompt_logical_length is None:
            a.endprompt_logical_length = a.max_sequence_length
        if a.endprompt_logical_length_min is None:
            a.endprompt_logical_length_min = a.endprompt_logical_length
        if not a.max_sequence_length <= a.endprompt_logical_length_min <= a.endprompt_logical_length:
            raise ValueError("EndPrompt logical lengths must cover the physical length and be ordered")
        if not any(a.endprompt_prompts.split("||")):
            raise ValueError("EndPrompt needs at least one nonempty prompt")
        if a.endprompt_prompt_loss_weight == a.endprompt_context_loss_weight == 0:
            raise ValueError("At least one EndPrompt loss weight must be positive")
    if not prepare and a.dtype == "float32" and a.attn_mechanism != "vanilla":
        raise ValueError("FP32 compute requires the small-shape vanilla backend; FlashAttention 2 requires BF16")
    for name in ("max_sequence_length", "total_batch_size", "gradient_accumulation_steps", "lora_rank", "loss_token_budget",
                 "remat_group_size", "streaming_shuffle_buffer", "packing_batch_size", "logging_steps", "data_buffer_budget_mib",
                 "learning_rate", "adam_epsilon", "host_memory_budget_gib",
                 "checkpoint_host_budget_gib", "pinned_memory_budget_gib", "hf_retry_initial_delay", "hf_retry_max_delay"):
        if getattr(a, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("learning_rate_end", "weight_decay", "max_grad_norm", "warmup_steps", "save_steps",
                 "save_total_limit", "cut_chunk_size", "projection_chunk_size", "prefetch_batches", "gpu_memory_budget_gib",
                 "endprompt_prompt_loss_weight", "endprompt_context_loss_weight", "microbatch_token_budget"):
        if getattr(a, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if prepare:
        a.shuffle = False
        return a
    if a.hub_model_repo_id and not a.save_optimizer_state:
        raise ValueError("Hub training checkpoints require --save-optimizer-state for exact continuation")
    if a.scheduler != "constant" and a.learning_rate_end > a.learning_rate:
        raise ValueError("learning_rate_end must not exceed learning_rate for a decay schedule")
    if a.max_steps != -1 and a.max_steps <= 0:
        raise ValueError("max_steps must be positive or -1")
    if a.max_steps < 0 and a.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive unless max_steps is set")
    if a.dataset_streaming and a.max_steps < 0 and not a.num_train_epochs.is_integer():
        raise ValueError("Epoch-driven streaming requires an integer num_train_epochs")
    if a.stop_after_updates is not None and a.stop_after_updates <= 0:
        raise ValueError("stop_after_updates must be positive")
    if not 0 <= a.adam_beta1 < 1 or not 0 <= a.adam_beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if not 0 <= a.maskgit_probability <= 1:
        raise ValueError("maskgit_probability must be in [0, 1]")
    if a.maskgit_probability and a.dataset_format == "raw" and a.dataset_text_field != "messages":
        raise ValueError("MaskGIT requires messages")
    if a.gradient_checkpointing == "disabled" and (a.remat_group_size > 1 or a.activation_offload == "cpu"):
        raise ValueError("Nested checkpointing and CPU offload require gradient checkpointing")
    if a.mlp_chunk_size is not None and a.mlp_chunk_size < 0:
        raise ValueError("mlp_chunk_size must be non-negative")
    for name in ("mlp_cache_budget_gib", "attention_cache_budget_gib"):
        if getattr(a, name) is not None and getattr(a, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if a.mlp_cache_budget_gib and (not a.lora or a.gradient_checkpointing == "enabled"):
        raise ValueError("MLP caching requires LoRA and disabled or automatic gradient checkpointing")
    if a.attention_cache_budget_gib and (a.gradient_checkpointing == "disabled" or a.remat_group_size != 1
                                       or a.attn_mechanism != "flash_attention_2"):
        raise ValueError("Attention caching requires single-layer FA2 checkpointing")
    if a.loss_implementation != "cut" and a.cut_chunk_size:
        raise ValueError("cut_chunk_size applies only to the cut backend")
    if a.lora_save_adapter_only is None:
        a.lora_save_adapter_only = a.lora
    if not a.lora and a.lora_save_adapter_only:
        raise ValueError("Adapter-only saving requires LoRA")
    if a.lora and a.save_optimizer_state and not a.lora_save_adapter_only:
        raise ValueError("Resumable LoRA checkpoints require adapter-only saving")
    for name in ("lora_train_embed", "lora_train_lm_head"):
        value = getattr(a, name)
        if a.lora_train_embed_and_lm_head and value is False:
            raise ValueError("The joint table-training flag conflicts with an explicit false table flag")
        setattr(a, name, a.lora_train_embed_and_lm_head if value is None else value)
    if not a.lora and (a.lora_train_embed or a.lora_train_lm_head):
        raise ValueError("LoRA table flags require LoRA; Full uses frozen_parameters")
    a.frozen_components = tuple(x.strip() for x in (a.frozen_parameters or "").replace("|", ",").split(",") if x.strip())
    if a.frozen_parameters is not None and not a.frozen_components:
        raise ValueError("frozen_parameters must name at least one component")
    if set(a.frozen_components)-{"embed", "lm_head", "norm"}:
        raise ValueError("frozen_parameters accepts embed, lm_head, and norm")
    if (a.lora_train_embed and "embed" in a.frozen_components) or (a.lora_train_lm_head and "lm_head" in a.frozen_components):
        raise ValueError("A table cannot be both frozen and trainable")
    if a.loss_implementation == "cut" and (a.lora_train_lm_head or (not a.lora and "lm_head" not in a.frozen_components)):
        raise ValueError("cut requires a frozen lm_head; use fused for head training")
    if a.quantization == "nf4" and not a.lora:
        raise ValueError("NF4 requires frozen-base LoRA")
    if a.checkpoint_host_budget_gib > a.host_memory_budget_gib or a.pinned_memory_budget_gib > a.host_memory_budget_gib:
        raise ValueError("Checkpoint and pinned budgets are part of the total host budget")
    return a


def parse_args(argv=None, *, prepare=False):
    p = parser(prepare)
    try:
        return validate(p.parse_args(argv), prepare=prepare)
    except ValueError as exc:
        p.error(str(exc))
