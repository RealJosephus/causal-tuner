"""Activation policies selected from the model configuration and execution pack."""

from dataclasses import dataclass
import torch

GIB = 1024**3


@dataclass(frozen=True)
class ExecutionPolicy:
    checkpoint: bool
    mlp_tile: int
    mlp_cache_bytes: int
    attention_cache_bytes: int
    calibrated: bool


def execution_policy(config, runtime, rows, longest, *, cuda, offload, quantized, grouped_attention=False,
                     available_bytes=None):
    calibrated = (runtime.adaptive and cuda and runtime.checkpoint and runtime.group_size == 1
        and runtime.rank == 256 and not runtime.train_embed and not runtime.train_head and not quantized
        and runtime.base_dtype == runtime.compute_dtype == torch.bfloat16 and runtime.train_dtype == torch.float32
        and runtime.attention == "flash_attention_2"
        and (config.hidden_size, config.intermediate_size, config.vocab_size, config.num_hidden_layers,
             config.num_attention_heads, config.head_dim) == (4096, 13568, 185600, 40, 32, 128))
    checkpoint, mlp_budget, attention_budget = runtime.checkpoint, 0, 0
    tile = 4096 if rows >= 32768 else 0
    if calibrated:
        checkpoint = rows > 32768
        if rows <= 12288:
            tile, mlp_budget = 0, 28*GIB
        elif rows <= 16384:
            tile, mlp_budget = 0, 24*GIB
        else:
            tile = 4096
        if available_bytes is not None:
            size = torch.empty((), dtype=runtime.compute_dtype).element_size()
            state_size = torch.empty((), dtype=runtime.train_dtype).element_size()
            # Reserve gradients and both Adam moments even before their first
            # allocation. Resident layers also retain norm/projection inputs,
            # FA2 tensors and MQA keys/values outside the optional MLP cache.
            state = 3*config.parameter_counts(runtime.rank)[1]*state_size
            kv = 2*config.num_key_value_heads*config.head_dim
            resident = rows*((6*config.hidden_size+kv)*size+8)*config.num_hidden_layers
            def workspace(chunk):
                chunk = chunk if runtime.mlp_tile is None else runtime.mlp_tile
                return (rows*(8*config.hidden_size+kv)*size
                        +3*min(chunk or rows, rows)*config.intermediate_size*size)
            if not checkpoint and state+resident+workspace(tile) > available_bytes:
                checkpoint, tile = True, 4096
            remaining = max(0, available_bytes-state-workspace(tile)
                            -(2*rows*config.hidden_size*size if checkpoint else resident))
            mlp_budget = min(mlp_budget, remaining)
        if checkpoint and offload and longest > 32768 and rows <= 131072 and not grouped_attention:
            attention_budget = 38*GIB
            if available_bytes is not None:
                attention_budget = min(attention_budget, remaining)
    if runtime.mlp_tile is not None:
        tile = runtime.mlp_tile
    if runtime.mlp_cache_budget_bytes is not None:
        mlp_budget = runtime.mlp_cache_budget_bytes
    if runtime.attention_cache_budget_bytes is not None:
        attention_budget = runtime.attention_cache_budget_bytes
    return ExecutionPolicy(checkpoint, tile, 0 if checkpoint else mlp_budget,
                           attention_budget if checkpoint and cuda else 0, calibrated)
