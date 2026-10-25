"""Pre-normalized decoder with independent standard HF projection keys."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig
from .lora import Projection
from .rotary import rotary_values, rotate
from ..data.segments import canonical_segments, position_ids, runs
from ..kernels.attention import flash_dense, prepare_dense_layout
from ..kernels.reference import dense_attention as dense_reference
from ..kernels.tiled import tiled
from ..training.memory import no_offload


@dataclass
class Runtime:
    rank: int = 256
    base_dtype: torch.dtype = torch.bfloat16
    train_dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.bfloat16
    train_embed: bool = False
    train_head: bool = False
    frozen: tuple = ()
    attention: str = "flash_attention_2"
    mlp_tile: int | None = None
    mlp_cache_budget_bytes: int | None = None
    attention_cache_budget_bytes: int | None = None
    projection_tile: int = 0
    checkpoint: bool = True
    adaptive: bool = False
    group_size: int = 1
    inner_checkpoint: bool = True
    gpu_budget_bytes: int | None = None


class RMSNorm(nn.Module):
    def __init__(self, size, eps, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype, device=device))
        self.eps = eps
        self._reconstruction_cache = None

    def _can_reconstruct(self, dtype):
        # Output division amplifies rounding near zero. Limit reconstruction
        # to well-scaled frozen BF16 weights; cache by storage and version.
        if dtype != torch.bfloat16 or self.weight.requires_grad or torch.is_inference(self.weight):
            return False
        key = (self.weight.data_ptr(), self.weight._version, dtype)
        if self._reconstruction_cache is None or self._reconstruction_cache[0] != key:
            if torch.cuda.is_current_stream_capturing():
                return False
            magnitude = self.weight.detach().to(dtype).abs()
            safe = bool(((magnitude >= 1/256) & (magnitude <= 256)).all())
            self._reconstruction_cache = key, safe
        return self._reconstruction_cache[1]

    def forward(self, x, *, reuse_grad=False):
        if x.is_cuda:
            from ..kernels.core_cuda import RMS
            reconstruct = reuse_grad and torch.is_grad_enabled() and self._can_reconstruct(x.dtype)
            return RMS.apply(x, self.weight, self.eps, reuse_grad, reconstruct)
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)+self.eps)
        return (normalized.to(x.dtype) * self.weight.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, config, runtime, device):
        super().__init__()
        self.config, self.runtime = config, runtime
        kwargs = dict(rank=runtime.rank, base_dtype=runtime.base_dtype, train_dtype=runtime.train_dtype,
                      compute_dtype=runtime.compute_dtype, device=device)
        h, d = config.hidden_size, config.head_dim
        self.q_proj = Projection(h, h, bias=True, **kwargs)
        self.k_proj = Projection(h, d, bias=True, **kwargs)
        self.v_proj = Projection(h, d, bias=True, **kwargs)
        self.o_proj = Projection(h, h, **kwargs)

    def forward(self, x, cosine, sine, segments, spans, attention_end, layout=None, *, residual=None):
        b, s, _ = x.shape
        c, r = self.config, self.runtime
        def project(module):
            return tiled(module, x, r.projection_tile, module.parameters())
        from ..kernels.linear import can_fuse, project_group
        if all(can_fuse(module, x) for module in (self.q_proj, self.k_proj, self.v_proj)):
            q, k, v = project_group((self.q_proj, self.k_proj, self.v_proj), x, r.projection_tile,
                                    rotary=(cosine, sine, c.head_dim))
            q, k = q.reshape(b, s, c.num_attention_heads, c.head_dim), k.reshape(b, s, 1, c.head_dim)
        else:
            q, k, v = (project(module) for module in (self.q_proj, self.k_proj, self.v_proj))
            q = rotate(q.reshape(b, s, c.num_attention_heads, c.head_dim), cosine, sine)
            k = rotate(k.reshape(b, s, 1, c.head_dim), cosine, sine)
        v = v.reshape(b, s, 1, c.head_dim)
        if r.attention == "vanilla":
            y = dense_reference(q, k, v, segments, attention_end)
        else:
            if r.gpu_budget_bytes is not None:
                workspace = q.numel()*(3*q.element_size()+4) + 4*(k.numel()+v.numel())
                if torch.cuda.memory_allocated(q.device)+workspace > r.gpu_budget_bytes:
                    raise MemoryError("Dense attention workspace exceeds the remaining GPU budget; choose an explicit memory configuration")
            y = flash_dense(q, k, v, spans, attention_end, layout=layout)
        y = y.reshape(b, s, -1)
        if can_fuse(self.o_proj, y):
            return project_group((self.o_proj,), y, r.projection_tile, residual=residual)[0]
        output = tiled(self.o_proj, y, r.projection_tile, self.o_proj.parameters())
        return output if residual is None else output+residual


class MLP(nn.Module):
    def __init__(self, config, runtime, device):
        super().__init__()
        self.tile = runtime.mlp_tile
        self.cache_intermediates = False
        kwargs = dict(rank=runtime.rank, base_dtype=runtime.base_dtype, train_dtype=runtime.train_dtype,
                      compute_dtype=runtime.compute_dtype, device=device)
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj, self.up_proj = Projection(h, i, **kwargs), Projection(h, i, **kwargs)
        self.down_proj = Projection(i, h, **kwargs)

    def cache_bytes(self, rows):
        size = torch.empty((), dtype=self.gate_proj.compute_dtype).element_size()
        return rows*size*sum(projection.lora_B.shape[0]+projection.rank
                             for projection in (self.gate_proj, self.up_proj))

    def untiled(self, x):
        if x.is_cuda:
            from ..kernels.mlp_cuda import silu_multiply
            return self.down_proj(silu_multiply(self.gate_proj(x), self.up_proj(x)))
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

    def forward(self, x, *, residual=None, tile=None, cache=None):
        from ..kernels.linear import can_fuse
        if all(can_fuse(module, x) for module in (self.gate_proj, self.up_proj, self.down_proj)):
            from ..kernels.mlp_cuda import tiled_lora_mlp
            return tiled_lora_mlp(self, x, residual, tile=tile, cache=cache)
        output = tiled(self.untiled, x, (self.tile if tile is None else tile) or 0, self.parameters())
        return output if residual is None else output+residual


class Layer(nn.Module):
    def __init__(self, config, runtime, device):
        super().__init__()
        self.self_attn = Attention(config, runtime, device)
        self.mlp = MLP(config, runtime, device)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, runtime.base_dtype, device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, runtime.base_dtype, device)

    def forward(self, x, *args, mlp_tile=None, mlp_cache=None):
        from ..kernels.linear import can_fuse
        shared_qkv = all(can_fuse(module, x) for module in (self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj))
        x = self.self_attn(self.input_layernorm(x, reuse_grad=shared_qkv), *args, residual=x)
        fused_mlp = all(can_fuse(module, x) for module in (self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj))
        return self.mlp(self.post_attention_layernorm(x, reuse_grad=fused_mlp), residual=x,
                        tile=mlp_tile, cache=mlp_cache)


class Decoder(nn.Module):
    def __init__(self, config, runtime, device):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=config.pad_token_id, dtype=runtime.base_dtype, device=device)
        self.layers = nn.ModuleList([Layer(config, runtime, device) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, runtime.base_dtype, device)


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig, runtime: Runtime, device=None):
        super().__init__()
        self.config, self.runtime = config, runtime
        if any(value is not None and value < 0 for value in
               (runtime.mlp_cache_budget_bytes, runtime.attention_cache_budget_bytes)):
            raise ValueError("Activation cache budgets must be non-negative")
        if runtime.mlp_cache_budget_bytes and (not runtime.rank or (runtime.checkpoint and not runtime.adaptive)):
            raise ValueError("MLP caching requires LoRA and disabled or automatic gradient checkpointing")
        if runtime.attention_cache_budget_bytes and (not runtime.checkpoint or runtime.group_size != 1
                                                     or runtime.attention != "flash_attention_2"):
            raise ValueError("Attention caching requires single-layer FA2 checkpointing")
        self.model = Decoder(config, runtime, device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False,
                                 dtype=runtime.base_dtype, device=device)
        self.offload = None
        self.last_execution = {}
        for name, param in self.named_parameters():
            component = "embed" if "embed_tokens" in name else "lm_head" if name.startswith("lm_head") else "norm" if "norm" in name else None
            if runtime.rank:
                trainable = "lora_" in name or (component == "embed" and runtime.train_embed) or (component == "lm_head" and runtime.train_head)
            else:
                trainable = component not in runtime.frozen
            param.requires_grad_(trainable)
            if trainable:
                param.data = param.data.to(runtime.train_dtype)

    def forward(self, batch):
        r = self.runtime
        if batch["input_ids"].is_cuda and r.compute_dtype == torch.bfloat16 and r.train_dtype == torch.float32:
            # Configure before forward: changing GEMM precision in backward
            # would make checkpoint replay differ from its original execution.
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if self.offload is not None:
            self.offload.begin_forward()
        segments = canonical_segments(batch["attention_mask"], batch.get("segment_ids"))
        positions = batch.get("position_ids")
        if positions is None:
            positions = position_ids(segments)
        if (positions < 0).any() or (positions >= self.config.max_position_embeddings).any():
            raise ValueError("position_ids exceed the model's logical position range")
        spans = batch.get("_spans") or runs(segments)
        from ..training.execution import execution_policy
        available = None
        if batch["input_ids"].is_cuda and r.adaptive:
            device = batch["input_ids"].device
            allocated = torch.cuda.memory_allocated(device)
            free, _ = torch.cuda.mem_get_info(device)
            available = free+torch.cuda.memory_reserved(device)-allocated
            if r.gpu_budget_bytes is not None:
                available = min(available, max(0, r.gpu_budget_bytes-allocated))
        policy = execution_policy(self.config, r, batch["input_ids"].numel(),
                                  max((end-start for _, start, end in spans), default=0),
                                  cuda=batch["input_ids"].is_cuda, offload=self.offload is not None,
                                  quantized=self.model.layers[0].mlp.gate_proj.quantized is not None,
                                  grouped_attention="attention_end" in batch, available_bytes=available)
        checkpoint_enabled = policy.checkpoint and self.training and torch.is_grad_enabled()
        group_size, inner_checkpoint = r.group_size, r.inner_checkpoint
        cosine, sine = rotary_values(positions, self.config, r.compute_dtype)
        x = self.model.embed_tokens(batch["input_ids"]).to(r.compute_dtype)
        remaining = policy.mlp_cache_bytes if x.is_cuda and torch.is_grad_enabled() else 0
        mlp_cached = [False]*len(self.model.layers)
        retained_mlp_bytes = 0
        for index in reversed(range(len(self.model.layers))):
            layer = self.model.layers[index]
            cost = layer.mlp.cache_bytes(x.numel()//x.shape[-1]) if remaining else 0
            layer.mlp.cache_intermediates = bool(cost and cost <= remaining)
            mlp_cached[index] = layer.mlp.cache_intermediates
            if layer.mlp.cache_intermediates:
                remaining -= cost
                retained_mlp_bytes += cost
        mlp_cached = tuple(mlp_cached)
        end = batch.get("attention_end")
        layout = prepare_dense_layout(x.shape[:2], spans, x.device, end) if r.attention == "flash_attention_2" else None
        attention_layers = 0
        attention_cost = 0
        if checkpoint_enabled and policy.attention_cache_bytes and group_size == 1 and r.attention == "flash_attention_2":
            operations = len(layout.groups) if layout.mode == "groups" else 1
            attention_cost = x.numel()*x.element_size() + x.numel()//x.shape[-1]*self.config.num_attention_heads*4 + operations*16
            attention_layers = min(len(self.model.layers), policy.attention_cache_bytes//attention_cost)
        self.last_execution = dict(rows=x.numel()//x.shape[-1], checkpoint=checkpoint_enabled,
                                   calibrated=policy.calibrated, mlp_tile=policy.mlp_tile,
                                   mlp_cache_layers=sum(mlp_cached), mlp_cache_bytes=retained_mlp_bytes,
                                   attention_cache_layers=attention_layers,
                                   attention_cache_bytes=attention_layers*attention_cost)
        def apply_layer(index, value):
            fn = lambda z: self.model.layers[index](z, cosine, sine, segments, spans, end, layout,
                                                   mlp_tile=policy.mlp_tile, mlp_cache=mlp_cached[index])
            if checkpoint_enabled and (group_size == 1 or inner_checkpoint):
                # Nested inner checkpoints must remain under the outer
                # checkpoint hooks; replacing those hooks would retain every
                # interior boundary and defeat grouping's memory reduction.
                options = {}
                if index >= len(self.model.layers)-attention_layers:
                    from ..kernels.attention_cache import attention_cache_contexts
                    options["context_fn"] = attention_cache_contexts
                with self.offload.hooks() if self.offload is not None and group_size == 1 else no_offload():
                    # The decoder has no stochastic forward operators.
                    return checkpoint(fn, value, use_reentrant=False, preserve_rng_state=False, **options)
            return fn(value)
        for start in range(0, len(self.model.layers), group_size):
            stop = min(start+group_size, len(self.model.layers))
            def group(value, start=start, stop=stop):
                for index in range(start, stop):
                    value = apply_layer(index, value)
                return value
            if checkpoint_enabled and group_size > 1:
                with self.offload.hooks() if self.offload is not None else no_offload():
                    x = checkpoint(group, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = group(x)
        return self.model.norm(x)
