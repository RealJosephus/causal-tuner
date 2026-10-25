"""Decoder architecture configuration and small-model fixtures."""

from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 4096
    intermediate_size: int = 13568
    vocab_size: int = 185600
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    head_dim: int = 128
    partial_rotary_factor: float = 0.5
    rope_theta: float = 1e9
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 2097152
    pad_token_id: int = 151329
    eos_token_id: int = 151650
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    rope_style: str = "gpt-j"
    rope_scaling: dict | None = None
    sliding_window: int | None = None
    use_sliding_window: bool = False
    initializer_range: float = 0.02

    def __post_init__(self):
        for key in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
                    "num_attention_heads", "head_dim", "max_position_embeddings"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.num_key_value_heads != 1 or self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("Only MQA with hidden_size = heads * head_dim is supported")
        if self.rotary_dim <= 0 or self.rotary_dim % 2 or self.rotary_dim > self.head_dim:
            raise ValueError("The partial rotary dimension must be positive and even")
        if (self.rope_style != "gpt-j" or self.rope_scaling is not None or self.sliding_window is not None
                or self.use_sliding_window or self.tie_word_embeddings or not self.attention_bias
                or self.attention_dropout != 0 or self.hidden_act != "silu"):
            raise ValueError("Unsupported architecture settings")
        for key in ("rope_theta", "rms_norm_eps", "initializer_range"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for value in (self.pad_token_id, *([self.eos_token_id] if isinstance(self.eos_token_id, int)
                                          else self.eos_token_id)):
            if not 0 <= value < self.vocab_size:
                raise ValueError("Special token IDs must index the vocabulary")

    @property
    def rotary_dim(self):
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        rope = raw.get("rope_scaling") or raw.get("rope_parameters", raw.get("rope_scaling"))
        if rope is not None:
            if not isinstance(rope, dict):
                raise ValueError("RoPE parameters must be a mapping")
            if rope.get("rope_type", rope.get("type", "default")) != "default":
                raise ValueError("Only unscaled RoPE is supported")
            for name in ("rope_theta", "partial_rotary_factor"):
                if name in rope:
                    raw[name] = rope[name]
            raw["rope_scaling"] = None
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})

    @classmethod
    def tiny(cls):
        return cls(hidden_size=32, intermediate_size=68, vocab_size=97, num_hidden_layers=2,
                   num_attention_heads=4, head_dim=8, pad_token_id=0, eos_token_id=1)

    def to_dict(self):
        return asdict(self)

    def parameter_counts(self, rank=256):
        h, i, d, v = self.hidden_size, self.intermediate_size, self.head_dim, self.vocab_size
        decoder = 2*h*h + 2*d*h + 3*i*h + h + 2*d + 2*h
        base = self.num_hidden_layers * decoder + 2*v*h + h
        adapters = self.num_hidden_layers * rank * (9*h + 3*i + 2*d)
        return base, adapters
