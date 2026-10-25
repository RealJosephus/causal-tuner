"""Independent projections with unit-scale LoRA and optional frozen NF4."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, inputs, outputs, *, bias=False, rank=0, base_dtype=torch.bfloat16,
                 train_dtype=torch.float32, compute_dtype=torch.bfloat16, device=None):
        super().__init__()
        self.compute_dtype, self.rank = compute_dtype, rank
        self.weight = nn.Parameter(torch.empty(outputs, inputs, dtype=base_dtype, device=device),
                                   requires_grad=not rank)
        self.bias = nn.Parameter(torch.zeros(outputs, dtype=base_dtype, device=device),
                                 requires_grad=not rank) if bias else None
        self.quantized = None
        if rank:
            self.lora_A = nn.Parameter(torch.empty(rank, inputs, dtype=train_dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(outputs, rank, dtype=train_dtype, device=device))
            nn.init.uniform_(self.lora_A, -math.sqrt(6/inputs), math.sqrt(6/inputs))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        from ..kernels.linear import can_fuse, project_group
        if can_fuse(self, x):
            return project_group((self,), x)[0]
        x = x.to(self.compute_dtype)
        if self.quantized is not None:
            output = self.quantized(x)
            if self.bias is not None:
                output = output + self.bias.to(x.dtype)
        else:
            output = F.linear(x, self.weight.to(x.dtype),
                              None if self.bias is None else self.bias.to(x.dtype))
        if self.rank:
            output = output + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return output

    def install_nf4(self, weight, device):
        if not self.rank or weight.requires_grad:
            raise ValueError("NF4 is restricted to frozen adapter base projections")
        if torch.device(device).type != "cuda":
            raise ValueError("NF4 requires a CUDA device")
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes to use NF4") from exc
        layer = bnb.nn.Linear4bit(weight.shape[1], weight.shape[0], bias=False,
                                  compute_dtype=self.compute_dtype, compress_statistics=True,
                                  quant_type="nf4", device="meta")
        layer.weight = bnb.nn.Params4bit(weight.detach().cpu(), requires_grad=False,
                                        compress_statistics=True, quant_type="nf4")
        self.quantized = layer.to(device)
        self.register_parameter("weight", None)

    def base_weight(self):
        if self.quantized is None:
            return self.weight
        import bitsandbytes.functional as bnbf
        return bnbf.dequantize_4bit(self.quantized.weight.data, self.quantized.weight.quant_state)
