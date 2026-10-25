"""Partial GPT-J RoPE evaluated only at the requested logical positions."""

import torch


def rotary_values(positions, config, dtype):
    channel = torch.arange(0, config.rotary_dim, 2, device=positions.device, dtype=torch.float32)
    inverse = 1.0 / (config.rope_theta ** (channel / config.rotary_dim))
    if positions.is_cuda:
        from ..kernels.core_cuda import position_values
        return position_values(positions, inverse, dtype)
    angle = positions.float()[..., None] * inverse
    return angle.cos().to(dtype), angle.sin().to(dtype)


def rotate(x, cosine, sine):
    if x.is_cuda:
        from ..kernels.core_cuda import PartialRoPE
        return PartialRoPE.apply(x, cosine.contiguous(), sine.contiguous())
    size = cosine.shape[-1] * 2
    cosine, sine = cosine.unsqueeze(-2), sine.unsqueeze(-2)
    even, odd = x[..., :size:2], x[..., 1:size:2]
    paired = torch.stack((even*cosine - odd*sine, odd*cosine + even*sine), dim=-1)
    return torch.cat((paired.flatten(-2), x[..., size:]), dim=-1)
