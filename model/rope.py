"""Rotary Position Embedding (RoPE) for table tennis rally Transformer attention layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000) -> None:
        super().__init__()
        if dim % 2 != 0:
            msg = f"RoPE requires an even dimension, got dim={dim}."
            raise ValueError(msg)
        if max_seq_len <= 0:
            msg = f"max_seq_len must be positive, got max_seq_len={max_seq_len}."
            raise ValueError(msg)

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device | str) -> tuple[Tensor, Tensor]:
        if seq_len <= 0:
            msg = f"seq_len must be positive, got seq_len={seq_len}."
            raise ValueError(msg)
        if seq_len > self.max_seq_len:
            msg = (
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len when constructing RotaryEmbedding."
            )
            raise ValueError(msg)

        t = torch.arange(seq_len, device=device).float()
        inv_freq = self.inv_freq.to(device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    if q.shape != k.shape:
        msg = f"q and k must have identical shapes, got q={q.shape}, k={k.shape}."
        raise ValueError(msg)
    if q.ndim != 4:
        msg = f"q and k must be 4D [batch, heads, seq_len, head_dim], got q.ndim={q.ndim}."
        raise ValueError(msg)

    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        msg = f"RoPE requires even head_dim, got head_dim={head_dim}."
        raise ValueError(msg)

    seq_len = q.shape[-2]
    if cos.shape != (seq_len, head_dim) or sin.shape != (seq_len, head_dim):
        msg = (
            "cos and sin must both have shape [seq_len, head_dim] matching q/k; "
            f"expected ({seq_len}, {head_dim}), got cos={cos.shape}, sin={sin.shape}."
        )
        raise ValueError(msg)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
