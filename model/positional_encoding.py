"""
═══════════════════════════════════════════════════════
 ROTARY POSITIONAL ENCODING (RoPE)
═══════════════════════════════════════════════════════

 Rotary Position Embedding from Su et al., 2021.
 Applied directly to Q/K in attention, not added to embeddings.
 Better extrapolation and relative position awareness than sinusoidal.

 Also keeps legacy sinusoidal PositionalEncoding for backward compat.
═══════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import math


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes cos/sin tables, applied to Q and K in attention.
    """

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute for max_len
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)  # [1, seq_len, dim]
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> tuple:
        """
        Returns (cos, sin) for the given sequence length.
        Shape: [1, seq_len, dim]
        """
        seq_len = seq_len or x.size(1)
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :],
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply RoPE to Q and K tensors.

    Args:
        q, k: [batch, heads, seq_len, head_dim]
        cos, sin: [1, seq_len, head_dim]
    """
    cos = cos.unsqueeze(1)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ── Legacy sinusoidal (kept for backward compat) ───────

class PositionalEncoding(nn.Module):
    """Legacy sinusoidal positional encoding (added to embeddings)."""

    def __init__(self, embed_dim: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
