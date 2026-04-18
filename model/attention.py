"""
═══════════════════════════════════════════════════════
 MULTI-HEAD ATTENTION — With RoPE support + KV-cache
═══════════════════════════════════════════════════════

 Supports:
   - Standard scaled dot-product attention
   - Rotary Position Embeddings (RoPE)
   - KV-cache for fast autoregressive inference
═══════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .positional_encoding import apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE and optional KV-cache.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward with RoPE and optional KV-cache.

        Returns:
            (output, new_kv_cache) — kv_cache is None during training.
        """
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads: [B, S, D] → [B, H, S, D/H]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if rope_cos is not None and rope_sin is not None:
            Q, K = apply_rotary_pos_emb(Q, K, rope_cos, rope_sin)

        # KV-cache for inference
        new_kv_cache = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            K = torch.cat([cached_k, K], dim=2)
            V = torch.cat([cached_v, V], dim=2)
            new_kv_cache = (K, V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self._scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.W_o(context)

        return output, new_kv_cache


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_padding_mask(input_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    mask = (input_ids != pad_id).unsqueeze(1).unsqueeze(2)
    return mask.float()
