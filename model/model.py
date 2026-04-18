"""
═══════════════════════════════════════════════════════
 MINI TRANSFORMER v2 — Upgraded model architecture
═══════════════════════════════════════════════════════

 Improvements over v1:
   - RoPE (Rotary Position Embedding) instead of sinusoidal
   - SwiGLU FFN instead of GELU FFN
   - Residual scaling (1/sqrt(num_layers))
   - Larger default config (256 dim, 8 heads, 5 layers)
   - KV-cache support for fast inference
   - Kaiming init for FFN, Xavier for attention
═══════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .embeddings import TokenEmbedding
from .positional_encoding import RotaryPositionalEncoding, PositionalEncoding
from .attention import MultiHeadAttention, create_causal_mask, create_padding_mask
from .feedforward import FeedForward


@dataclass
class TransformerConfig:
    """Configuration for MiniTransformer v2."""
    vocab_size: int = 5000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 5
    ff_dim: int = 768
    max_seq_len: int = 256
    dropout: float = 0.2
    padding_idx: int = 0


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with RoPE + SwiGLU + residual scaling.
    Pre-LN architecture.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(
            embed_dim=config.embed_dim,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )
        # Residual scaling: 1/sqrt(2*num_layers) per DeepNorm
        self._res_scale = 1.0 / math.sqrt(2 * config.num_layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention with RoPE
        normed = self.ln1(x)
        attn_out, new_kv = self.attention(
            normed, normed, normed,
            mask=mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            kv_cache=kv_cache,
        )
        x = x + attn_out * self._res_scale

        # Feed-forward
        normed = self.ln2(x)
        ff_out = self.ffn(normed)
        x = x + ff_out * self._res_scale

        return x, new_kv


class MiniTransformer(nn.Module):
    """
    GPT-like decoder-only transformer v2.

    Uses RoPE for positional encoding, SwiGLU FFN,
    residual scaling, and supports KV-cache inference.
    """

    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()
        self.config = config or TransformerConfig()
        c = self.config

        self.token_embedding = TokenEmbedding(
            vocab_size=c.vocab_size,
            embed_dim=c.embed_dim,
            padding_idx=c.padding_idx,
        )
        # RoPE applied in attention, not here
        self.rope = RotaryPositionalEncoding(
            dim=c.embed_dim // c.num_heads,
            max_len=c.max_seq_len,
        )
        self.embed_dropout = nn.Dropout(c.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(c, layer_idx=i) for i in range(c.num_layers)
        ])

        self.ln_final = nn.LayerNorm(c.embed_dim)
        self.output_head = nn.Linear(c.embed_dim, c.vocab_size, bias=False)

        self._init_weights()
        # Weight tying
        self.output_head.weight = self.token_embedding.embedding.weight

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'w_gate' in name or 'w_up' in name or 'w_down' in name:
                    # Kaiming for FFN
                    nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        device = input_ids.device
        causal_mask = create_causal_mask(seq_len, device=device)
        padding_mask = create_padding_mask(input_ids, pad_id=self.config.padding_idx)
        return causal_mask * padding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_kv_cache: bool = False,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, List]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            mask: Optional attention mask
            use_kv_cache: Enable KV-cache (for inference)
            kv_caches: List of (K, V) tuples per layer

        Returns:
            logits [batch_size, seq_len, vocab_size]
            (or tuple with new kv_caches if use_kv_cache=True)
        """
        seq_len = input_ids.size(1)

        if mask is None:
            mask = self.build_attention_mask(input_ids)

        # Embedding (no positional encoding — RoPE is in attention)
        x = self.token_embedding(input_ids)
        x = self.embed_dropout(x)

        # RoPE cos/sin
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            x, new_kv = block(x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin, kv_cache=kv)
            if use_kv_cache:
                new_kv_caches.append(new_kv)

        x = self.ln_final(x)
        logits = self.output_head(x)

        if use_kv_cache:
            return logits, new_kv_caches
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> dict:
        return {
            "vocab_size": self.config.vocab_size,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "num_layers": self.config.num_layers,
            "ff_dim": self.config.ff_dim,
            "max_seq_len": self.config.max_seq_len,
            "total_parameters": self.count_parameters(),
            "total_parameters_human": f"{self.count_parameters() / 1e6:.2f}M",
            "architecture": "GPT-v2 (RoPE + SwiGLU + ResScale)",
        }

    @classmethod
    def from_config(cls, **kwargs) -> "MiniTransformer":
        config = TransformerConfig(**kwargs)
        return cls(config)
