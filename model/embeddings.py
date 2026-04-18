"""
═══════════════════════════════════════════════════════
 TOKEN EMBEDDING — Maps token IDs to dense vectors
═══════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Standard token embedding layer.

    Maps integer token IDs to dense vectors of size `embed_dim`.
    Applies sqrt(embed_dim) scaling (as in "Attention Is All You Need").
    """

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        """
        Args:
            vocab_size:  Total number of tokens in vocabulary.
            embed_dim:   Dimensionality of embeddings.
            padding_idx: Index of the <PAD> token (zeroed out).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        self._scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs tensor of shape [batch_size, seq_len].

        Returns:
            Embeddings tensor of shape [batch_size, seq_len, embed_dim].
        """
        return self.embedding(x) * self._scale
