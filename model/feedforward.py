"""
═══════════════════════════════════════════════════════
 SWIGLU FEED-FORWARD NETWORK — Modern FFN variant
═══════════════════════════════════════════════════════

 SwiGLU (Shazeer, 2020) replaces the standard 2-layer FFN:
   Standard: GELU(xW₁)W₂
   SwiGLU:   (SiLU(xW_gate) ⊙ xW_up) W_down

 Benefits: better gradient flow, improved training stability.
═══════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Three linear layers: gate, up, down with SiLU gating.
    Inner dimension is 2/3 of ff_dim to keep param count similar.
    """

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        # Use 2/3 inner dim for SwiGLU (compensates for 3 matrices vs 2)
        inner_dim = int(ff_dim * 2 / 3)

        self.w_gate = nn.Linear(embed_dim, inner_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: (SiLU(xW_gate) ⊙ xW_up) W_down"""
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        x = self.w_down(x)
        return self.dropout(x)


# Backward compatibility
class FeedForward(SwiGLUFeedForward):
    """Alias for SwiGLUFeedForward (backward compat)."""
    pass
