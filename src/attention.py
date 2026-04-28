"""
attention.py — Scaled dot-product and multi-head self-attention.

This is the central module of the project.  Everything else is scaffolding.

─────────────────────────────────────────────────────────────────────────────
INTUITION
─────────────────────────────────────────────────────────────────────────────
After mini-embedding, every word has a fixed vector.
"bank" always maps to [0.3, -0.1, ...] regardless of context.

Self-attention fixes this: each word is allowed to look at every other word
in the sentence and blend their information into its own representation.

  "I deposited money at the bank"
                              ↑
       "bank" attends to "deposited" and "money" → financial sense

  "We sat by the river bank"
                        ↑
       "bank" attends to "river" and "sat" → geographical sense

Same word, different context → different output vector.

─────────────────────────────────────────────────────────────────────────────
THE MATH — scaled dot-product attention
─────────────────────────────────────────────────────────────────────────────
Inputs: sequence of vectors  X  shape (seq_len, emb_dim)

Three learnable projections:
    Q = X @ W_Q      queries  — "what am I looking for?"
    K = X @ W_K      keys     — "what do I contain?"
    V = X @ W_V      values   — "what do I send if selected?"

Attention scores:
    scores = Q @ K^T / sqrt(d_k)       (seq_len, seq_len)
    weights = softmax(scores, dim=-1)  (each row sums to 1)

Output:
    out = weights @ V                  (seq_len, d_v)

The division by sqrt(d_k) prevents the dot products from growing so large
that softmax saturates and gradients vanish.

─────────────────────────────────────────────────────────────────────────────
MULTI-HEAD ATTENTION
─────────────────────────────────────────────────────────────────────────────
Run h independent attention heads in parallel, each with its own W_Q/W_K/W_V.
Concatenate their outputs, project back to emb_dim.

Why? Each head can specialise in a different kind of relationship:
  head 1 → syntactic dependencies
  head 2 → semantic similarity
  head 3 → positional proximity
  ...

This is the direct building block used in every transformer encoder layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Scaled dot-product attention  (single head)
# ─────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention.

    Args:
        Q: queries  (batch, heads, seq_len, d_k)
        K: keys     (batch, heads, seq_len, d_k)
        V: values   (batch, heads, seq_len, d_v)
        mask: optional boolean mask (batch, 1, seq_len, seq_len)
              True  → position is masked out (e.g. padding or future tokens)

    Returns:
        output:  (batch, heads, seq_len, d_v)
        weights: (batch, heads, seq_len, seq_len)  — for visualisation
    """
    d_k = Q.size(-1)

    # Raw attention scores: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask before softmax (set masked positions to -inf → prob ≈ 0)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Normalise: each query position gets a probability distribution over keys
    weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(weights, V)

    return output, weights


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Single attention head  (explicit, for educational clarity)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionHead(nn.Module):
    """
    One attention head: projects X into Q, K, V then runs scaled dot-product.

    Args:
        emb_dim (int): input embedding dimensionality
        head_dim (int): dimensionality of Q, K, V inside this head (= emb_dim // n_heads)
    """

    def __init__(self, emb_dim: int, head_dim: int) -> None:
        super().__init__()
        self.W_Q = nn.Linear(emb_dim, head_dim, bias=False)
        self.W_K = nn.Linear(emb_dim, head_dim, bias=False)
        self.W_V = nn.Linear(emb_dim, head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (batch, seq_len, emb_dim)
            mask: (batch, 1, seq_len, seq_len) optional
        Returns:
            output:  (batch, seq_len, head_dim)
            weights: (batch, seq_len, seq_len)
        """
        Q = self.W_Q(x)   # (batch, seq_len, head_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        # Add a dummy heads dimension so we can reuse scaled_dot_product_attention
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        out, weights = scaled_dot_product_attention(Q, K, V, mask)

        return out.squeeze(1), weights.squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-head attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention as described in Vaswani et al. (2017).

    Runs `n_heads` attention heads in parallel, concatenates their outputs,
    and projects back to `emb_dim` with a final linear layer W_O.

    Args:
        emb_dim (int):  model embedding dimension  (must be divisible by n_heads)
        n_heads (int):  number of parallel attention heads

    Architecture:
        head_dim = emb_dim // n_heads

        For each head h:
            Q_h = X @ W_Q_h       (batch, seq_len, head_dim)
            K_h = X @ W_K_h
            V_h = X @ W_V_h
            head_h = softmax(Q_h @ K_h^T / sqrt(head_dim)) @ V_h

        concat = [head_1 | head_2 | ... | head_h]   (batch, seq_len, emb_dim)
        output = concat @ W_O                        (batch, seq_len, emb_dim)
    """

    def __init__(self, emb_dim: int, n_heads: int) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0, (
            f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads

        # Single fused projection for all heads — more efficient than a loop
        self.W_Q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (batch, seq_len, emb_dim)
            mask: (batch, 1, seq_len, seq_len) optional

        Returns:
            output:  (batch, seq_len, emb_dim)
            weights: (batch, n_heads, seq_len, seq_len)  — one matrix per head
        """
        batch, seq_len, _ = x.shape

        # Project and split into heads
        # (batch, seq_len, emb_dim) → (batch, n_heads, seq_len, head_dim)
        Q = self._split_heads(self.W_Q(x), batch, seq_len)
        K = self._split_heads(self.W_K(x), batch, seq_len)
        V = self._split_heads(self.W_V(x), batch, seq_len)

        # Scaled dot-product attention across all heads at once
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (batch, n_heads, seq_len, head_dim)
        # weights:  (batch, n_heads, seq_len, seq_len)

        # Merge heads back: (batch, seq_len, emb_dim)
        merged = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_dim)

        # Final output projection
        output = self.W_O(merged)   # (batch, seq_len, emb_dim)

        return output, weights

    def _split_heads(
        self, x: torch.Tensor, batch: int, seq_len: int
    ) -> torch.Tensor:
        """Reshapes (batch, seq_len, emb_dim) → (batch, n_heads, seq_len, head_dim)."""
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention("
            f"emb_dim={self.emb_dim}, "
            f"n_heads={self.n_heads}, "
            f"head_dim={self.head_dim})"
        )
