"""
model.py — Self-attention encoder block.

Stacks the components in the order they appear in a real transformer encoder:

    token indices
         │
         ▼
    EmbeddingLayer              (vocab_size → emb_dim)
         │
         ▼
    MultiHeadAttention          (emb_dim → emb_dim, context-aware)
         │  + residual connection
         ▼
    LayerNorm                   (stabilises training)
         │
         ▼
    FeedForward  (Linear → ReLU → Linear)
         │  + residual connection
         ▼
    LayerNorm
         │
         ▼
    Linear → vocab logits       (for next-token training objective)

The residual connections ("x = x + sublayer(x)") are the other key idea from
the transformer paper — they let gradients flow directly to early layers and
allow the network to learn *incremental* refinements rather than full
re-representations at each step.

This is a single encoder block.  The mini-transformer step will stack N of
these and add positional encoding.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.embedding import EmbeddingLayer
from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside every transformer block.

    Two linear layers with a ReLU in between.  The inner dimension is
    conventionally 4× the embedding dimension (following the original paper).

    Args:
        emb_dim (int):   input/output dimension
        ff_dim (int):    inner (hidden) dimension  (default: 4 × emb_dim)
    """

    def __init__(self, emb_dim: int, ff_dim: int = None) -> None:
        super().__init__()
        ff_dim = ff_dim or 4 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """
    One complete transformer encoder block:
        MultiHeadAttention + residual + LayerNorm
        FeedForward        + residual + LayerNorm

    Args:
        emb_dim (int):  embedding dimension
        n_heads (int):  number of attention heads
        ff_dim (int):   feedforward inner dimension (default 4 × emb_dim)
        dropout (float): dropout probability (default 0.0 for this small demo)
    """

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        ff_dim: int = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, n_heads)
        self.ff        = FeedForward(emb_dim, ff_dim)
        self.norm1     = nn.LayerNorm(emb_dim)
        self.norm2     = nn.LayerNorm(emb_dim)
        self.drop      = nn.Dropout(dropout)

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
            out:     (batch, seq_len, emb_dim) — context-aware representations
            weights: (batch, n_heads, seq_len, seq_len) — attention maps
        """
        # ── Sub-layer 1: multi-head self-attention + residual + norm ─────────
        attn_out, weights = self.attention(x, mask)
        x = self.norm1(x + self.drop(attn_out))    # residual connection

        # ── Sub-layer 2: feedforward + residual + norm ────────────────────────
        x = self.norm2(x + self.drop(self.ff(x)))  # residual connection

        return x, weights


class AttentionModel(nn.Module):
    """
    Full model: Embedding → SelfAttentionBlock → Linear → logits.

    Trained on next-token prediction: given a sequence of words, predict
    the next word at each position.  This is the same objective as GPT,
    just at toy scale.

    Args:
        vocab_size (int):        vocabulary size
        emb_dim (int):           embedding dimensionality
        n_heads (int):           number of attention heads
        ff_dim (int):            feedforward inner size (default 4 × emb_dim)
        pretrained_emb (str):    optional path to mini-embedding .pt weights
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        n_heads: int,
        ff_dim: int = None,
        pretrained_emb: str = None,
    ) -> None:
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, emb_dim, pretrained_emb)
        self.block     = SelfAttentionBlock(emb_dim, n_heads, ff_dim)
        self.head      = nn.Linear(emb_dim, vocab_size)  # output projection

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (batch, seq_len)  — word indices
            mask: optional causal or padding mask

        Returns:
            logits:  (batch, seq_len, vocab_size)
            weights: (batch, n_heads, seq_len, seq_len)
        """
        emb            = self.embedding(x)          # (batch, seq_len, emb_dim)
        ctx, weights   = self.block(emb)            # context-aware vectors
        logits         = self.head(ctx)             # (batch, seq_len, vocab_size)
        return logits, weights

    def get_context_vectors(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the context-aware vectors (after attention) and attention
        weights, without the final logit projection.
        Useful for visualising what attention does to the embedding vectors.

        Args:
            x: (batch, seq_len) — word indices

        Returns:
            ctx_vectors: (batch, seq_len, emb_dim)
            weights:     (batch, n_heads, seq_len, seq_len)
        """
        with torch.no_grad():
            emb          = self.embedding(x)
            ctx, weights = self.block(emb)
        return ctx, weights

    def __repr__(self) -> str:
        emb   = self.embedding.embedding
        block = self.block
        v, d  = emb.weight.shape
        return (
            f"AttentionModel(\n"
            f"  vocab_size={v}, emb_dim={d},\n"
            f"  {block.attention},\n"
            f"  ff_dim={block.ff.net[0].out_features},\n"
            f"  embedding={self.embedding._source}\n"
            f")"
        )
