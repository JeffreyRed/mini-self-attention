"""
embedding.py — Embedding layer initialised from mini-embedding weights.

In mini-embedding we trained a (vocab_size × emb_dim) matrix E from scratch.
Here we reuse that same matrix as the *input* to the attention mechanism.

Two modes:
  1. load_pretrained()  — loads the .pt file saved by mini-embedding's main.py
  2. random_init()      — fresh random embeddings (useful if no .pt file exists)

This separation makes the connection explicit: attention is a layer that sits
*on top of* embeddings, not a replacement for them.
"""

import torch
import torch.nn as nn
from pathlib import Path


class EmbeddingLayer(nn.Module):
    """
    Standard token embedding lookup, optionally warm-started from
    mini-embedding's trained weights.

    Args:
        vocab_size (int):   number of unique tokens
        emb_dim (int):      embedding dimensionality
        pretrained_path (str | None): path to a .pt tensor saved by
                            mini-embedding (outputs/embeddings.pt).
                            If None or file not found, uses random init.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        pretrained_path: str = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        if pretrained_path and Path(pretrained_path).exists():
            weights = torch.load(pretrained_path, weights_only=True)
            if weights.shape == (vocab_size, emb_dim):
                self.embedding.weight = nn.Parameter(weights.clone())
                self._source = f"pretrained  ({pretrained_path})"
            else:
                self._source = (
                    f"random  (shape mismatch: "
                    f"file={tuple(weights.shape)} vs "
                    f"expected=({vocab_size},{emb_dim}))"
                )
        else:
            self._source = "random  (no pretrained file)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor (batch, seq_len) of word indices
        Returns:
            FloatTensor (batch, seq_len, emb_dim)
        """
        return self.embedding(x)

    def __repr__(self) -> str:
        v, d = self.embedding.weight.shape
        return f"EmbeddingLayer(vocab={v}, dim={d}, source={self._source})"
