"""
utils.py — Inspection helpers for attention weights and context vectors.

The key diagnostic after training is the attention weight matrix:
a (seq_len × seq_len) grid where entry [i, j] tells you how much
position i attended to position j when building its output vector.

High weight at [i, j] means:
  "to represent word i in this context, I borrowed heavily from word j"
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

from src.model import AttentionModel


def get_attention_weights(
    model: AttentionModel,
    sentence: List[int],
) -> torch.Tensor:
    """
    Returns attention weights for a single encoded sentence.

    Args:
        model:    trained AttentionModel
        sentence: list of word indices (output of Tokenizer.encode())

    Returns:
        weights: (n_heads, seq_len, seq_len)
    """
    model.eval()
    x = torch.tensor(sentence).unsqueeze(0)   # (1, seq_len)
    with torch.no_grad():
        _, weights = model(x)
    return weights.squeeze(0)   # (n_heads, seq_len, seq_len)


def print_attention_table(
    weights: torch.Tensor,
    words: List[str],
    head: int = 0,
) -> None:
    """
    Prints a human-readable attention table for one head.

    Rows = query positions (what each word is asking about)
    Cols = key positions   (what each word offers as context)

    Args:
        weights: (n_heads, seq_len, seq_len)
        words:   list of word strings for this sentence
        head:    which head to display (default 0)
    """
    w = weights[head].numpy()
    seq_len = len(words)
    col_w = max(len(wd) for wd in words) + 2

    print(f"\n  Attention weights — head {head}")
    print(f"  {'':>{col_w}}", end="")
    for word in words:
        print(f"  {word:>{col_w}}", end="")
    print()

    for i, row_word in enumerate(words):
        print(f"  {row_word:>{col_w}}", end="")
        for j in range(seq_len):
            val = w[i, j]
            bar = "█" * int(val * 8)
            print(f"  {val:.3f}{bar:>3}", end="")
        print()
    print()


def compare_vectors(
    model: AttentionModel,
    sentence: List[int],
    words: List[str],
) -> None:
    """
    Shows how attention changes each word's vector vs. the raw embedding.

    Prints cosine similarity between:
      - the raw embedding of word i
      - the context-aware vector of word i after attention

    A low similarity means attention significantly changed the representation
    (the word was heavily influenced by its neighbours).
    A high similarity means the word was mostly self-referential.

    Args:
        model:    trained AttentionModel
        sentence: list of word indices
        words:    list of word strings
    """
    model.eval()
    x = torch.tensor(sentence).unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        raw_emb = model.embedding(x).squeeze(0)   # (seq_len, emb_dim)
        ctx, _  = model.get_context_vectors(x)
        ctx     = ctx.squeeze(0)                  # (seq_len, emb_dim)

    print("\n  Raw embedding  vs  context-aware vector (cosine similarity)")
    print("  Low score = attention changed this word's meaning significantly\n")
    for i, word in enumerate(words):
        sim = F.cosine_similarity(
            raw_emb[i].unsqueeze(0),
            ctx[i].unsqueeze(0),
        ).item()
        bar = "█" * int((1 - abs(sim)) * 20)
        print(f"    {word:<12}  sim={sim:+.4f}  change: {bar}")
    print()
