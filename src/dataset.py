"""
dataset.py — Sentence-level dataset for next-token prediction.

Difference from mini-embedding:
  mini-embedding  → chopped corpus into (target, context) word pairs
  mini-attention  → keeps full sentences intact as sequences

For each encoded sentence [w0, w1, w2, w3]:
    input:  [w0, w1, w2]       (all words except the last)
    target: [w1, w2, w3]       (all words except the first, shifted by 1)

This is called "teacher-forced next-token prediction".  At every position i
the model sees words 0..i and must predict word i+1.  This forces the model
to build useful context-aware representations — which is exactly what
attention learns to produce.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class SequenceDataset(Dataset):
    """
    Wraps an encoded corpus as (input_sequence, target_sequence) pairs
    for next-token prediction.

    Args:
        encoded_sentences (List[List[int]]): output of Tokenizer.encode_all()
        min_len (int): discard sentences shorter than this (default 2)
    """

    def __init__(
        self,
        encoded_sentences: List[List[int]],
        min_len: int = 2,
    ) -> None:
        self.pairs: List[Tuple[List[int], List[int]]] = []

        for sentence in encoded_sentences:
            if len(sentence) < min_len + 1:
                continue
            self.pairs.append((sentence[:-1], sentence[1:]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

    def __repr__(self) -> str:
        return f"SequenceDataset(sentences={len(self.pairs)})"


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pads variable-length sequences in a batch to the same length.

    PyTorch DataLoader calls this automatically when batch_size > 1.
    Shorter sequences are padded with index 0 on the right.

    Returns:
        src: (batch, max_seq_len)
        tgt: (batch, max_seq_len)
        pad_mask: (batch, 1, max_seq_len, max_seq_len)  — True where padded
    """
    srcs, tgts = zip(*batch)
    max_len = max(s.size(0) for s in srcs)

    padded_src = torch.zeros(len(srcs), max_len, dtype=torch.long)
    padded_tgt = torch.zeros(len(tgts), max_len, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        padded_src[i, :s.size(0)] = s
        padded_tgt[i, :t.size(0)] = t

    # Padding mask: True at padded positions
    pad_mask = (padded_src == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
    pad_mask = pad_mask.expand(-1, 1, max_len, max_len)

    return padded_src, padded_tgt, pad_mask
