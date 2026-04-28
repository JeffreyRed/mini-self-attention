"""
train.py — Training loop for the self-attention model.

Same five-step PyTorch loop as mini-embedding
(zero_grad → forward → loss → backward → step),
but now operating on full sequences instead of word pairs.

Loss: CrossEntropyLoss over every position in every sentence.
At each position i, the model sees context [w_0 .. w_i] and must predict w_{i+1}.
The loss is averaged across all positions and all sentences in the batch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import AttentionModel
from src.dataset import SequenceDataset, collate_fn


def train(
    model: AttentionModel,
    dataset: SequenceDataset,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 4,
    verbose: bool = True,
    snapshots: list = None,
    snapshot_every: int = 5,
) -> list:
    """
    Trains the AttentionModel on next-token prediction.

    Args:
        model:          AttentionModel instance
        dataset:        SequenceDataset of (input, target) pairs
        epochs:         number of full passes over the data
        lr:             Adam learning rate
        batch_size:     sequences per gradient step
        verbose:        print loss every 10 epochs
        snapshots:      if provided, (epoch, attention_weights) tuples appended
                        every snapshot_every epochs — used for visualisation
        snapshot_every: snapshot frequency in epochs

    Returns:
        List of per-epoch mean losses.
    """
    loader   = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)
    loss_fn  = nn.CrossEntropyLoss(ignore_index=0)   # ignore padding token
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt, mask in loader:
            optimizer.zero_grad()

            logits, _ = model(src, mask)
            # logits: (batch, seq_len, vocab_size)
            # tgt:    (batch, seq_len)
            # CrossEntropyLoss expects (N, C, ...) format
            loss = loss_fn(logits.transpose(1, 2), tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(loader)
        loss_history.append(mean_loss)

        # Save attention weights snapshot for visualisation
        if snapshots is not None and (epoch % snapshot_every == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                # Use the first batch for a consistent snapshot
                src0, _, mask0 = next(iter(loader))
                _, w = model(src0[:1], mask0[:1])   # single sentence
                snapshots.append((epoch, w.detach().clone()))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch [{epoch:>3}/{epochs}]  Loss: {mean_loss:.4f}")

    return loss_history
