"""
visualize.py — Attention heatmaps, head comparisons, and loss curves.

The main diagnostic plot for attention is the attention heatmap:
a colour grid of shape (seq_len × seq_len) where brighter = more attention.
Reading row i tells you which words word i attended to most.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from typing import List

PALETTE = {
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "accent":    "#58a6ff",
    "highlight": "#f78166",
    "muted":     "#8b949e",
    "green":     "#3fb950",
}


# ── Attention heatmap ─────────────────────────────────────────────────────────

def plot_attention(
    weights: "torch.Tensor",
    words: List[str],
    sentence_label: str = "",
    save_path: str = None,
) -> None:
    """
    Plots one attention heatmap per head for a single sentence.

    Args:
        weights:        (n_heads, seq_len, seq_len) — from get_attention_weights()
        words:          list of word strings for this sentence
        sentence_label: printed as the figure title
        save_path:      if provided, saves the figure here
    """
    n_heads = weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    fig.patch.set_facecolor(PALETTE["bg"])

    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        w = weights[h].numpy()
        ax.set_facecolor(PALETTE["bg"])
        im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right",
                           fontsize=9, color=PALETTE["text"],
                           fontfamily="monospace")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9, color=PALETTE["text"],
                           fontfamily="monospace")

        # Annotate cells with numeric values
        for i in range(len(words)):
            for j in range(len(words)):
                val = w[i, j]
                color = "white" if val > 0.5 else PALETTE["muted"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

        ax.set_title(f"Head {h}", color=PALETTE["text"], fontsize=10, pad=8)
        ax.set_xlabel("Keys  (attended to)", color=PALETTE["muted"], fontsize=8)
        ax.set_ylabel("Queries  (each word)", color=PALETTE["muted"], fontsize=8)

    title = f"Self-Attention Weights   \"{sentence_label}\""
    fig.suptitle(title, color=PALETTE["text"], fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Attention heatmap saved → {save_path}")

    plt.show()


# ── All-sentences attention overview ─────────────────────────────────────────

def plot_all_sentences(
    model: "AttentionModel",
    tokenizer: "Tokenizer",
    save_path: str = None,
) -> None:
    """
    Plots the mean attention pattern (averaged across heads) for every
    sentence in the corpus, in one figure.

    This gives a quick overview of where the model pays attention globally.
    """
    import torch
    from src.utils import get_attention_weights

    sentences = tokenizer.sentences
    n = len(sentences)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9))
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = axes.flatten()

    for idx, sentence in enumerate(sentences):
        encoded = tokenizer.encode(sentence)
        if len(encoded) < 2:
            continue
        weights = get_attention_weights(model, encoded)           # (H, T, T)
        mean_w  = weights.mean(dim=0).numpy()                     # (T, T)

        ax = axes[idx]
        ax.set_facecolor(PALETTE["bg"])
        ax.imshow(mean_w, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(sentence)))
        ax.set_xticklabels(sentence, rotation=45, ha="right",
                           fontsize=8, color=PALETTE["text"],
                           fontfamily="monospace")
        ax.set_yticks(range(len(sentence)))
        ax.set_yticklabels(sentence, fontsize=8, color=PALETTE["text"],
                           fontfamily="monospace")
        ax.set_title(f"\"{' '.join(sentence)}\"",
                     color=PALETTE["text"], fontsize=8, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Mean attention (all heads) — all corpus sentences",
                 color=PALETTE["text"], fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"All-sentences plot saved → {save_path}")

    plt.show()


# ── Loss curve ────────────────────────────────────────────────────────────────

def plot_loss(loss_history: list, save_path: str = None) -> None:
    """Plots the training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"])

    ax.plot(loss_history, color=PALETTE["highlight"], linewidth=2, zorder=3)
    ax.set_title("Training Loss", color=PALETTE["text"],
                 fontsize=13, pad=14, loc="left")
    ax.set_xlabel("Epoch", color=PALETTE["muted"])
    ax.set_ylabel("Cross-Entropy Loss", color=PALETTE["muted"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Loss curve saved → {save_path}")
    plt.show()


# ── Animated attention evolution ──────────────────────────────────────────────

def animate_attention(
    snapshots: list,
    words: List[str],
    head: int = 0,
    save_path: str = None,
    interval: int = 300,
) -> None:
    """
    Animates how one head's attention pattern evolves during training.

    Args:
        snapshots:  list of (epoch, weights_tensor) from train()
                    weights_tensor shape: (1, n_heads, seq_len, seq_len)
        words:      word strings for the snapshot sentence
        head:       which head to animate
        save_path:  if provided, saves as .gif
        interval:   ms between frames
    """
    if not snapshots:
        print("No snapshots to animate.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    seq_len = len(words)
    im = ax.imshow(
        np.zeros((seq_len, seq_len)),
        cmap="Blues", vmin=0, vmax=1, aspect="auto",
    )
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(words, rotation=45, ha="right",
                       fontsize=9, color=PALETTE["text"], fontfamily="monospace")
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(words, fontsize=9, color=PALETTE["text"],
                       fontfamily="monospace")
    ax.set_xlabel("Keys", color=PALETTE["muted"], fontsize=8)
    ax.set_ylabel("Queries", color=PALETTE["muted"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    title = ax.set_title("", color=PALETTE["text"], fontsize=10, pad=8)

    def update(frame_idx):
        epoch, w = snapshots[frame_idx]
        # w shape: (1, n_heads, seq_len, seq_len) → take head, drop batch
        data = w[0, head, :seq_len, :seq_len].numpy()
        im.set_data(data)
        title.set_text(
            f"Attention  head {head}  ·  epoch {epoch}  "
            f"[{frame_idx + 1}/{len(snapshots)}]"
        )
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=interval,
        blit=False,
    )

    if save_path:
        anim.save(save_path, writer="pillow", dpi=100)
        print(f"Attention animation saved → {save_path}")

    plt.tight_layout()
    plt.show()
