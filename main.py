"""
main.py — End-to-end pipeline for mini-self-attention.

Usage:
    python main.py 

Pipeline:
    1. Load & tokenise corpus  (reused from mini-embedding)
    2. Build dataset           (full sentences, next-token objective)
    3. Init model              (optionally warm-start embeddings from mini-embedding)
    4. Train
    5. Inspect attention weights  (table + per-word vector comparison)
    6. Save model
    7. Plot: attention heatmaps, all-sentences overview, loss curve, animation
"""

import torch
from pathlib import Path

from src.tokenizer  import Tokenizer
from src.dataset    import SequenceDataset
from src.model      import AttentionModel
from src.train      import train
from src.utils      import (
    get_attention_weights,
    print_attention_table,
    compare_vectors,
)
from src.visualize  import (
    plot_attention,
    plot_all_sentences,
    plot_loss,
    animate_attention,
)

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH      = "data/corpus.txt"
PRETRAINED_EMB   = "../mini-embedding/outputs/embeddings.pt"  # set None to skip
EMB_DIM          = 8       # must match mini-embedding's EMBEDDING_DIM if loading
N_HEADS          = 2       # number of attention heads  (EMB_DIM must be divisible)
FF_DIM           = 32      # feedforward inner dimension
EPOCHS           = 200
LR               = 1e-3
BATCH_SIZE       = 4
OUTPUTS_DIR      = Path("outputs")
INSPECT_SENTENCE = "I like cats"    # sentence to inspect in detail after training
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── 1. Tokenise ───────────────────────────────────────────────────────────
    print("\n── Tokeniser ───────────────────────────────────────")
    tok = Tokenizer(CORPUS_PATH)
    print(tok)
    print(f"  Vocabulary: {list(tok.word2idx.keys())}\n")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    print("── Dataset ─────────────────────────────────────────")
    encoded = tok.encode_all()
    dataset = SequenceDataset(encoded)
    print(dataset)
    print("  Training objective: next-token prediction")
    print("  Each sentence  →  (input[:-1], target[1:])")
    for sent in tok.sentences[:3]:
        enc = tok.encode(sent)
        print(f"    {sent}  →  input={tok.decode(enc[:-1])}  target={tok.decode(enc[1:])}")
    print()

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("── Model ───────────────────────────────────────────")
    model = AttentionModel(
        vocab_size    = tok.vocab_size,
        emb_dim       = EMB_DIM,
        n_heads       = N_HEADS,
        ff_dim        = FF_DIM,
        pretrained_emb= PRETRAINED_EMB,
    )
    print(model, "\n")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("── Training ────────────────────────────────────────")
    snapshot_every = max(1, EPOCHS // 30)
    snapshots: list = []

    loss_history = train(
        model, dataset,
        epochs        = EPOCHS,
        lr            = LR,
        batch_size    = BATCH_SIZE,
        snapshots     = snapshots,
        snapshot_every= snapshot_every,
    )

    # ── 5. Inspect ────────────────────────────────────────────────────────────
    inspect_words   = INSPECT_SENTENCE.split()
    inspect_encoded = tok.encode(inspect_words)

    if len(inspect_encoded) >= 2:
        print(f"\n── Attention inspection: \"{INSPECT_SENTENCE}\" ───────────")
        weights = get_attention_weights(model, inspect_encoded)

        for h in range(N_HEADS):
            print_attention_table(weights, inspect_words, head=h)

        compare_vectors(model, inspect_encoded, inspect_words)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    model_path = OUTPUTS_DIR / "attention_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved → {model_path}\n")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    if len(inspect_encoded) >= 2:
        weights = get_attention_weights(model, inspect_encoded)
        plot_attention(
            weights, inspect_words,
            sentence_label = INSPECT_SENTENCE,
            save_path      = str(OUTPUTS_DIR / "attention_heatmap.png"),
        )

    plot_all_sentences(
        model, tok,
        save_path = str(OUTPUTS_DIR / "all_sentences_attention.png"),
    )

    plot_loss(
        loss_history,
        save_path = str(OUTPUTS_DIR / "loss_curve.png"),
    )

    if snapshots and len(inspect_encoded) >= 2:
        # Filter snapshots to the seq_len of the inspect sentence
        seq_len = len(inspect_encoded)
        valid   = [(e, w) for e, w in snapshots
                   if w.shape[-1] >= seq_len]
        if valid:
            animate_attention(
                valid, inspect_words,
                head      = 0,
                save_path = str(OUTPUTS_DIR / "attention_animation.gif"),
            )


if __name__ == "__main__":
    main()
