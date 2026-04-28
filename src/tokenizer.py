"""
tokenizer.py — Vocabulary builder and sentence encoder.

Reuses the same corpus and vocabulary logic as mini-embedding so the two
projects are directly comparable.  The key difference is the output:
mini-embedding produced (target, context) pairs for Skip-gram training.
Here we produce full encoded sentences — integer sequences of length T —
because attention operates over an entire sequence at once, not word pairs.
"""

from typing import List, Dict, Tuple


class Tokenizer:
    """
    Builds a vocabulary from a corpus and encodes sentences as index sequences.

    Args:
        path (str): Path to plain-text corpus (one sentence per line).

    Attributes:
        word2idx (Dict[str, int]): word  → integer index
        idx2word (Dict[int, str]): index → word
        vocab_size (int):          number of unique tokens
        sentences (List[List[str]]): raw tokenised sentences
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            self.sentences: List[List[str]] = [
                line.strip().split() for line in f if line.strip()
            ]
        self._build_vocab()

    # ------------------------------------------------------------------
    def _build_vocab(self) -> None:
        words = [w for s in self.sentences for w in s]
        vocab = sorted(set(words))
        self.word2idx: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
        self.idx2word: Dict[int, str] = {i: w for w, i in self.word2idx.items()}
        self.vocab_size: int = len(self.word2idx)

    # ------------------------------------------------------------------
    def encode(self, sentence: List[str]) -> List[int]:
        """Converts a list of word strings to a list of integer indices."""
        return [self.word2idx[w] for w in sentence if w in self.word2idx]

    def decode(self, indices: List[int]) -> List[str]:
        """Converts a list of integer indices back to word strings."""
        return [self.idx2word[i] for i in indices]

    def encode_all(self) -> List[List[int]]:
        """Returns every sentence in the corpus as an index sequence."""
        return [self.encode(s) for s in self.sentences]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"sentences={len(self.sentences)})"
        )
