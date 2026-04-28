"""mini-self-attention — source package."""
from src.tokenizer  import Tokenizer
from src.embedding  import EmbeddingLayer
from src.attention  import scaled_dot_product_attention, MultiHeadAttention
from src.model      import AttentionModel
from src.dataset    import SequenceDataset, collate_fn
from src.train      import train
from src.utils      import get_attention_weights, print_attention_table, compare_vectors
from src.visualize  import plot_attention, plot_loss, animate_attention

__all__ = [
    "Tokenizer", "EmbeddingLayer",
    "scaled_dot_product_attention", "MultiHeadAttention",
    "AttentionModel",
    "SequenceDataset", "collate_fn",
    "train",
    "get_attention_weights", "print_attention_table", "compare_vectors",
    "plot_attention", "plot_loss", "animate_attention",
]
