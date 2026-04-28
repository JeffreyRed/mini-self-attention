# Theory & Code Walkthrough — mini-self-attention

> Step 2 of the mini-LLM series. Prerequisite: [mini-embedding](../mini-embedding).

---

## Table of Contents

1. [The problem attention solves](#1-the-problem-attention-solves)
2. [The core idea — dynamic context](#2-the-core-idea--dynamic-context)
3. [Queries, Keys, and Values](#3-queries-keys-and-values)
4. [Scaled dot-product attention — the math](#4-scaled-dot-product-attention--the-math)
5. [Why scale by sqrt(d_k)?](#5-why-scale-by-sqrtd_k)
6. [Multi-head attention](#6-multi-head-attention)
7. [Residual connections and LayerNorm](#7-residual-connections-and-layernorm)
8. [The feedforward block](#8-the-feedforward-block)
9. [The complete encoder block](#9-the-complete-encoder-block)
10. [Training objective — next-token prediction](#10-training-objective--next-token-prediction)
11. [Code walkthrough](#11-code-walkthrough)
    - [tokenizer.py](#tokenizerpy)
    - [embedding.py](#embeddingpy)
    - [attention.py](#attentionpy)
    - [model.py](#modelpy)
    - [dataset.py](#datasetpy)
    - [train.py](#trainpy)
    - [utils.py](#utilspy)
    - [visualize.py](#visualizepy)
    - [main.py](#mainpy)
12. [The full data flow](#12-the-full-data-flow)
13. [What the attention weights tell you](#13-what-the-attention-weights-tell-you)
14. [Connection to transformers](#14-connection-to-transformers)

---

## 1. The problem attention solves

In mini-embedding, every word was assigned a fixed vector regardless of context:

```
"bank"  →  [0.3, -0.1, 0.8, ...]   always the same vector
```

This is fine for capturing general word similarity, but breaks for polysemy —
words with multiple meanings depending on context:

```
"I deposited money at the bank"     → financial institution
"We sat by the river bank"          → geographical feature
```

Both sentences produce the exact same vector for `"bank"`. The model has no
way to distinguish them. This is the fundamental limitation of static embeddings.

**Attention fixes this.** Instead of giving each word a fixed vector, attention
lets each word dynamically blend information from other words in the sentence.
The output vector for `"bank"` in the first sentence will be influenced by
`"deposited"` and `"money"`. In the second sentence it will be influenced by
`"river"` and `"sat"`. Same word, genuinely different representations.

---

## 2. The core idea — dynamic context

The intuition is simple: to understand a word, look at its neighbours.

Attention formalises this as a **weighted average**:

```
output["bank"] = 0.6 × value["deposited"]
               + 0.3 × value["money"]
               + 0.05 × value["at"]
               + 0.05 × value["bank"]
```

The weights (0.6, 0.3, 0.05, 0.05) are not fixed — they are **computed from
the content of the sentence itself**. The model learns to produce high weights
for relevant neighbours and low weights for irrelevant ones.

This is the entirety of what self-attention does. Everything else is
implementation detail.

---

## 3. Queries, Keys, and Values

Attention uses a database analogy to formalise "relevance":

| Concept | Role | Analogy |
|---|---|---|
| **Query  Q** | what this word is looking for | search query |
| **Key    K** | what each word can offer | index entry |
| **Value  V** | what each word actually sends if selected | document content |

Each word produces all three from its embedding vector via learned projections:

```
Q_i = embedding_i  @  W_Q       "what am I looking for?"
K_i = embedding_i  @  W_K       "what do I contain?"
V_i = embedding_i  @  W_V       "what do I send if selected?"
```

`W_Q`, `W_K`, `W_V` are the learned weight matrices — the parameters the model
trains. They are different for every attention head.

The relevance of word `j` to word `i` is measured by how similar `Q_i` is to
`K_j` — the query of word i dot-producted with the key of word j.

---

## 4. Scaled dot-product attention — the math

Given a sequence of `T` words with embedding dimension `d`:

```
X  shape: (T, d)     — input embeddings (one row per word)

Q = X @ W_Q          shape: (T, d_k)
K = X @ W_K          shape: (T, d_k)
V = X @ W_V          shape: (T, d_v)
```

**Step 1 — raw scores:**

```
scores = Q @ K^T / sqrt(d_k)      shape: (T, T)
```

Entry `scores[i, j]` = how much word `i` should attend to word `j`.

**Step 2 — softmax normalisation:**

```
weights = softmax(scores, dim=-1)  shape: (T, T)
```

Each row now sums to 1. Entry `weights[i, j]` is the probability that word `i`
attends to word `j`.

**Step 3 — weighted sum of values:**

```
output = weights @ V               shape: (T, d_v)
```

Row `i` of the output is a blend of all value vectors, weighted by how much
word `i` attended to each position.

**Complete formula:**

```
Attention(Q, K, V)  =  softmax( Q @ K^T / sqrt(d_k) )  @  V
```

This is equation (1) from Vaswani et al. (2017) — *Attention Is All You Need*.

---

## 5. Why scale by sqrt(d_k)?

Without the scaling factor, dot products grow large when `d_k` is large.
For example with `d_k = 64`:

```
Q_i  and  K_j  are random vectors of length 64.
Their dot product has variance ≈ 64.
Standard deviation ≈ 8.
```

Large dot products push the softmax into its saturation region where gradients
become extremely small (nearly zero). The model stops learning.

Dividing by `sqrt(d_k)` brings the variance back to 1 regardless of dimension,
keeping softmax in its useful range throughout training.

---

## 6. Multi-head attention

Running a single attention computation is good. Running `h` independent ones
in parallel is better.

```
head_1  =  Attention(X @ W_Q1, X @ W_K1, X @ W_V1)
head_2  =  Attention(X @ W_Q2, X @ W_K2, X @ W_V2)
  ...
head_h  =  Attention(X @ W_Qh, X @ W_Kh, X @ W_Vh)

MultiHead(X)  =  concat(head_1, ..., head_h)  @  W_O
```

Each head has its own `W_Q`, `W_K`, `W_V`. Each can specialise:

- Head 1 might focus on syntactic relationships (subject → verb)
- Head 2 might focus on semantic similarity (synonyms, related concepts)
- Head 3 might focus on local proximity (adjacent words)

The heads run on reduced dimension `head_dim = emb_dim / n_heads` so the total
parameter count stays the same as a single attention with full `emb_dim`.

Implementation detail: rather than a loop over heads, we use a single fused
projection `W_Q` of shape `(emb_dim, emb_dim)` and then reshape to split into
heads. This is equivalent but much faster on a GPU.

---

## 7. Residual connections and LayerNorm

After every sublayer (attention or feedforward), two things happen:

**Residual connection:**

```
x  =  x  +  sublayer(x)
```

The input is added directly to the output. This has two benefits:

1. Gradients flow directly through the `+` operation to earlier layers —
   solves the vanishing gradient problem in deep networks.
2. The sublayer only needs to learn *incremental refinements* to the
   representation, not a full re-encoding. Much easier to optimise.

**LayerNorm:**

```
x  =  LayerNorm(x)
```

Normalises each vector to zero mean and unit variance across the feature
dimension. This stabilises training by preventing the values from drifting to
extreme ranges between layers.

Together the pattern is called **Add & Norm** and appears after every sublayer
in every transformer block ever published.

---

## 8. The feedforward block

After attention, each position passes independently through a small two-layer
MLP:

```
FFN(x)  =  ReLU( x @ W_1 + b_1 )  @  W_2  +  b_2
```

The inner dimension is conventionally `4 × emb_dim` (e.g. `emb_dim=8` →
`ff_dim=32` here).

Why is this needed after attention? Attention mixes *which* information to
collect from other positions. The feedforward block then processes *what to do*
with that collected information — it applies a non-linear transformation at
each position independently. The combination of attention (global mixing) and
feedforward (local processing) is what gives the transformer its power.

---

## 9. The complete encoder block

Putting it all together:

```
Input:  X  (batch, seq_len, emb_dim)
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  MultiHeadAttention(X)                              │
│  Q = X@W_Q   K = X@W_K   V = X@W_V                 │
│  scores = Q@K^T / sqrt(d_k)                         │
│  weights = softmax(scores)                          │
│  head_out = weights @ V          for each head      │
│  out = concat(heads) @ W_O                          │
└─────────────────────────────────────────────────────┘
              │  attn_out
              ▼
        x = LayerNorm( x + attn_out )     ← residual + norm
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  FeedForward(x)                                     │
│  = ReLU( x @ W_1 ) @ W_2                           │
└─────────────────────────────────────────────────────┘
              │  ff_out
              ▼
        x = LayerNorm( x + ff_out )       ← residual + norm
              │
              ▼
Output: context-aware vectors  (batch, seq_len, emb_dim)
```

This is one encoder block. BERT uses 12 of these stacked. GPT uses decoder
blocks (same structure plus a causal mask). The mini-transformer step will
stack `N` of these and add positional encoding.

---

## 10. Training objective — next-token prediction

**Key difference from mini-embedding:**

| | mini-embedding | mini-self-attention |
|---|---|---|
| Input to model | single word index | full sentence (sequence of indices) |
| Training pairs | (target word, context word) | (sentence[:-1], sentence[1:]) |
| Task | predict context word from target | predict next word at every position |
| Output | one probability distribution | T probability distributions |

For the sentence `["I", "like", "cats"]` encoded as `[0, 9, 4]`:

```
input:   [0,  9]       →   "I like"
target:  [9,  4]       →   "like cats"

At position 0: model sees [I]        → must predict "like"
At position 1: model sees [I, like]  → must predict "cats"
```

The loss is the average cross-entropy across all positions and all sentences in
the batch. This objective forces the model to build context-aware
representations — to predict `"cats"` after `"I like"`, the model must encode
the fact that `"like"` is typically followed by something positive and
concrete.

Note: for a fully correct causal (GPT-style) model, a triangular mask would
prevent position `i` from attending to positions `> i`. This repo keeps things
simple and uses a padding mask only, giving bidirectional attention (like BERT).
The mini-transformer step will add the causal mask.

---

## 11. Code walkthrough

### `tokenizer.py`

Same vocabulary-building logic as mini-embedding's `dataset.py`, but the
output is different. Instead of `(target, context)` pairs it returns full
encoded sentences — integer sequences — because attention operates over the
entire sequence at once.

```python
tok = Tokenizer("data/corpus.txt")
tok.encode(["I", "like", "cats"])   # → [0, 9, 4]
tok.decode([0, 9, 4])               # → ["I", "like", "cats"]
tok.encode_all()                    # → [[0, 9, 4], [0, 9, 3], ...]
```

---

### `embedding.py`

A thin wrapper around `nn.Embedding` that can optionally load the weights saved
by mini-embedding's `outputs/embeddings.pt`. If the file exists and the shape
matches, those trained weights are used as the starting point. If not, random
initialisation is used — the model still trains fine either way.

This makes the connection between the two projects concrete: the vectors that
mini-embedding spent 150 epochs learning are directly reused as the input layer
here.

```python
# With pretrained weights (preferred)
emb = EmbeddingLayer(vocab_size=18, emb_dim=8,
                     pretrained_path="../mini-embedding/outputs/embeddings.pt")

# Without (random init)
emb = EmbeddingLayer(vocab_size=18, emb_dim=8)
```

---

### `attention.py`

The central file. Three components:

**`scaled_dot_product_attention(Q, K, V, mask)`**

The pure mathematical operation. No learnable parameters here — just the
`Q @ K^T / sqrt(d_k)` → softmax → `@ V` computation. Takes tensors of shape
`(batch, heads, seq_len, d_k)` and returns output + weight tensors.

```python
scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output  = torch.matmul(weights, V)
```

**`AttentionHead`**

A single head: three `nn.Linear` layers (W_Q, W_K, W_V) plus a call to
`scaled_dot_product_attention`. Included separately for educational clarity —
you can instantiate one head and step through it manually.

**`MultiHeadAttention`**

The production version. Uses a single fused W_Q/W_K/W_V of shape
`(emb_dim, emb_dim)` then reshapes to split into `n_heads` heads internally
via `_split_heads()`. This is equivalent to running heads in a loop but
processes all of them in one matrix multiplication.

```python
# _split_heads: (batch, seq_len, emb_dim) → (batch, n_heads, seq_len, head_dim)
Q = x.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
```

After attention, heads are merged back:

```python
# (batch, n_heads, seq_len, head_dim) → (batch, seq_len, emb_dim)
merged = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, emb_dim)
output = self.W_O(merged)
```

The `.contiguous()` call is required because `.transpose()` returns a
non-contiguous view of memory, and `.view()` requires contiguous storage.

---

### `model.py`

**`FeedForward`**

Two-layer MLP with ReLU. The inner dimension defaults to `4 × emb_dim`
following the original transformer paper. Each position is processed
independently — no information crosses between positions here.

**`SelfAttentionBlock`**

One complete encoder block: MultiHeadAttention + residual + LayerNorm,
followed by FeedForward + residual + LayerNorm.

```python
# Sub-layer 1
attn_out, weights = self.attention(x, mask)
x = self.norm1(x + attn_out)        # residual + norm

# Sub-layer 2
x = self.norm2(x + self.ff(x))      # residual + norm
```

**`AttentionModel`**

The full model: EmbeddingLayer → SelfAttentionBlock → Linear head → logits.
The linear head projects from `emb_dim` back to `vocab_size`, giving one
probability distribution per position.

`get_context_vectors()` is a convenience method that stops before the linear
head and returns the raw context-aware embeddings — useful for comparing
how much attention changed each word's vector.

---

### `dataset.py`

**`SequenceDataset`**

Takes the list of encoded sentences and produces `(input, target)` pairs
via the shift-by-one trick:

```python
input  = sentence[:-1]   # all words except last
target = sentence[1:]    # all words except first
```

**`collate_fn`**

Sentences in the corpus have different lengths. When the DataLoader batches
them together, they must be padded to the same length. `collate_fn` right-pads
all sequences with index `0` and builds a padding mask that tells the model
which positions are real and which are padding.

The padding mask is a boolean tensor of shape `(batch, 1, seq_len, seq_len)`
where `True` means "mask this position out" (set to `-inf` before softmax).

---

### `train.py`

Same five-step loop as mini-embedding, but with two additions:

**Gradient clipping:**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents individual gradient steps from being too large, which is especially
important when training attention models on small datasets where occasional
large losses can destabilise training.

**`ignore_index=0` in CrossEntropyLoss:**

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
```

Tells the loss function to skip positions where the target is `0` (padding).
Without this, the model would waste capacity trying to predict padding tokens.

---

### `utils.py`

**`get_attention_weights(model, sentence)`**

Runs a single forward pass and returns the `(n_heads, seq_len, seq_len)`
attention weight tensor for inspection.

**`print_attention_table(weights, words, head)`**

Prints the attention matrix as a readable table. Rows are query positions,
columns are key positions, values are attention probabilities. Includes a
small `█` bar per cell proportional to the weight.

**`compare_vectors(model, sentence, words)`**

Computes cosine similarity between each word's raw embedding vector and its
context-aware vector after attention. Low similarity means attention
significantly changed that word's representation — it borrowed heavily from
its neighbours.

```
"cats"    sim=+0.71   change: ████
"like"    sim=+0.84   change: ██
"I"       sim=+0.92   change: █
```

---

### `visualize.py`

**`plot_attention(weights, words)`**

One subplot per attention head. Each subplot is a heatmap of shape
`(seq_len × seq_len)` with numeric values in cells. Brighter blue = more
attention. The x-axis is "keys" (what was attended to), y-axis is "queries"
(which word is attending).

**`plot_all_sentences(model, tokenizer)`**

Runs every sentence in the corpus through the model and plots the mean
attention (averaged across heads) for each one in a grid. Useful for seeing
global patterns — does the model generally attend to adjacent words? Or does
it look further back?

**`animate_attention(snapshots, words, head)`**

Animates how one head's attention pattern changes during training. Early
epochs often show near-uniform attention (the model has not yet learned what
to focus on). Later epochs show structured patterns — some positions strongly
attending to specific other positions.

---

### `main.py`

The entry point. Runs the full pipeline and exposes all configuration at the
top of the file. One config worth noting:

```python
PRETRAINED_EMB = "../mini-embedding/outputs/embeddings.pt"
```

If you run mini-embedding first and this path is correct, the attention model
starts with meaningful word vectors rather than random noise. Training
converges faster and the early attention patterns are more interpretable.

Set to `None` to train from scratch.

---

## 12. The full data flow

Tracing `"I like cats"` encoded as `[0, 9, 4]` through one forward pass:

```
input tensor: [0, 9, 4]     shape: (1, 3)
                 │
                 ▼  EmbeddingLayer
[ [0.12, 0.34, ...],        shape: (1, 3, 8)
  [0.21, 0.44, ...],        ← one 8-dim vector per word
  [0.91, 0.80, ...] ]
                 │
                 ▼  MultiHeadAttention  (n_heads=2, head_dim=4)
  W_Q, W_K, W_V projections  →  Q, K, V  shape each: (1, 2, 3, 4)
  scores = Q @ K^T / sqrt(4)              shape: (1, 2, 3, 3)
  weights = softmax(scores)               shape: (1, 2, 3, 3)  ← attention maps
  attn_out = weights @ V                  shape: (1, 2, 3, 4)
  merged = concat heads                   shape: (1, 3, 8)
  out = W_O(merged)                       shape: (1, 3, 8)
                 │
                 ▼  residual + LayerNorm
  x = LayerNorm(x + out)                  shape: (1, 3, 8)
                 │
                 ▼  FeedForward  (ff_dim=32)
  x = LayerNorm(x + FFN(x))              shape: (1, 3, 8)
                 │
                 ▼  Linear head
  logits                                  shape: (1, 3, 18)
                 │
                 ▼  CrossEntropyLoss  vs  target [9, 4, ?]
  loss  →  backprop  →  update all weights
```

---

## 13. What the attention weights tell you

After training, the `(seq_len × seq_len)` attention matrix for each head is the
most informative diagnostic. Here is how to read it:

```
         I      like    cats
I     [ 0.70,  0.20,   0.10 ]  ← "I" mostly attends to itself
like  [ 0.15,  0.60,   0.25 ]  ← "like" attends mostly to itself and "cats"
cats  [ 0.10,  0.55,   0.35 ]  ← "cats" attends strongly to "like"
```

Reading this: `"cats"` has a high attention weight toward `"like"`. This means
when building the context-aware vector for `"cats"`, the model borrows
significantly from the `"like"` vector. The representation of `"cats"` in this
sentence is influenced by the fact that it was liked — not just by what `"cats"`
means in isolation.

Different heads capture different patterns:
- One head might show diagonal dominance (words attend to themselves)
- Another might show that verbs attend to their objects
- Another might show local neighbourhood patterns

On this tiny corpus the patterns will be weak. With real data at GPT scale,
the heads develop strikingly specialised behaviours.

---

## 14. Connection to transformers

This project implements one encoder block. The full transformer architecture
adds two things on top:

**Positional encoding** — self-attention has no built-in notion of word order.
`"cats like I"` and `"I like cats"` produce the same attention scores because
the positions are not encoded. Positional encoding injects position information
into the embeddings before the first attention block. This is the first thing
added in the mini-transformer step.

**Stacking N blocks** — BERT uses 12 encoder blocks, GPT-3 uses 96 decoder
blocks. Each block refines the representations produced by the previous one.
The mini-transformer step will make `N` configurable.

```
mini-embedding       token index → static vector
     ↓
mini-self-attention  static vector → context-aware vector  (1 block)   ← you are here
     ↓
mini-transformer     + positional encoding + N stacked blocks
     ↓
mini-gpt             + causal mask + generation
```

---

*Next: `mini-transformer` — positional encoding, stacked encoder blocks, and a
text generation loop.*
