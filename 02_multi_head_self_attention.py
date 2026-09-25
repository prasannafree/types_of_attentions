"""
================================================================================
02 — MULTI-HEAD SELF-ATTENTION (MHA)
================================================================================

PAPER: "Attention Is All You Need" (Vaswani et al., 2017)

THE KEY IDEA:
    Instead of doing ONE big attention operation, split Q, K, V into h
    independent "heads", run attention in parallel on each, then concatenate.

    Why? A single attention head can only focus on ONE type of relationship
    at each position. Multiple heads let the model SIMULTANEOUSLY attend to:
        - Head 1: syntactic relationships (subject ↔ verb)
        - Head 2: coreference (pronoun ↔ noun it refers to)
        - Head 3: positional patterns (nearby tokens)
        - Head 4: semantic similarity
        ... etc.

FORMULA:
    head_i = Attention(Q · W_Q^i, K · W_K^i, V · W_V^i)
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

    Where each W_Q^i is (d_model, d_k), d_k = d_model / h

WHAT IT COSTS vs SINGLE-HEAD:
    - Same total FLOPs! Each head operates on d_model/h dimensions.
    - h heads × (d_model/h)² per head = d_model²/h total per head matrix
    - But you do h of them → d_model² total, SAME as single-head.
    - The win is purely representational (richer, more diverse attention patterns).

MEMORY:
    - Still O(n²) for the attention matrix (per head), but now h copies of it.
    - In practice: O(h · n²) attention weight storage, though h is typically 8-16.

SELF vs CROSS ATTENTION:
    - Self-attention: Q, K, V all come from the SAME sequence.
    - Cross-attention: Q from one sequence, K, V from another (e.g., decoder
      attending to encoder in translation).
    - This script implements SELF-attention (all from same input).

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention as described in the original Transformer.

    Architecture:
        Input x → [W_Q, W_K, W_V projections]
                → split into h heads
                → scaled dot-product attention per head
                → concatenate heads
                → W_O projection → output

    Each head gets its own "view" of the input in a lower-dimensional subspace.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # --- Projections ---
        # In practice, we use ONE big linear layer and then reshape into heads.
        # This is mathematically equivalent to h separate smaller projections,
        # but much faster on GPU (one big matmul vs h small ones).
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                     # (batch, seq_len, d_model)
        mask: torch.Tensor | None = None,     # (batch, 1, seq_len, seq_len)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # ---- Step 1: Project to Q, K, V ----
        Q = self.W_Q(x)  # (batch, seq, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # ---- Step 2: Reshape into multiple heads ----
        # (batch, seq, d_model) → (batch, seq, h, d_k) → (batch, h, seq, d_k)
        #
        # Why transpose? We want each head to be an independent "batch" dimension
        # so that the attention matmul operates on (seq, d_k) per head.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Now: (batch, h, seq, d_k)

        # ---- Step 3: Scaled dot-product attention (per head) ----
        # scores: (batch, h, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # context: (batch, h, seq_q, d_k)
        context = torch.matmul(weights, V)

        # ---- Step 4: Concatenate heads ----
        # (batch, h, seq, d_k) → (batch, seq, h, d_k) → (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # ---- Step 5: Final projection ----
        output = self.W_O(context)

        return output, weights


# ==============================================================================
# DEMO
# ==============================================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """(1, 1, seq, seq) — broadcastable across batch and heads."""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)


def demo():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8  # → d_k = 64/8 = 8 per head

    x = torch.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)

    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mha(x, mask=mask)

    print("=" * 70)
    print("MULTI-HEAD SELF-ATTENTION — DEMO")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  d_model    = {d_model}")
    print(f"  num_heads  = {num_heads}")
    print(f"  d_k (per head) = {d_model // num_heads}")

    print(f"\nShapes:")
    print(f"  Input:             {x.shape}")
    print(f"  Output:            {output.shape}")
    print(f"  Attention weights: {weights.shape}  (batch, heads, seq_q, seq_k)")

    # --- Show that different heads learn different patterns ---
    print(f"\n{'─' * 70}")
    print("ATTENTION PATTERNS PER HEAD (batch=0, showing which positions attend where)")
    print(f"{'─' * 70}")
    for head_idx in range(num_heads):
        w = weights[0, head_idx].detach().numpy()
        # Show which position each query attends to most
        max_attend = w.argmax(axis=-1)
        print(f"  Head {head_idx}: each position attends most to → {max_attend.tolist()}")

    # --- Parameter count comparison ---
    print(f"\n{'─' * 70}")
    print("PARAMETER COUNT ANALYSIS")
    print(f"{'─' * 70}")
    total = sum(p.numel() for p in mha.parameters())
    print(f"  Total params: {total:,}")
    print(f"  W_Q: {d_model}×{d_model} = {d_model**2:,}")
    print(f"  W_K: {d_model}×{d_model} = {d_model**2:,}")
    print(f"  W_V: {d_model}×{d_model} = {d_model**2:,}")
    print(f"  W_O: {d_model}×{d_model} = {d_model**2:,}")
    print(f"  Total = 4 × d_model² = {4 * d_model**2:,}")
    print(f"\n  Note: Same parameter count whether num_heads=1 or num_heads=8!")
    print(f"  The heads don't add parameters — they split existing capacity.")

    # --- KV Cache size (important for inference) ---
    print(f"\n{'─' * 70}")
    print("KV CACHE (why this matters for inference)")
    print(f"{'─' * 70}")
    kv_cache_per_layer = 2 * seq_len * d_model  # K and V, each (seq, d_model)
    print(f"  Per layer, per sequence:")
    print(f"    K cache: {seq_len} × {d_model} = {seq_len * d_model:,} floats")
    print(f"    V cache: {seq_len} × {d_model} = {seq_len * d_model:,} floats")
    print(f"    Total:   {kv_cache_per_layer:,} floats")
    print(f"\n  For a 32-layer model with seq_len=2048:")
    kv_32 = 2 * 2048 * d_model * 32
    print(f"    {kv_32:,} floats = {kv_32 * 4 / 1024 / 1024:.1f} MB (fp32)")
    print(f"    This grows linearly with batch_size — the bottleneck for LLM serving!")
    print(f"    → MQA and GQA (scripts 03, 04) exist to reduce this.")


if __name__ == "__main__":
    demo()
