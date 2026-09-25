"""
================================================================================
01 — SCALED DOT-PRODUCT ATTENTION
================================================================================

THE FOUNDATION:
    This is the atomic operation inside every transformer. All other attention
    variants in this series are built on top of (or modify) this formula.

FORMULA:
    Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V

    Where:
        Q  = Query matrix   (batch, seq_len_q, d_k)
        K  = Key matrix     (batch, seq_len_k, d_k)
        V  = Value matrix   (batch, seq_len_k, d_v)
        d_k = dimension of keys (used for scaling)

INTUITION:
    Think of it like a database lookup, but "soft" (fuzzy):
        1. You have a QUERY — "what am I looking for?"
        2. You compare the query against every KEY — "how relevant is each item?"
        3. The comparison scores (after softmax) become WEIGHTS.
        4. You use those weights to take a weighted average of VALUES.

    The "scaled" part (dividing by √d_k) prevents the dot products from
    growing too large when d_k is big. Large dot products push softmax into
    regions with tiny gradients (saturation), making training unstable.

WHAT IT COSTS:
    - Time complexity:  O(n² · d)     where n = sequence length
    - Memory complexity: O(n²)         for storing the attention matrix
    This quadratic cost in sequence length is WHY all the variants below exist.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# IMPLEMENTATION 1: Pure function (no learnable parameters)
# ==============================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,    # (batch, seq_q, d_k)
    key: torch.Tensor,      # (batch, seq_k, d_k)
    value: torch.Tensor,    # (batch, seq_k, d_v)
    mask: torch.Tensor | None = None,  # (batch, seq_q, seq_k) or broadcastable
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention from scratch.

    Returns:
        output:  (batch, seq_q, d_v) — the weighted sum of values
        weights: (batch, seq_q, seq_k) — the attention weights (for visualization)
    """
    d_k = query.size(-1)

    # ---- Step 1: Compute raw attention scores ----
    # Q · Kᵀ → (batch, seq_q, seq_k)
    # Each entry (i, j) measures "how much should position i attend to position j?"
    scores = torch.matmul(query, key.transpose(-2, -1))

    # ---- Step 2: Scale ----
    # Without scaling, when d_k is large (e.g. 512), the dot products can be
    # very large in magnitude. softmax of large values → near-one-hot → vanishing
    # gradients. Dividing by √d_k keeps the variance of the scores ≈ 1.
    scores = scores / math.sqrt(d_k)

    # ---- Step 3: Apply mask (optional) ----
    # For CAUSAL (autoregressive) decoding, we mask future positions so the model
    # can't "cheat" by looking ahead. We set masked positions to -inf so that
    # softmax gives them 0 weight.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # ---- Step 4: Softmax → attention weights ----
    # Converts raw scores to a probability distribution over keys for each query.
    weights = F.softmax(scores, dim=-1)

    # ---- Step 5: Weighted sum of values ----
    # (batch, seq_q, seq_k) @ (batch, seq_k, d_v) → (batch, seq_q, d_v)
    output = torch.matmul(weights, value)

    return output, weights


# ==============================================================================
# IMPLEMENTATION 2: nn.Module with linear projections (closer to real use)
# ==============================================================================

class ScaledDotProductAttentionLayer(nn.Module):
    """
    A single-head attention layer with learnable W_Q, W_K, W_V, W_O projections.

    In practice, raw input tokens don't come pre-split into Q, K, V.
    Instead, the model LEARNS how to project the input into these roles.

    x → W_Q → Q
    x → W_K → K
    x → W_V → V

    This is what makes attention so powerful: the model decides what to
    query for, what to expose as searchable keys, and what information
    to return as values — all learned end-to-end.
    """

    def __init__(self, d_model: int, d_k: int | None = None, d_v: int | None = None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.W_O = nn.Linear(d_v, d_model, bias=False)  # project back to d_model

        self.d_k = d_k

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        Q = self.W_Q(x)  # (batch, seq, d_k)
        K = self.W_K(x)  # (batch, seq, d_k)
        V = self.W_V(x)  # (batch, seq, d_v)

        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        return self.W_O(output), weights


# ==============================================================================
# DEMO: Run it and visualize what's happening
# ==============================================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Creates a lower-triangular mask for autoregressive (causal) attention.

    Position i can attend to positions 0..i but NOT i+1..n-1.

        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1
    """
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)  # (1, seq, seq)


def demo():
    torch.manual_seed(42)

    # --- Configuration ---
    batch_size = 2
    seq_len = 6
    d_model = 16

    # --- Create dummy input (imagine these are token embeddings) ---
    x = torch.randn(batch_size, seq_len, d_model)

    print("=" * 70)
    print("SCALED DOT-PRODUCT ATTENTION — DEMO")
    print("=" * 70)

    # --- 1. Without mask (bidirectional, like BERT) ---
    layer = ScaledDotProductAttentionLayer(d_model=d_model)
    output, weights = layer(x)
    print(f"\nInput shape:            {x.shape}")
    print(f"Output shape:           {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights for batch=0 (each row sums to 1):")
    print(weights[0].detach().numpy().round(3))

    # --- 2. With causal mask (autoregressive, like GPT) ---
    mask = create_causal_mask(seq_len)
    output_causal, weights_causal = layer(x, mask=mask)
    print(f"\nCausal attention weights for batch=0:")
    print(f"(Notice: upper triangle is 0 — can't look at future tokens)")
    print(weights_causal[0].detach().numpy().round(3))

    # --- 3. Show the scaling effect ---
    print("\n" + "=" * 70)
    print("WHY SCALING MATTERS")
    print("=" * 70)
    Q = torch.randn(1, 4, 512)  # high dimension
    K = torch.randn(1, 4, 512)

    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    scores_scaled = scores_unscaled / math.sqrt(512)

    print(f"\nd_k = 512")
    print(f"Unscaled scores — mean: {scores_unscaled.mean():.2f}, "
          f"std: {scores_unscaled.std():.2f}")
    print(f"Scaled scores   — mean: {scores_scaled.mean():.2f}, "
          f"std: {scores_scaled.std():.2f}")

    weights_unscaled = F.softmax(scores_unscaled, dim=-1)
    weights_scaled = F.softmax(scores_scaled, dim=-1)
    print(f"\nSoftmax of unscaled (near one-hot, bad gradients):")
    print(weights_unscaled[0].detach().numpy().round(4))
    print(f"Softmax of scaled (smoother, better gradients):")
    print(weights_scaled[0].detach().numpy().round(4))

    # --- 4. Parameter count ---
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"\nTotal learnable parameters: {total_params:,}")
    print(f"  W_Q: {d_model}×{d_model} = {d_model*d_model}")
    print(f"  W_K: {d_model}×{d_model} = {d_model*d_model}")
    print(f"  W_V: {d_model}×{d_model} = {d_model*d_model}")
    print(f"  W_O: {d_model}×{d_model} = {d_model*d_model}")


if __name__ == "__main__":
    demo()
