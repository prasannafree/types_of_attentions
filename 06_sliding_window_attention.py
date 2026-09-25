"""
================================================================================
06 — SLIDING WINDOW ATTENTION (SWA)
================================================================================

PAPER: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
       Also used in: Mistral 7B, Mixtral, and many long-context models.

THE PROBLEM:
    Standard attention is O(n²) in sequence length. For n=100K tokens,
    the attention matrix has 10 BILLION entries. That's impractical.

    But does every token REALLY need to attend to every other token?
    In many cases, LOCAL context is most important (nearby words matter most).

THE SLIDING WINDOW IDEA:
    Each token only attends to its w nearest neighbors (w = window size).

    Standard attention (seq_len=8):          Sliding window (w=3):
    ┌─┬─┬─┬─┬─┬─┬─┬─┐                     ┌─┬─┬─┬─┬─┬─┬─┬─┐
    │■│■│■│■│■│■│■│■│                     │■│■│ │ │ │ │ │ │
    │■│■│■│■│■│■│■│■│                     │■│■│■│ │ │ │ │ │
    │■│■│■│■│■│■│■│■│                     │ │■│■│■│ │ │ │ │
    │■│■│■│■│■│■│■│■│                     │ │ │■│■│■│ │ │ │
    │■│■│■│■│■│■│■│■│                     │ │ │ │■│■│■│ │ │
    │■│■│■│■│■│■│■│■│                     │ │ │ │ │■│■│■│ │
    │■│■│■│■│■│■│■│■│                     │ │ │ │ │ │■│■│■│
    │■│■│■│■│■│■│■│■│                     │ │ │ │ │ │ │■│■│
    └─┴─┴─┴─┴─┴─┴─┴─┘                     └─┴─┴─┴─┴─┴─┴─┴─┘
       64 entries                              ~24 entries

WHAT IT REDUCES:
    - Time complexity:  O(n² · d)  →  O(n · w · d)
    - Memory complexity: O(n²)    →  O(n · w)
    - When w << n, this is almost LINEAR in sequence length!

EFFECTIVE RECEPTIVE FIELD:
    A key insight: even though each layer only sees w tokens, stacking L layers
    gives an effective receptive field of L × w tokens (information propagates
    through the layers like in a CNN).

    With w=4096 and L=32 layers: effective field = 32 × 4096 = 131,072 tokens!

MISTRAL'S APPROACH:
    Mistral 7B uses SWA with w=4096 in EVERY layer. Combined with a rolling
    KV cache (only keep the last w positions), this gives:
    - Fixed memory regardless of sequence length!
    - O(n·w) compute instead of O(n²)
    - Works because information flows through layers

HOW IT COMBINES WITH CAUSAL MASKING:
    For autoregressive models, we combine SWA with causal masking:
    Position i can attend to positions max(0, i-w+1) ... i

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Create a sliding window attention mask.

    Args:
        seq_len:     length of the sequence
        window_size: how many positions each token can attend to
        causal:      if True, also apply causal masking (can't look ahead)

    Returns:
        mask: (1, 1, seq_len, seq_len) boolean mask (True = attend, False = mask out)
    """
    # Start with all-ones (attend everywhere)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    # Apply sliding window: position i can attend to [i-w+1, i+w-1]
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) >= window_size:
                mask[i, j] = False

    # Apply causal masking: position i can't attend to j > i
    if causal:
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        mask = mask & causal_mask

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)


def create_sliding_window_mask_efficient(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Efficient vectorized version (no Python loops).
    """
    # Create position indices
    rows = torch.arange(seq_len).unsqueeze(1)  # (seq, 1)
    cols = torch.arange(seq_len).unsqueeze(0)  # (1, seq)

    # Window constraint: |i - j| < window_size
    mask = (rows - cols).abs() < window_size

    # Causal constraint: j <= i
    if causal:
        mask = mask & (cols <= rows)

    return mask.unsqueeze(0).unsqueeze(0)


class SlidingWindowAttention(nn.Module):
    """
    Multi-Head Attention with Sliding Window masking.

    This implementation uses masking to zero out positions outside the window.
    In production (e.g., FlashAttention), the implementation is more optimized
    to avoid even computing those entries.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.causal = causal

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Create sliding window mask
        mask = create_sliding_window_mask_efficient(
            seq_len, self.window_size, self.causal
        ).to(x.device)

        # Standard multi-head attention with the window mask
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply window mask
        scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        # Replace NaN with 0 (happens when all positions are masked, e.g., empty window)
        weights = weights.nan_to_num(0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)

        return output, weights


# ==============================================================================
# DEMO
# ==============================================================================

def demo():
    torch.manual_seed(42)

    seq_len = 12
    d_model = 32
    num_heads = 4
    window_size = 4

    x = torch.randn(1, seq_len, d_model)

    print("=" * 70)
    print("SLIDING WINDOW ATTENTION — DEMO")
    print("=" * 70)

    # --- Visualize masks ---
    print(f"\nWindow size = {window_size}, Sequence length = {seq_len}")

    print(f"\n{'─' * 70}")
    print("MASK VISUALIZATION (■ = attend, · = masked)")
    print(f"{'─' * 70}")

    # Bidirectional window
    mask_bidir = create_sliding_window_mask_efficient(seq_len, window_size, causal=False)
    print(f"\nBidirectional sliding window (w={window_size}):")
    m = mask_bidir[0, 0]
    for i in range(seq_len):
        row = "  "
        for j in range(seq_len):
            row += "■ " if m[i, j] else "· "
        print(row)

    # Causal window (what Mistral uses)
    mask_causal = create_sliding_window_mask_efficient(seq_len, window_size, causal=True)
    print(f"\nCausal sliding window (w={window_size}):")
    m = mask_causal[0, 0]
    for i in range(seq_len):
        row = "  "
        for j in range(seq_len):
            row += "■ " if m[i, j] else "· "
        print(row)

    # Full causal (for comparison)
    full_causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    print(f"\nFull causal attention (for comparison):")
    for i in range(seq_len):
        row = "  "
        for j in range(seq_len):
            row += "■ " if full_causal[i, j] else "· "
        print(row)

    # --- Run the model ---
    swa = SlidingWindowAttention(
        d_model=d_model,
        num_heads=num_heads,
        window_size=window_size,
    )
    output, weights = swa(x)

    print(f"\n{'─' * 70}")
    print("MODEL OUTPUT")
    print(f"{'─' * 70}")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")

    # --- Complexity comparison ---
    print(f"\n{'─' * 70}")
    print("COMPLEXITY COMPARISON")
    print(f"{'─' * 70}")

    full_ops = seq_len * seq_len
    window_ops = sum(min(window_size, i + 1) for i in range(seq_len))  # causal window

    print(f"\n  Attention entries computed:")
    print(f"    Full causal:    {full_ops:>6} entries  (n²/2 = {seq_len}² / 2)")
    print(f"    Sliding window: {window_ops:>6} entries  (n × w = {seq_len} × {window_size})")
    print(f"    Reduction:      {full_ops / window_ops:.1f}×")

    print(f"\n  For real-world scale (seq_len=32768, window=4096):")
    n, w = 32768, 4096
    full = n * n // 2
    windowed = n * w
    print(f"    Full causal:    {full:>12,} entries")
    print(f"    Sliding window: {windowed:>12,} entries")
    print(f"    Reduction:      {full / windowed:.1f}×")

    # --- Effective receptive field ---
    print(f"\n{'─' * 70}")
    print("EFFECTIVE RECEPTIVE FIELD")
    print(f"{'─' * 70}")
    print(f"""
    Layer 1: token at position i sees positions [i-{window_size-1}, i]
    Layer 2: those tokens had already seen their own windows
             → effective field: [i-{2*(window_size-1)}, i]
    Layer L: effective field = L × (w-1) = L × {window_size-1}

    ┌─────────┬───────────────────────────────┐
    │ Layers  │ Effective receptive field      │
    ├─────────┼───────────────────────────────┤
    │    1    │ {1*(window_size-1):>6} tokens                    │
    │    8    │ {8*(window_size-1):>6} tokens                    │
    │   32    │ {32*(window_size-1):>6} tokens                   │
    │   80    │ {80*(window_size-1):>6} tokens                   │
    └─────────┴───────────────────────────────┘

    Mistral 7B: 32 layers × (4096-1) = 131,040 token effective field.
    This is why SWA works despite the local window!
    """)

    # --- Rolling KV cache ---
    print("─" * 70)
    print("ROLLING KV CACHE (Mistral's optimization)")
    print("─" * 70)
    print("""
    With SWA, you only need to cache the last w positions!
    Old entries that fall outside the window can be overwritten.

    Standard KV cache:  grows with sequence length -> O(n)
    Rolling KV cache:   fixed size = w -> O(1) memory!

    At position t, cache contains: positions [t-w+1, t]
    When generating token t+1:
      - Compute K_(t+1), V_(t+1)
      - Overwrite cache[(t+1) mod w] with K_(t+1), V_(t+1)
      - Attend only to the w entries in the cache

    This means you can generate INFINITELY long sequences with FIXED memory!
    """)


if __name__ == "__main__":
    demo()
