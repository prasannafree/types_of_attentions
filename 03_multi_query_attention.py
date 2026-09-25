"""
================================================================================
03 — MULTI-QUERY ATTENTION (MQA)
================================================================================

PAPER: "Fast Transformer Decoding: One Write-Head is All You Need"
       (Noam Shazeer, 2019)

THE PROBLEM MQA SOLVES:
    During autoregressive inference (generating one token at a time), we must
    cache the K and V tensors for ALL previous positions ("KV cache").

    In standard MHA with h heads:
        KV cache per layer = 2 × seq_len × h × d_k

    For a large model (e.g., h=96, d_k=128, 96 layers, seq_len=8192):
        KV cache = 2 × 8192 × 96 × 128 × 96 layers × 2 bytes (fp16)
                 ≈ 36 GB per sequence!

    When serving many users simultaneously, this becomes the bottleneck.

THE MQA SOLUTION:
    Use h separate Query projections (one per head), but only ONE shared
    Key projection and ONE shared Value projection.

    MHA:  h query heads, h key heads, h value heads   → h × d_k params for K, V
    MQA:  h query heads, 1 key head,  1 value head    → 1 × d_k params for K, V

    The single K and V are broadcast across all query heads.

WHAT IT REDUCES:
    ┌────────────────────────┬───────────────┬───────────────┐
    │                        │  MHA          │  MQA          │
    ├────────────────────────┼───────────────┼───────────────┤
    │ KV cache per layer     │ 2·n·h·d_k     │ 2·n·d_k       │
    │ KV params (W_K + W_V)  │ 2·d·h·d_k     │ 2·d·d_k       │
    │ Reduction factor       │ 1×            │ h× smaller    │
    └────────────────────────┴───────────────┴───────────────┘

    For h=96 heads, the KV cache is 96× smaller!

QUALITY IMPACT:
    - Slight degradation in perplexity compared to full MHA.
    - Often negligible when the model is large enough.
    - The inference speedup (especially for long sequences, large batches)
      usually outweighs the small quality loss.
    - Used in: PaLM, Falcon, StarCoder, and many production models.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: all query heads share a single key head and value head.

    Key difference from MHA:
        MHA:  W_K is (d_model, d_model) → produces h independent K heads
        MQA:  W_K is (d_model, d_k)     → produces 1 K head, broadcast to all queries
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q: still has h heads (each head gets its own query projection)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # → (batch, seq, h*d_k)

        # K, V: only ONE head each (this is the core change!)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)  # → (batch, seq, d_k)
        self.W_V = nn.Linear(d_model, self.d_k, bias=False)  # → (batch, seq, d_k)

        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # ---- Project Q (h heads), K (1 head), V (1 head) ----
        Q = self.W_Q(x)  # (batch, seq, d_model) = (batch, seq, h*d_k)
        K = self.W_K(x)  # (batch, seq, d_k) — single head!
        V = self.W_V(x)  # (batch, seq, d_k) — single head!

        # ---- Reshape Q into h heads ----
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (batch, h, seq, d_k)

        # ---- K, V: add a "head" dimension of size 1, then broadcast ----
        # (batch, seq, d_k) → (batch, 1, seq, d_k)
        K = K.unsqueeze(1)  # Will be broadcast to (batch, h, seq, d_k) during matmul
        V = V.unsqueeze(1)

        # ---- Attention (same formula, but K and V are shared) ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, h, seq_q, seq_k) — broadcasting handles the h dimension

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # (batch, h, seq_q, seq_k) @ (batch, 1, seq_k, d_k) → (batch, h, seq_q, d_k)
        context = torch.matmul(weights, V)

        # ---- Concatenate and project ----
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)

        return output, weights


# ==============================================================================
# SIDE-BY-SIDE COMPARISON: MHA vs MQA
# ==============================================================================

def demo():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mqa(x, mask=mask)

    print("=" * 70)
    print("MULTI-QUERY ATTENTION — DEMO")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  d_model    = {d_model}")
    print(f"  num_heads  = {num_heads}")
    print(f"  d_k        = {d_model // num_heads}")

    print(f"\nShapes:")
    print(f"  Input:             {x.shape}")
    print(f"  Output:            {output.shape}")
    print(f"  Attention weights: {weights.shape}")

    # --- Parameter comparison ---
    d_k = d_model // num_heads
    mqa_params = sum(p.numel() for p in mqa.parameters())

    mha_kv_params = 2 * d_model * d_model      # W_K and W_V in MHA
    mqa_kv_params = 2 * d_model * d_k           # W_K and W_V in MQA

    print(f"\n{'─' * 70}")
    print("PARAMETER COMPARISON (MHA vs MQA)")
    print(f"{'─' * 70}")
    print(f"  W_Q:  MHA = {d_model*d_model:,}  |  MQA = {d_model*d_model:,}  (same)")
    print(f"  W_K:  MHA = {d_model*d_model:,}  |  MQA = {d_model*d_k:,}      ({num_heads}× smaller)")
    print(f"  W_V:  MHA = {d_model*d_model:,}  |  MQA = {d_model*d_k:,}      ({num_heads}× smaller)")
    print(f"  W_O:  MHA = {d_model*d_model:,}  |  MQA = {d_model*d_model:,}  (same)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total: MHA = {4*d_model**2:,}  |  MQA = {mqa_params:,}")

    # --- KV Cache comparison (the BIG win) ---
    print(f"\n{'─' * 70}")
    print("KV CACHE COMPARISON (the main reason MQA exists)")
    print(f"{'─' * 70}")
    mha_kv_cache = 2 * seq_len * d_model          # h heads
    mqa_kv_cache = 2 * seq_len * d_k              # 1 head
    print(f"  MHA KV cache per layer: 2 × {seq_len} × {d_model} = {mha_kv_cache:,} floats")
    print(f"  MQA KV cache per layer: 2 × {seq_len} × {d_k}  = {mqa_kv_cache:,} floats")
    print(f"  Reduction: {mha_kv_cache / mqa_kv_cache:.0f}× smaller!")

    print(f"\n  For a real model (d=4096, h=32, seq=8192, 32 layers):")
    real_mha = 2 * 8192 * 4096 * 32 * 2  # fp16
    real_mqa = 2 * 8192 * 128 * 32 * 2   # d_k=128
    print(f"    MHA KV cache: {real_mha / 1e9:.1f} GB")
    print(f"    MQA KV cache: {real_mqa / 1e9:.2f} GB")
    print(f"    Savings: {real_mha / real_mqa:.0f}× ← this is why serving teams love MQA")

    # --- Verify broadcasting works correctly ---
    print(f"\n{'─' * 70}")
    print("BROADCASTING VERIFICATION")
    print(f"{'─' * 70}")
    print(f"  Q shape after split: (batch={batch_size}, h={num_heads}, seq={seq_len}, d_k={d_k})")
    print(f"  K shape (1 head):    (batch={batch_size}, 1, seq={seq_len}, d_k={d_k})")
    print(f"  K is broadcast to match Q's head dimension automatically.")
    print(f"  Each query head sees the SAME keys and values.")
    print(f"  But each head's W_Q projects differently → different attention patterns.")


if __name__ == "__main__":
    demo()
