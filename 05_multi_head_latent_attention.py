"""
================================================================================
05 — MULTI-HEAD LATENT ATTENTION (MLA)
================================================================================

PAPER: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
        Language Model" (DeepSeek-AI, 2024)

THE PROBLEM:
    GQA reduces KV cache by sharing KV heads, but it still stores per-head
    key and value vectors. For extremely long contexts or high throughput,
    even GQA's cache can be too large.

THE MLA IDEA:
    Instead of caching K and V directly, compress them into a LOW-RANK
    LATENT representation, and reconstruct K and V from it on-the-fly.

    Standard approach (MHA/GQA):
        Input → W_K → K (cache this)
        Input → W_V → V (cache this)

    MLA approach:
        Input → W_DKV (down-project) → c_kv (cache THIS — much smaller!)
        At attention time:
            c_kv → W_UK → K (reconstruct)
            c_kv → W_UV → V (reconstruct)

    The compressed latent c_kv has dimension d_c << d_model, so the cache
    is drastically smaller.

WHAT IT REDUCES:
    ┌────────────────────────┬───────────────────────┬──────────────────────┐
    │                        │ GQA (g KV heads)      │ MLA                  │
    ├────────────────────────┼───────────────────────┼──────────────────────┤
    │ KV cache per token     │ 2 × g × d_k           │ d_c (single vector!) │
    │ With d_c << 2·g·d_k   │                       │ Much smaller          │
    └────────────────────────┴───────────────────────┴──────────────────────┘

    DeepSeek-V2 uses d_c = 512 while standard would need 2 × 128 × 128 = 32768.
    That's a 64× reduction!

THE CATCH:
    Reconstructing K and V from the latent adds computation. But computation is
    cheap relative to memory bandwidth during inference. The bottleneck for LLM
    serving is usually memory bandwidth (loading KV cache from HBM), not FLOPs.

ALSO: Decoupled Rotary Position Embedding (RoPE)
    MLA has a subtlety with positional encodings. Since K is reconstructed
    from a compressed latent, you can't apply RoPE to K directly (it would
    need to be absorbed into the latent, breaking the compression). DeepSeek
    solves this by adding a SEPARATE small RoPE key that's computed and cached
    independently.

    For simplicity, this implementation shows the core compression mechanism
    without RoPE. See the comments for where RoPE would be added.

WHO USES IT:
    DeepSeek-V2, DeepSeek-V3, DeepSeek-R1

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2.

    Key idea: compress KV into a low-rank latent vector for caching.

    The query side can also be compressed (down-project then up-project),
    but we focus on the KV compression which is the main innovation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_compress: int,    # dimension of the compressed latent (d_c)
        d_rope: int = 64,   # dimension for decoupled RoPE keys (simplified here)
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_compress = d_compress
        self.d_rope = d_rope

        # ---- Query projection (can also be compressed; simplified here) ----
        self.W_Q = nn.Linear(d_model, d_model, bias=False)

        # ---- KV compression (the core innovation) ----
        # Down-project: input → compressed latent c_kv
        self.W_DKV = nn.Linear(d_model, d_compress, bias=False)

        # Up-project: c_kv → full K and V for all heads
        self.W_UK = nn.Linear(d_compress, d_model, bias=False)  # reconstruct K
        self.W_UV = nn.Linear(d_compress, d_model, bias=False)  # reconstruct V

        # ---- Decoupled RoPE key (separate small key for positional info) ----
        # In the full implementation, this would have RoPE applied to it.
        # Here we just show that it exists as a separate cached component.
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)

        # ---- Output projection ----
        # We need to account for the extra RoPE key dimension
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # ---- Query (standard) ----
        Q = self.W_Q(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (batch, h, seq, d_k)

        # ---- KV Compression (the magic) ----
        # Step 1: Compress input to low-rank latent
        c_kv = self.W_DKV(x)  # (batch, seq, d_compress) ← THIS IS WHAT WE CACHE!

        # Step 2: Reconstruct full K and V from the compressed latent
        K = self.W_UK(c_kv)   # (batch, seq, d_model) ← reconstructed on-the-fly
        V = self.W_UV(c_kv)   # (batch, seq, d_model)

        # Reshape into heads
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # ---- Decoupled RoPE key (simplified — no actual rotation applied) ----
        # In the real implementation:
        #   k_rope = self.W_KR(x)  → apply RoPE → cache separately
        #   K_full = concat(K_from_latent, k_rope) along head dim
        #   Q_full = concat(Q_content, q_rope) along head dim
        # We skip this for clarity but note it exists.

        # ---- Standard attention ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)

        # ---- Output ----
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)

        return output, weights

    def get_cache_size(self, seq_len: int) -> dict:
        """Show what needs to be cached during inference."""
        return {
            "compressed_latent_c_kv": (seq_len, self.d_compress),
            "rope_key_k_r": (seq_len, self.d_rope),
            "total_floats": seq_len * (self.d_compress + self.d_rope),
        }


# ==============================================================================
# DEMO
# ==============================================================================

def demo():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8
    d_k = d_model // num_heads
    d_compress = 16  # much smaller than d_model!

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    mla = MultiHeadLatentAttention(
        d_model=d_model,
        num_heads=num_heads,
        d_compress=d_compress,
    )
    output, weights = mla(x, mask=mask)

    print("=" * 70)
    print("MULTI-HEAD LATENT ATTENTION (MLA) — DEMO")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  d_model     = {d_model}")
    print(f"  num_heads   = {num_heads}")
    print(f"  d_k         = {d_k}")
    print(f"  d_compress  = {d_compress}  ← the compressed latent dimension")

    print(f"\nShapes:")
    print(f"  Input:             {x.shape}")
    print(f"  Output:            {output.shape}")
    print(f"  Attention weights: {weights.shape}")

    # --- Cache comparison ---
    print(f"\n{'─' * 70}")
    print("KV CACHE COMPARISON: WHY MLA IS SO EFFICIENT")
    print(f"{'─' * 70}")

    mha_cache = 2 * seq_len * num_heads * d_k  # full MHA
    gqa_cache = 2 * seq_len * 2 * d_k          # GQA with 2 KV heads
    mqa_cache = 2 * seq_len * d_k              # MQA
    mla_cache_info = mla.get_cache_size(seq_len)
    mla_cache = mla_cache_info["total_floats"]

    print(f"  MHA  (8 KV heads):  {mha_cache:>6,} floats  (2 × {seq_len} × {num_heads} × {d_k})")
    print(f"  GQA  (2 KV heads):  {gqa_cache:>6,} floats  (2 × {seq_len} × 2 × {d_k})")
    print(f"  MQA  (1 KV head):   {mqa_cache:>6,} floats  (2 × {seq_len} × {d_k})")
    print(f"  MLA  (compressed):  {mla_cache:>6,} floats  ({seq_len} × ({d_compress} + 64))")
    print(f"\n  MLA caches only c_kv ({d_compress}-dim) + rope_key (64-dim) per position.")
    print(f"  K and V are RECONSTRUCTED from c_kv on-the-fly during attention.")

    # --- The compression pipeline ---
    print(f"\n{'─' * 70}")
    print("THE COMPRESSION PIPELINE")
    print(f"{'─' * 70}")
    print(f"""
    ┌─────────────┐     W_DKV          ┌──────────────┐
    │  Input x    │ ──────────────────► │  c_kv        │  ← CACHE THIS
    │  ({d_model}-dim)    │   ({d_model}→{d_compress})         │  ({d_compress}-dim)     │     (tiny!)
    └─────────────┘                     └──────┬───────┘
                                               │
                              ┌────────────────┼────────────────┐
                              │ W_UK           │                │ W_UV
                              │ ({d_compress}→{d_model})         │                │ ({d_compress}→{d_model})
                              ▼                                 ▼
                        ┌──────────┐                      ┌──────────┐
                        │  K       │                      │  V       │
                        │  ({d_model}-dim) │                      │  ({d_model}-dim) │
                        └──────────┘                      └──────────┘
                              │                                 │
                              ▼                                 ▼
                        [Standard Scaled Dot-Product Attention]

    The reconstruction (W_UK, W_UV) adds FLOPs but saves memory bandwidth.
    During inference, memory bandwidth is the bottleneck, not compute.
    """)

    # --- Parameter count ---
    total = sum(p.numel() for p in mla.parameters())
    print(f"{'─' * 70}")
    print(f"Parameter count: {total:,}")
    print(f"  W_Q:   {d_model} × {d_model} = {d_model**2:,}")
    print(f"  W_DKV: {d_model} × {d_compress} = {d_model*d_compress:,}")
    print(f"  W_UK:  {d_compress} × {d_model} = {d_compress*d_model:,}")
    print(f"  W_UV:  {d_compress} × {d_model} = {d_compress*d_model:,}")
    print(f"  W_KR:  {d_model} × 64 = {d_model*64:,}")
    print(f"  W_O:   {d_model} × {d_model} = {d_model**2:,}")
    print(f"\n  Note: More parameters than MQA/GQA because of the up-projection")
    print(f"  matrices (W_UK, W_UV). But the CACHE is much smaller.")
    print(f"  Trade-off: more compute (cheap) for less memory (expensive).")


if __name__ == "__main__":
    demo()
