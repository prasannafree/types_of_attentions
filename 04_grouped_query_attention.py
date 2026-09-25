"""
================================================================================
04 — GROUPED-QUERY ATTENTION (GQA)
================================================================================

PAPER: "GQA: Training Generalized Multi-Query Attention from Multi-Head Checkpoints"
       (Ainslie et al., 2023 — Google Research)

THE SPECTRUM:
    MHA, GQA, and MQA exist on a continuum:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  MHA ◄──────────── GQA ──────────────► MQA                          │
    │  (h KV heads)   (g KV heads, 1<g<h)   (1 KV head)                  │
    │                                                                      │
    │  Best quality    ◄── trade-off ──►    Best inference speed           │
    │  Worst cache                          Worst quality (slightly)       │
    └──────────────────────────────────────────────────────────────────────┘

    GQA is the Goldilocks zone:
        - MQA can degrade quality too much on some tasks.
        - MHA's KV cache is too expensive for production serving.
        - GQA groups query heads: every g_size = h/num_kv_heads query heads
          share one KV head.

EXAMPLE (h=8, num_kv_heads=2):
    Query heads:  [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
    KV heads:     [KV0,          , KV1,          ]
    Grouping:      Q0,Q1,Q2,Q3 → KV0    Q4,Q5,Q6,Q7 → KV1

    Each group of 4 query heads shares one KV head.

WHAT IT REDUCES:
    ┌──────────────────┬────────────────┬────────────────┬────────────────┐
    │                  │  MHA (h=8)     │  GQA (g=2)     │  MQA (g=1)     │
    ├──────────────────┼────────────────┼────────────────┼────────────────┤
    │ Query heads      │  8             │  8             │  8             │
    │ KV heads         │  8             │  2             │  1             │
    │ KV cache size    │  8 × d_k       │  2 × d_k       │  1 × d_k       │
    │ KV params        │  2 × d × 8d_k  │  2 × d × 2d_k  │  2 × d × d_k   │
    │ Quality          │  Best          │  Near-MHA      │  Slight loss   │
    └──────────────────┴────────────────┴────────────────┴────────────────┘

WHO USES IT:
    LLaMA 2 70B, LLaMA 3, Mistral 7B, Mixtral, Gemma, CodeLlama, etc.
    It has become the de facto standard for production LLMs.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention: groups of query heads share KV heads.

    Args:
        d_model:      model dimension
        num_heads:    number of query heads (h)
        num_kv_heads: number of key-value heads (g), must divide num_heads

    Special cases:
        num_kv_heads == num_heads → standard MHA
        num_kv_heads == 1         → MQA
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_heads  # how many Q heads per KV head

        # Query: full h heads
        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)

        # Key, Value: only num_kv_heads heads
        self.W_K = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)

        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads.

        (batch, num_kv_heads, seq, d_k) → (batch, num_heads, seq, d_k)

        This is the key operation: each KV head is duplicated group_size times
        so it can be paired with its group of query heads.

        Example (num_heads=8, num_kv_heads=2, group_size=4):
            [KV0, KV1] → [KV0, KV0, KV0, KV0, KV1, KV1, KV1, KV1]
        """
        if self.group_size == 1:
            return x  # MHA case, no repetition needed

        batch, num_kv_heads, seq_len, d_k = x.shape
        # Insert a new dimension and expand (no memory copy with expand!)
        x = x.unsqueeze(2)                     # (batch, g, 1, seq, d_k)
        x = x.expand(-1, -1, self.group_size, -1, -1)  # (batch, g, group_size, seq, d_k)
        x = x.reshape(batch, self.num_heads, seq_len, d_k)  # (batch, h, seq, d_k)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # ---- Project ----
        Q = self.W_Q(x)  # (batch, seq, h * d_k)
        K = self.W_K(x)  # (batch, seq, g * d_k)
        V = self.W_V(x)  # (batch, seq, g * d_k)

        # ---- Reshape into heads ----
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # ---- Repeat KV heads to match Q heads ----
        K = self._repeat_kv(K)  # (batch, h, seq, d_k)
        V = self._repeat_kv(V)

        # ---- Standard attention ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)

        # ---- Concat and project ----
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)

        return output, weights


# ==============================================================================
# DEMO: Compare MHA vs GQA vs MQA
# ==============================================================================

def demo():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8
    d_k = d_model // num_heads

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    print("=" * 70)
    print("GROUPED-QUERY ATTENTION — DEMO")
    print("=" * 70)

    configs = [
        ("MHA (8 KV heads)", 8),
        ("GQA (4 KV heads)", 4),
        ("GQA (2 KV heads)", 2),
        ("MQA (1 KV head)",  1),
    ]

    results = []
    for name, num_kv_heads in configs:
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        output, weights = gqa(x, mask=mask)
        total_params = sum(p.numel() for p in gqa.parameters())
        kv_cache = 2 * seq_len * num_kv_heads * d_k
        results.append((name, num_kv_heads, total_params, kv_cache, output.shape))

    print(f"\n{'Variant':<22} {'KV heads':>10} {'Params':>10} {'KV Cache':>12} {'Output':>20}")
    print("─" * 76)
    for name, kv_h, params, cache, shape in results:
        print(f"  {name:<20} {kv_h:>10} {params:>10,} {cache:>12,} {str(shape):>20}")

    # --- Detailed breakdown ---
    print(f"\n{'─' * 70}")
    print("HOW _repeat_kv WORKS (the key trick)")
    print(f"{'─' * 70}")
    print(f"""
    For GQA with 8 query heads and 2 KV heads:

    Query heads:  [Q₀] [Q₁] [Q₂] [Q₃] | [Q₄] [Q₅] [Q₆] [Q₇]
                   ↕    ↕    ↕    ↕       ↕    ↕    ↕    ↕
    KV heads:     [──── KV₀ ────────]   [──── KV₁ ────────]

    _repeat_kv expands KV₀ → [KV₀, KV₀, KV₀, KV₀] and
                       KV₁ → [KV₁, KV₁, KV₁, KV₁]

    So each group of 4 query heads attends using the same keys and values.
    But each query head still has its OWN projection (W_Q), so they compute
    DIFFERENT attention patterns even though they share the same KV head.
    """)

    # --- Real-world model configs ---
    print(f"{'─' * 70}")
    print("REAL-WORLD MODEL CONFIGURATIONS")
    print(f"{'─' * 70}")
    print(f"""
    ┌──────────────┬──────────┬───────────┬──────────┬──────────────────┐
    │ Model        │ d_model  │ Q heads   │ KV heads │ Type             │
    ├──────────────┼──────────┼───────────┼──────────┼──────────────────┤
    │ GPT-3 175B   │ 12288    │ 96        │ 96       │ MHA              │
    │ PaLM 540B    │ 18432    │ 48        │ 1        │ MQA              │
    │ LLaMA 2 7B   │ 4096     │ 32        │ 32       │ MHA              │
    │ LLaMA 2 70B  │ 8192     │ 64        │ 8        │ GQA (8 groups)   │
    │ LLaMA 3 8B   │ 4096     │ 32        │ 8        │ GQA (4 groups)   │
    │ Mistral 7B   │ 4096     │ 32        │ 8        │ GQA (4 groups)   │
    │ Gemma 7B     │ 3072     │ 16        │ 16       │ MHA              │
    └──────────────┴──────────┴───────────┴──────────┴──────────────────┘
    """)


if __name__ == "__main__":
    demo()
