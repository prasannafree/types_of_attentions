"""
================================================================================
07 — SPARSE / HYBRID ATTENTION (Longformer & BigBird style)
================================================================================

PAPERS:
    - "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
    - "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)

THE PROBLEM:
    Sliding window attention (SWA) gives local context efficiently.
    But some tasks NEED global context:
        - Document classification needs to aggregate info from everywhere.
        - Question answering needs to connect the question with distant passages.

    Pure SWA can miss long-range dependencies that span more than L×w tokens.

THE SOLUTION: Combine multiple attention patterns!

LONGFORMER'S THREE PATTERNS:
    1. LOCAL (sliding window): Each token attends to w neighbors → O(n·w)
    2. GLOBAL: Selected special tokens (e.g., [CLS]) attend to ALL tokens → O(n·g)
    3. DILATED (optional): Attend to every k-th token for medium-range → O(n·w/k)

    ┌────────────────────────────────────────┐
    │ Full attention    Longformer hybrid     │
    │ ┌─────────────┐  ┌─────────────┐       │
    │ │■■■■■■■■■■■■│  │■■■■■■■■■■■■│ ← row 0 = global token │
    │ │■■■■■■■■■■■■│  │■■■·····■···│       │
    │ │■■■■■■■■■■■■│  │■·■■■······│       │
    │ │■■■■■■■■■■■■│  │■··■■■·····│       │
    │ │■■■■■■■■■■■■│  │■···■■■····│       │
    │ │■■■■■■■■■■■■│  │■····■■■···│       │
    │ │■■■■■■■■■■■■│  │■·····■■■··│       │
    │ │■■■■■■■■■■■■│  │■······■■■·│       │
    │ │■■■■■■■■■■■■│  │■·······■■■│       │
    │ └─────────────┘  └─────────────┘       │
    │  n² entries       ~n·(w+g) entries     │
    └────────────────────────────────────────┘

BIGBIRD'S THREE PATTERNS:
    1. LOCAL (sliding window): Same as Longformer
    2. GLOBAL: Same as Longformer
    3. RANDOM: Each token also attends to r random positions → adds diversity

    BigBird proved this combination is a UNIVERSAL APPROXIMATOR of
    full attention (Turing complete!).

WHAT IT REDUCES:
    ┌───────────────────┬──────────────┬───────────────────┐
    │                   │ Full Attn    │ Sparse/Hybrid      │
    ├───────────────────┼──────────────┼───────────────────┤
    │ Time complexity   │ O(n²·d)      │ O(n·(w+g+r)·d)    │
    │ Memory            │ O(n²)        │ O(n·(w+g+r))       │
    │ Global context?   │ Yes          │ Yes (via globals)   │
    │ Local context?    │ Yes          │ Yes (via window)    │
    │ Long-range?       │ Yes          │ Yes (global+random) │
    └───────────────────┴──────────────┴───────────────────┘

USE CASES:
    - Document classification with long documents
    - Long-range question answering
    - Summarization of long texts
    - Any task where n > 4096 and you need both local and global attention

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_longformer_mask(
    seq_len: int,
    window_size: int,
    global_token_indices: list[int],
) -> torch.Tensor:
    """
    Create Longformer-style attention mask combining:
      1. Sliding window (local attention)
      2. Global tokens (attend to/from everything)

    Args:
        seq_len:              sequence length
        window_size:          sliding window size
        global_token_indices: positions that get global attention (e.g., [0] for [CLS])

    Returns:
        mask: (seq_len, seq_len) boolean tensor
    """
    rows = torch.arange(seq_len).unsqueeze(1)
    cols = torch.arange(seq_len).unsqueeze(0)

    # 1. Local: sliding window
    local_mask = (rows - cols).abs() < window_size

    # 2. Global: selected tokens attend to everything and everything attends to them
    global_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for idx in global_token_indices:
        global_mask[idx, :] = True   # global token attends to all
        global_mask[:, idx] = True   # all tokens attend to global token

    # Combine
    mask = local_mask | global_mask

    return mask


def create_bigbird_mask(
    seq_len: int,
    window_size: int,
    global_token_indices: list[int],
    num_random: int,
) -> torch.Tensor:
    """
    Create BigBird-style mask: local + global + random.

    The random connections are key: they reduce the graph diameter,
    ensuring any token can reach any other in O(log n) hops.
    """
    # Start with Longformer mask (local + global)
    mask = create_longformer_mask(seq_len, window_size, global_token_indices)

    # 3. Random: each token attends to r random other tokens
    for i in range(seq_len):
        if i in global_token_indices:
            continue  # global tokens already attend to everything
        random_indices = torch.randperm(seq_len)[:num_random]
        mask[i, random_indices] = True
        mask[random_indices, i] = True  # make it symmetric for stability

    return mask


class SparseHybridAttention(nn.Module):
    """
    Hybrid attention supporting Longformer and BigBird patterns.

    NOTE: This is an EDUCATIONAL implementation using masking. Production
    implementations (like the actual Longformer code) use custom CUDA kernels
    to avoid materializing the full n×n attention matrix.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 3,
        global_token_indices: list[int] | None = None,
        num_random: int = 0,
        mode: str = "longformer",  # "longformer" or "bigbird"
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert mode in ("longformer", "bigbird")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.global_token_indices = global_token_indices or [0]
        self.num_random = num_random
        self.mode = mode

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _create_mask(self, seq_len: int) -> torch.Tensor:
        if self.mode == "longformer":
            mask = create_longformer_mask(
                seq_len, self.window_size, self.global_token_indices
            )
        else:  # bigbird
            mask = create_bigbird_mask(
                seq_len, self.window_size, self.global_token_indices, self.num_random
            )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        mask = self._create_mask(seq_len).to(x.device)

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
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

    seq_len = 16
    d_model = 32
    num_heads = 4
    window_size = 3

    x = torch.randn(1, seq_len, d_model)

    print("=" * 70)
    print("SPARSE / HYBRID ATTENTION — DEMO")
    print("=" * 70)

    # --- Visualize different mask patterns ---
    def print_mask(mask_2d, name):
        print(f"\n{name}:")
        n = mask_2d.shape[0]
        header = "     " + " ".join(f"{j:>2}" for j in range(n))
        print(header)
        for i in range(n):
            row = f"  {i:>2} "
            for j in range(n):
                row += " ■ " if mask_2d[i, j] else " · "
            # Count how many this row attends to
            count = mask_2d[i].sum().item()
            row += f"  ({int(count)})"
            print(row)
        total = mask_2d.sum().item()
        full = n * n
        print(f"  Total entries: {int(total)}/{full} ({100*total/full:.1f}%)")

    # Full attention
    full = torch.ones(seq_len, seq_len, dtype=torch.bool)
    print_mask(full, "Full Attention")

    # Local only (sliding window)
    rows = torch.arange(seq_len).unsqueeze(1)
    cols = torch.arange(seq_len).unsqueeze(0)
    local = (rows - cols).abs() < window_size
    print_mask(local, f"Local Only (window={window_size})")

    # Longformer (local + global)
    longformer = create_longformer_mask(seq_len, window_size, [0, seq_len - 1])
    print_mask(longformer, "Longformer (local + global at positions [0, last])")

    # BigBird (local + global + random)
    bigbird = create_bigbird_mask(seq_len, window_size, [0], num_random=2)
    print_mask(bigbird, "BigBird (local + global at [0] + 2 random)")

    # --- Run models ---
    print(f"\n{'─' * 70}")
    print("MODEL COMPARISON")
    print(f"{'─' * 70}")

    configs = [
        ("Longformer", "longformer", [0, seq_len - 1], 0),
        ("BigBird",    "bigbird",    [0],              3),
    ]

    for name, mode, globals_idx, n_random in configs:
        model = SparseHybridAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            global_token_indices=globals_idx,
            num_random=n_random,
            mode=mode,
        )
        output, weights = model(x)
        params = sum(p.numel() for p in model.parameters())

        # Count non-zero attention entries
        mask = model._create_mask(seq_len)
        density = mask.float().mean().item()

        print(f"\n  {name}:")
        print(f"    Output shape: {output.shape}")
        print(f"    Parameters:   {params:,}")
        print(f"    Mask density: {density:.1%} (vs 100% for full attention)")

    # --- Scaling analysis ---
    print(f"\n{'─' * 70}")
    print("SCALING: WHY SPARSE ATTENTION MATTERS AT SCALE")
    print(f"{'─' * 70}")
    print(f"""
    ┌────────────┬──────────────────┬───────────────────────────┐
    │ seq_len    │ Full Attention   │ Sparse (w=512, g=2, r=3) │
    ├────────────┼──────────────────┼───────────────────────────┤""")
    for n in [512, 1024, 4096, 16384, 65536, 131072]:
        w, g, r = 512, 2, 3
        full_entries = n * n
        sparse_entries = n * (2 * w + g + r)  # approximate
        ratio = full_entries / sparse_entries
        print(f"    │ {n:>10,} │ {full_entries:>16,} │ {sparse_entries:>16,}  ({ratio:>5.1f}×) │")
    print(f"    └────────────┴──────────────────┴───────────────────────────┘")

    print(f"""
    Key insight: as sequence length grows, the savings INCREASE.
    At n=131K, sparse attention is ~128× cheaper than full attention!

    This is why Longformer/BigBird can handle 16K-64K+ tokens while
    standard transformers struggle beyond 2K-4K tokens.
    """)

    # --- When to use what ---
    print(f"{'─' * 70}")
    print("WHEN TO USE EACH PATTERN")
    print(f"{'─' * 70}")
    print(f"""
    LOCAL (sliding window):
      ✓ Syntactic parsing, NER, POS tagging
      ✓ When local context dominates
      ✗ Misses long-range dependencies

    GLOBAL tokens:
      ✓ Classification ([CLS] token aggregates everything)
      ✓ Question answering (question tokens need global view)
      ✓ Summarization (summary tokens need full document access)

    RANDOM connections (BigBird's addition):
      ✓ Graph theory: random edges reduce diameter → faster info propagation
      ✓ Acts like "skip connections" in the attention graph
      ✓ Provably makes the architecture Turing complete
    """)


if __name__ == "__main__":
    demo()
