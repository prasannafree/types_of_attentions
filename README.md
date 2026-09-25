# Types of Attention Mechanisms

A hands-on learning repository implementing **8 attention mechanisms** from scratch in PyTorch. Each script is self-contained, heavily commented, and includes runnable demos that visualize what's happening.

## Quick Start

```bash
# Run any script directly
uv run python 01_scaled_dot_product_attention.py
uv run python 02_multi_head_self_attention.py
# ... etc
```

## The Scripts

| # | Script | Mechanism | Key Innovation |
|---|--------|-----------|----------------|
| 01 | `01_scaled_dot_product_attention.py` | **Scaled Dot-Product** | The atomic building block — Q·Kᵀ/√d_k → softmax → weighted V |
| 02 | `02_multi_head_self_attention.py` | **Multi-Head Attention (MHA)** | Split into h parallel heads for diverse attention patterns |
| 03 | `03_multi_query_attention.py` | **Multi-Query Attention (MQA)** | Share K,V across all heads → h× smaller KV cache |
| 04 | `04_grouped_query_attention.py` | **Grouped-Query Attention (GQA)** | Groups of query heads share KV → sweet spot between MHA & MQA |
| 05 | `05_multi_head_latent_attention.py` | **Multi-Head Latent Attention (MLA)** | Compress KV into low-rank latent → reconstruct on-the-fly |
| 06 | `06_sliding_window_attention.py` | **Sliding Window Attention (SWA)** | Each token attends to w neighbors only → near-linear cost |
| 07 | `07_sparse_hybrid_attention.py` | **Sparse/Hybrid (Longformer, BigBird)** | Local + global + random patterns → O(n) with global context |
| 08 | `08_linear_attention.py` | **Linear Attention (RetNet, RWKV, DeltaNet)** | Drop softmax → O(1) per-token inference with fixed-size state |

## Recommended Learning Path

### Phase 1: The Foundation (Start Here)
1. **Script 01** — Understand the core formula: Q, K, V, scaling, masking
2. **Script 02** — See how multiple heads split the same computation for richer representations

### Phase 2: The KV Cache Problem (Inference Optimization)
3. **Script 03** — MQA: the extreme solution (1 KV head)
4. **Script 04** — GQA: the practical compromise (used in LLaMA 3, Mistral)
5. **Script 05** — MLA: compression-based approach (used in DeepSeek)

### Phase 3: Breaking the Quadratic Wall
6. **Script 06** — SWA: local attention windows
7. **Script 07** — Sparse: combining local + global + random patterns
8. **Script 08** — Linear: the paradigm shift to O(1) per-token inference

## The Big Picture

```
                        ATTENTION MECHANISMS EVOLUTION
                        ═══════════════════════════════

    ┌─────────────────────────────────────────────────────────────────┐
    │                    Scaled Dot-Product (01)                      │
    │                    The Foundation — O(n²)                       │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────┐
    │                 Multi-Head Attention (02)                        │
    │              h parallel heads, same cost                        │
    └──────┬───────────────────┬──────────────────────┬───────────────┘
           │                   │                      │
    ┌──────▼──────┐  ┌────────▼────────┐  ┌──────────▼──────────┐
    │ KV Cache    │  │ Sequence Length  │  │  Paradigm Shift     │
    │ Reduction   │  │ Reduction       │  │  (Drop Softmax)     │
    ├─────────────┤  ├─────────────────┤  ├─────────────────────┤
    │ MQA (03)    │  │ SWA (06)        │  │ Linear Attn (08)    │
    │ GQA (04)    │  │ Sparse (07)     │  │ RetNet, RWKV        │
    │ MLA (05)    │  │ Longformer      │  │ Gated DeltaNet      │
    │             │  │ BigBird         │  │                     │
    └─────────────┘  └─────────────────┘  └─────────────────────┘
```

## What Each Mechanism Reduces

| Mechanism | Training Cost | Inference/Token | KV Cache | Quality |
|-----------|--------------|----------------|----------|---------|
| MHA | O(n²·d) | O(n·d) | O(n·h·d_k) | Baseline |
| MQA | O(n²·d) | O(n·d) | **O(n·d_k)** ← h× smaller | Slight loss |
| GQA | O(n²·d) | O(n·d) | **O(n·g·d_k)** | Near-MHA |
| MLA | O(n²·d) | O(n·d) | **O(n·d_c)** ← compressed | Near-MHA |
| SWA | **O(n·w·d)** | O(w·d) | O(w·d) fixed! | Local only |
| Sparse | **O(n·(w+g)·d)** | - | - | Near-full |
| Linear | **O(n·d²)** | **O(d²)** constant! | **O(d²)** fixed! | Lower |

## Key Concepts to Understand

### Why Scaling? (Script 01)
Without dividing by √d_k, large dimensions produce huge dot products → softmax saturates → vanishing gradients. Scaling keeps variance ≈ 1.

### Why Multiple Heads? (Script 02)
One head can only learn one type of relationship per position. Multiple heads learn syntax, semantics, coreference, positional patterns simultaneously — without adding parameters.

### The KV Cache Problem (Scripts 03-05)
During autoregressive generation, K and V for all past tokens must be stored. For large models (96 heads, 128 dims, 32K context), this can exceed 30GB per sequence. MQA/GQA/MLA reduce this dramatically.

### The Quadratic Wall (Scripts 06-08)
Standard attention computes an n×n matrix. For n=100K, that's 10B entries. Solutions:
- **Local windows**: only compute nearby entries
- **Sparse patterns**: compute strategic subsets
- **Linear attention**: avoid the n×n matrix entirely via algebraic tricks

### Parallel vs Recurrent Duality (Script 08)
Linear attention models are unique: they can run in parallel during training (like transformers) OR recurrently during inference (like RNNs). This gives the best of both worlds.

## Real-World Usage

| Model | Attention Type | Why |
|-------|---------------|-----|
| GPT-3/4 | MHA | Quality is paramount, inference cost managed via infrastructure |
| LLaMA 2 7B | MHA | Small model, cache manageable |
| LLaMA 2/3 70B | **GQA** | Large model needs cache reduction for serving |
| Mistral 7B | **GQA + SWA** | Combines both: small cache + long context |
| DeepSeek-V2/V3 | **MLA** | Extreme compression for efficient serving |
| PaLM 540B | **MQA** | Maximum cache reduction for production |
| Longformer | **Sparse** | Designed for long documents (16K+ tokens) |
| RWKV, Mamba | **Linear** | Streaming/edge deployment, infinite context |

## Papers to Read

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original transformer (MHA)
2. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA
3. [GQA: Training Generalized Multi-Query Attention](https://arxiv.org/abs/2305.13245) — GQA
4. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA
5. [Longformer](https://arxiv.org/abs/2004.05150) — Sliding window + global attention
6. [Big Bird](https://arxiv.org/abs/2007.14062) — Local + global + random
7. [Transformers are RNNs](https://arxiv.org/abs/2006.16236) — Linear attention
8. [RetNet: Retentive Network](https://arxiv.org/abs/2307.08621) — Retention with decay
9. [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — Delta rule + gating
10. [RWKV](https://arxiv.org/abs/2305.13048) — Parallelizable RNN with linear attention
