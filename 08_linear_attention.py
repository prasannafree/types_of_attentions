"""
================================================================================
08 — LINEAR ATTENTION (RetNet, RWKV, Gated DeltaNet)
================================================================================

THE FUNDAMENTAL SHIFT:
    All previous attention mechanisms compute:
        Attention(Q, K, V) = softmax(QKᵀ) · V     → O(n²) because of QKᵀ

    Linear attention asks: "What if we DROP the softmax?"

    Without softmax, we can use the associativity of matrix multiplication:
        (Q · Kᵀ) · V = Q · (Kᵀ · V)

    The left form is O(n²·d) — compute the n×n attention matrix first.
    The right form is O(n·d²) — compute a d×d matrix (Kᵀ·V) first!

    When d << n (which is common), this is a MASSIVE speedup.
    And crucially: the right form can be computed RECURRENTLY!

WHY THIS MATTERS:
    ┌───────────────────┬───────────────┬───────────────┬──────────────────┐
    │                   │ Softmax Attn  │ Linear Attn   │ Impact           │
    ├───────────────────┼───────────────┼───────────────┼──────────────────┤
    │ Training          │ O(n²·d)       │ O(n·d²)       │ Better for n>>d  │
    │ Inference (1 tok) │ O(n·d)        │ O(d²)         │ Constant time!   │
    │ Memory (KV cache) │ O(n·d)        │ O(d²)         │ Fixed size!      │
    │ Can be recurrent? │ No            │ Yes           │ RNN-like decode  │
    └───────────────────┴───────────────┴───────────────┴──────────────────┘

    Linear attention has O(1) per-token inference cost (no growing KV cache!).
    The "state" is a fixed d×d matrix that gets updated with each new token.

THE THREE VARIANTS WE IMPLEMENT:

1. VANILLA LINEAR ATTENTION (Katharopoulos et al., 2020)
   Replace softmax with a feature map φ: Attention = φ(Q)·(φ(K)ᵀ·V)
   Problem: can be unstable; no "forgetting" mechanism.

2. RetNet-STYLE (Sun et al., 2023 — "Retentive Network")
   Adds exponential decay: recent tokens matter more than distant ones.
   Retention(Q,K,V) = (Q·Kᵀ ⊙ D) · V  where D is a decay matrix.
   Can switch between parallel (training) and recurrent (inference) modes.

3. GATED DELTANET-STYLE (Yang et al., 2024)
   Uses a gating mechanism and "delta rule" for state updates.
   The gate controls what to remember/forget, similar to LSTM gates.
   State update: S_t = (1 - β_t·k_t·k_tᵀ)·S_{t-1} + β_t·v_t·k_tᵀ
   This is like "erase old info about k_t, then write new v_t for k_t."

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. VANILLA LINEAR ATTENTION
# ==============================================================================

class LinearAttention(nn.Module):
    """
    Vanilla linear attention with feature map φ.

    Key insight: by applying a non-negative feature map φ to Q and K
    (instead of softmax to QKᵀ), we can rearrange the computation:

        Standard:  softmax(Q·Kᵀ/√d) · V     — must compute n×n matrix
        Linear:    φ(Q) · (φ(K)ᵀ · V)        — compute d×d matrix instead

    Common feature maps:
        - ELU + 1: φ(x) = elu(x) + 1  (ensures non-negativity)
        - ReLU: φ(x) = relu(x)
        - 1 + x/√d: first-order Taylor of exp  (used here)

    Limitations:
        - No sharp attention (softmax creates peaky distributions; linear is smoother)
        - Can be numerically unstable (denominator can be very small)
        - No built-in forgetting (state accumulates everything)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ELU + 1 feature map: ensures non-negative outputs."""
        return F.elu(x) + 1

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel mode (for training): processes all positions at once.
        Uses the trick: φ(Q) · (φ(K)ᵀ · V)
        """
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q = self._feature_map(Q)
        K = self._feature_map(K)

        # The key trick: compute Kᵀ·V first → (batch, h, d_k, d_k)
        # This is O(n·d²) instead of O(n²·d)
        KV = torch.matmul(K.transpose(-2, -1), V)  # (batch, h, d_k, d_k)

        # Then Q · (KᵀV) → (batch, h, seq, d_k)
        output = torch.matmul(Q, KV)

        # Normalize (equivalent to dividing by sum of attention weights)
        Z = torch.matmul(Q, K.transpose(-2, -1).sum(dim=-1, keepdim=True))
        output = output / (Z + 1e-6)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(output)

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recurrent mode (for inference): processes one token at a time.
        Maintains a running state S = Σ φ(k_i) · v_iᵀ
        """
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.d_k)
        K = self.W_K(x).view(batch, seq_len, self.num_heads, self.d_k)
        V = self.W_V(x).view(batch, seq_len, self.num_heads, self.d_k)

        Q = self._feature_map(Q)
        K = self._feature_map(K)

        # State: (batch, h, d_k, d_k) — fixed size regardless of sequence length!
        S = torch.zeros(batch, self.num_heads, self.d_k, self.d_k, device=x.device)
        z = torch.zeros(batch, self.num_heads, self.d_k, 1, device=x.device)

        outputs = []
        for t in range(seq_len):
            q_t = Q[:, t]  # (batch, h, d_k)
            k_t = K[:, t]
            v_t = V[:, t]

            # Update state: S += k_t · v_tᵀ
            S = S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            z = z + k_t.unsqueeze(-1)

            # Query the state: output = q_t · S / (q_t · z)
            out_t = torch.einsum('bhd,bhde->bhe', q_t, S)
            norm_t = torch.einsum('bhd,bhd->bh', q_t, z.squeeze(-1)).unsqueeze(-1)
            out_t = out_t / (norm_t + 1e-6)

            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # (batch, seq, h, d_k)
        output = output.contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(output)

    def forward(self, x: torch.Tensor, mode: str = "parallel") -> torch.Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        else:
            return self.forward_recurrent(x)


# ==============================================================================
# 2. RetNet-STYLE RETENTION
# ==============================================================================

class RetNetRetention(nn.Module):
    """
    Retentive Network (RetNet) — linear attention with exponential decay.

    The decay matrix D makes attention "forget" distant tokens:
        D[i,j] = γ^(i-j)  for i >= j, 0 otherwise

    Where γ (gamma) is a decay factor (e.g., 0.95 or 0.99).

    This gives the best of both worlds:
        - RNN-like efficiency during inference (recurrent mode)
        - Transformer-like parallelism during training (parallel mode)
    """

    def __init__(self, d_model: int, num_heads: int, gamma: float = 0.95):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.gamma = gamma

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Group normalization for stability
        self.group_norm = nn.GroupNorm(num_heads, d_model)

    def _create_decay_matrix(self, seq_len: int) -> torch.Tensor:
        """
        Create the decay matrix D where D[i,j] = γ^(i-j) for causal positions.

        Example (gamma=0.9, seq_len=4):
            [[1.000, 0.000, 0.000, 0.000],
             [0.900, 1.000, 0.000, 0.000],
             [0.810, 0.900, 1.000, 0.000],
             [0.729, 0.810, 0.900, 1.000]]

        Nearby tokens have weight ~1, distant tokens decay toward 0.
        """
        pos = torch.arange(seq_len).float()
        # D[i,j] = gamma^(i-j) when i >= j, else 0
        decay = self.gamma ** (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        return decay * causal_mask

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel mode for training."""
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute Q·Kᵀ and apply decay (element-wise multiply)
        D = self._create_decay_matrix(seq_len).to(x.device)  # (seq, seq)
        retention = torch.matmul(Q, K.transpose(-2, -1)) * D.unsqueeze(0).unsqueeze(0)

        output = torch.matmul(retention, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        # GroupNorm expects (batch, channels, ...) so transpose
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)
        return self.W_O(output)

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recurrent mode for inference — O(1) per token!

        State update: S_t = γ · S_{t-1} + k_t · v_tᵀ
        Output:       o_t = q_t · S_t
        """
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.d_k)
        K = self.W_K(x).view(batch, seq_len, self.num_heads, self.d_k)
        V = self.W_V(x).view(batch, seq_len, self.num_heads, self.d_k)

        S = torch.zeros(batch, self.num_heads, self.d_k, self.d_k, device=x.device)
        outputs = []

        for t in range(seq_len):
            q_t = Q[:, t]
            k_t = K[:, t]
            v_t = V[:, t]

            # Decay old state, add new information
            S = self.gamma * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)

            # Query the state
            out_t = torch.einsum('bhd,bhde->bhe', q_t, S)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)
        output = output.contiguous().view(batch, seq_len, self.d_model)
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)
        return self.W_O(output)

    def forward(self, x: torch.Tensor, mode: str = "parallel") -> torch.Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        else:
            return self.forward_recurrent(x)


# ==============================================================================
# 3. GATED DELTANET-STYLE ATTENTION
# ==============================================================================

class GatedDeltaNetAttention(nn.Module):
    """
    Simplified Gated DeltaNet — linear attention with gated state updates.

    Key idea: Instead of just adding to the state, use a GATE to control
    how much to remember vs forget (similar to LSTM/GRU gates).

    The "delta rule" update:
        S_t = S_{t-1} - β_t · (S_{t-1} · k_t) · k_tᵀ + β_t · v_t · k_tᵀ

    In words:
        1. Remove what the state currently associates with key k_t
        2. Write the new value v_t for key k_t
        3. β_t controls how aggressively to update (learned gate)

    This is inspired by Hopfield networks and associative memory:
    the state S is a key-value memory that can be selectively updated.

    Advantage over RetNet: more precise control over what to remember/forget.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Gate: controls how much to update the state
        self.W_beta = nn.Linear(d_model, num_heads, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recurrent forward pass (the natural mode for DeltaNet).

        For each token:
            1. Compute gate β_t = sigmoid(W_β · x_t) ∈ [0, 1]
            2. Erase: remove old association for k_t
            3. Write: store new v_t → k_t association
            4. Read: query state with q_t
        """
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.d_k)
        K = self.W_K(x).view(batch, seq_len, self.num_heads, self.d_k)
        V = self.W_V(x).view(batch, seq_len, self.num_heads, self.d_k)

        # Normalize keys (important for stability of delta rule)
        K = F.normalize(K, dim=-1)

        # Compute gates
        beta = torch.sigmoid(self.W_beta(x))  # (batch, seq, h)

        # State: (batch, h, d_k, d_k)
        S = torch.zeros(batch, self.num_heads, self.d_k, self.d_k, device=x.device)
        outputs = []

        for t in range(seq_len):
            q_t = Q[:, t]                   # (batch, h, d_k)
            k_t = K[:, t]                   # (batch, h, d_k)
            v_t = V[:, t]                   # (batch, h, d_k)
            beta_t = beta[:, t].unsqueeze(-1)  # (batch, h, 1)

            # ---- Delta rule update ----
            # Step 1: What does the state currently recall for k_t?
            old_value = torch.einsum('bhde,bhd->bhe', S, k_t)  # (batch, h, d_k)

            # Step 2: Compute the "delta" (error) — difference between new and old
            delta = v_t - old_value  # (batch, h, d_k)

            # Step 3: Update state: S += β_t · delta · k_tᵀ
            # This erases old info and writes new info, controlled by gate β_t
            S = S + beta_t.unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', k_t, delta)

            # Step 4: Read from state
            out_t = torch.einsum('bhd,bhde->bhe', q_t, S)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)
        output = output.contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(output)


# ==============================================================================
# DEMO
# ==============================================================================

def demo():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 16
    d_model = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)

    print("=" * 70)
    print("LINEAR ATTENTION — DEMO")
    print("=" * 70)

    # --- 1. Vanilla Linear Attention ---
    print(f"\n{'━' * 70}")
    print("1. VANILLA LINEAR ATTENTION")
    print(f"{'━' * 70}")

    linear = LinearAttention(d_model=d_model, num_heads=num_heads)

    out_par = linear(x, mode="parallel")
    out_rec = linear(x, mode="recurrent")

    print(f"  Parallel output shape: {out_par.shape}")
    print(f"  Recurrent output shape: {out_rec.shape}")

    # Check equivalence
    diff = (out_par - out_rec).abs().max().item()
    print(f"  Max difference (parallel vs recurrent): {diff:.6f}")
    print(f"  (Should be ~0 — they compute the same thing differently!)")
    print(f"\n  State size: ({d_model // num_heads} × {d_model // num_heads}) × {num_heads} heads")
    print(f"  = {(d_model // num_heads)**2 * num_heads} floats (FIXED, regardless of seq_len!)")

    # --- 2. RetNet Retention ---
    print(f"\n{'━' * 70}")
    print("2. RetNet RETENTION (with exponential decay)")
    print(f"{'━' * 70}")

    retnet = RetNetRetention(d_model=d_model, num_heads=num_heads, gamma=0.9)

    out_par_ret = retnet(x, mode="parallel")
    out_rec_ret = retnet(x, mode="recurrent")

    print(f"  Parallel output shape: {out_par_ret.shape}")
    print(f"  Recurrent output shape: {out_rec_ret.shape}")

    diff_ret = (out_par_ret - out_rec_ret).abs().max().item()
    print(f"  Max difference (parallel vs recurrent): {diff_ret:.6f}")

    # Show decay effect
    decay = retnet._create_decay_matrix(8)
    print(f"\n  Decay matrix (γ=0.9, seq_len=8):")
    for i in range(8):
        row = "    "
        for j in range(8):
            row += f"{decay[i, j]:.3f} "
        print(row)
    print(f"  → Token 7 weights token 0 by only {0.9**7:.3f} (long-ago = forgotten)")

    # --- 3. Gated DeltaNet ---
    print(f"\n{'━' * 70}")
    print("3. GATED DeltaNet (selective memory)")
    print(f"{'━' * 70}")

    deltanet = GatedDeltaNetAttention(d_model=d_model, num_heads=num_heads)
    out_delta = deltanet(x)

    print(f"  Output shape: {out_delta.shape}")
    print(f"  Parameters: {sum(p.numel() for p in deltanet.parameters()):,}")

    # --- Comparison table ---
    print(f"\n{'━' * 70}")
    print("COMPARISON: SOFTMAX ATTENTION vs LINEAR VARIANTS")
    print(f"{'━' * 70}")
    print(f"""
    ┌────────────────────┬────────────┬────────────┬───────────┬───────────────┐
    │ Property           │ Softmax    │ Linear     │ RetNet    │ DeltaNet      │
    ├────────────────────┼────────────┼────────────┼───────────┼───────────────┤
    │ Training cost      │ O(n²·d)    │ O(n·d²)    │ O(n·d²)*  │ O(n·d²)       │
    │ Inference/token    │ O(n·d)     │ O(d²)      │ O(d²)     │ O(d²)         │
    │ State size         │ O(n·d)     │ O(d²)      │ O(d²)     │ O(d²)         │
    │ Forgetting         │ Implicit   │ None       │ Exp decay │ Learned gate  │
    │ Expressiveness     │ Highest    │ Lower      │ Medium    │ Higher        │
    │ Parallel training  │ Yes        │ Yes        │ Yes       │ Yes (chunked) │
    │ Recurrent infer.   │ No         │ Yes        │ Yes       │ Yes           │
    └────────────────────┴────────────┴────────────┴───────────┴───────────────┘

    * RetNet can also use a "chunkwise" mode: parallel within chunks, recurrent across chunks.

    KEY TAKEAWAY:
    Linear attention models offer O(1) per-token inference cost with a fixed-size
    state, making them ideal for streaming and very long sequences. The trade-off
    is lower expressiveness compared to softmax attention, but gating (DeltaNet)
    and decay (RetNet) close much of this gap.
    """)

    # --- RWKV note ---
    print("━" * 70)
    print("NOTE ON RWKV")
    print("━" * 70)
    print("""
    RWKV (Receptance Weighted Key Value) is another linear attention variant
    that uses a similar recurrent formulation with learned decay. It's closely
    related to RetNet but was developed independently.

    RWKV-6 state update:
        S_t = diag(w_t) * S_(t-1) + k_t * v_t^T

    Where w_t is a learned, input-dependent decay vector (per head, per dim).
    This is like RetNet but with DIFFERENT decay rates per dimension,
    giving finer-grained control over what to forget.

    RWKV has been scaled to 14B+ parameters and performs competitively
    with transformer models of similar size, with much faster inference.
    """)


if __name__ == "__main__":
    demo()
