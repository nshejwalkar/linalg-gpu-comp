"""
Triton-Accelerated Blocked Householder QR — Competition Submission v2

Key improvement over v1 (submission_blocked_wy.py):
  - Panel factorization is a single Triton kernel (one CUDA launch per outer block)
  - Eliminates Python loop overhead for the b sequential Householder steps
  - Trailing update still uses torch.bmm (tensor cores via cuBLAS)

Triton kernel strategy:
  - One CUDA program per batch element (parallelism across batch dimension)
  - Load the full panel (n-k, b) into shared memory / registers
  - Apply b sequential Householder steps entirely within the kernel
  - Write back updated panel + tau values

This matters when:
  - n is large (many Python iterations → significant overhead in v1)
  - batch is small (CUDA parallelism comes from within the panel, not across batch)
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t


# ────────────────────────────────────────────────────────────────────────────
# Triton kernel: panel factorization for one outer block
# ────────────────────────────────────────────────────────────────────────────

@triton.jit
def _panel_qr_kernel(
    A_ptr,     # float32 (batch, n, n) — input/output
    tau_ptr,   # float32 (batch, n)    — output
    k_start,   # int: starting column of this block
    n,         # int: matrix size
    stride_Ab, stride_Ar, stride_Ac,   # strides for A
    stride_tb, stride_tc,              # strides for tau
    M_POW2: tl.constexpr,  # next power of 2 >= (n - k_start), for the panel height
    B: tl.constexpr,       # panel width = block size (must be power of 2)
    UPDATE_ONLY_PANEL: tl.constexpr,   # True → update H[k:, k:k+B] only
):
    """
    Compute Householder QR for a panel of width B starting at column k_start.
    One program per batch element.

    The kernel:
      1. Loads the panel A[bid, k_start:, k_start:k_start+B] into SRAM
      2. Applies B sequential Householder steps (inner loop over j in [0..B-1])
      3. Writes back the updated panel and tau values

    Parallelism: across batch dimension (program_id(0) = batch index)
    Sequential: over j within the panel (unavoidable dependency chain)
    """
    bid = tl.program_id(0)
    m   = n - k_start  # actual panel height

    # Row/col offset vectors for the panel tile
    rows = tl.arange(0, M_POW2)   # 0 .. M_POW2-1
    cols = tl.arange(0, B)        # 0 .. B-1

    # Pointers to A[bid, k_start:, k_start:k_start+B]
    A_panel_base = (A_ptr
                    + bid     * stride_Ab
                    + k_start * stride_Ar
                    + k_start * stride_Ac)
    ptrs = A_panel_base + rows[:, None] * stride_Ar + cols[None, :] * stride_Ac
    mask = (rows[:, None] < m) & (cols[None, :] < B)

    # Load panel into registers
    panel = tl.load(ptrs, mask=mask, other=0.0)   # (M_POW2, B)

    # Tau output buffer (in registers)
    tau_vals = tl.zeros([B], dtype=tl.float32)

    for j in tl.static_range(B):
        # Skip if j is beyond the actual matrix column count
        if j >= m:
            break

        # ── Extract column j (active rows: j .. m-1) ──────────────────────
        active = (rows >= j) & (rows < m)   # boolean mask, shape (M_POW2,)
        col_j  = tl.where(active, panel[:, j], 0.0)  # (M_POW2,) zero-padded

        # ── Compute Householder reflector ──────────────────────────────────
        # alpha = col_j[j] (the diagonal element)
        alpha = tl.sum(tl.where(rows == j, col_j, 0.0))

        # norm = ||col_j[j:]||₂
        norm_sq = tl.sum(col_j * col_j)   # sum over active rows (zeros elsewhere)
        norm    = tl.sqrt(norm_sq)

        # beta = -sign(alpha) * norm  (the new R diagonal)
        s     = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta  = -s * norm

        # v[0] = alpha - beta,  v[1:] = col_j[j+1:]
        v0     = alpha - beta                     # scalar
        v_norm_sq = v0 * v0 + tl.sum(tl.where(rows > j, col_j * col_j, 0.0))

        tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)
        tau_vals = tau_vals + tl.where(cols == j, tau_j, 0.0)

        # u = v / v[0]: u[j]=1 (implicit), u[j+1:] = col_j[j+1:] / v0
        safe_v0 = tl.where(v0 != 0.0, v0, 1.0)
        u_sub   = tl.where(rows > j, col_j / safe_v0, 0.0)  # (M_POW2,)
        #   u[j]   = 1  (implicit)
        #   u[i>j] = col_j[i] / v0

        # ── Apply reflector to trailing panel columns j+1 .. B-1 ──────────
        # u^T @ panel[:, c]  for c > j
        #   = panel[j, c] + sum_{i>j} u_sub[i] * panel[i, c]
        trailing_cols = cols > j

        # j-th row of panel for c > j
        row_j = tl.sum(tl.where(rows[:, None] == j, panel, 0.0), axis=0)   # (B,)
        # u_sub^T @ panel for c > j
        usub_T_panel = tl.sum(u_sub[:, None] * panel, axis=0)               # (B,)

        uTpanel = tl.where(trailing_cols, row_j + usub_T_panel, 0.0)       # (B,)

        # panel[j, c]   -= tau * 1 * uTpanel[c]   for c > j
        panel = panel - tl.where(
            (rows == j)[:, None] & trailing_cols[None, :],
            tau_j * uTpanel[None, :],
            0.0
        )
        # panel[i>j, c] -= tau * u_sub[i] * uTpanel[c]   for c > j
        panel = panel - tl.where(
            (rows > j)[:, None] & trailing_cols[None, :],
            tau_j * u_sub[:, None] * uTpanel[None, :],
            0.0
        )

        # ── Store: R diagonal + Householder vector ────────────────────────
        # H[j, j] = beta
        panel = tl.where(
            (rows == j)[:, None] & (cols == j)[None, :],
            tl.full([M_POW2, B], beta, dtype=tl.float32),
            panel
        )
        # H[i>j, j] = u_sub[i]
        panel = tl.where(
            (rows > j)[:, None] & (cols == j)[None, :],
            u_sub[:, None] * tl.ones([M_POW2, B], dtype=tl.float32),
            panel
        )

    # ── Write back ────────────────────────────────────────────────────────
    tl.store(ptrs, panel, mask=mask)

    # Write tau values
    tau_base = tau_ptr + bid * stride_tb + k_start * stride_tc
    tl.store(tau_base + cols, tau_vals, mask=cols < B)


# ────────────────────────────────────────────────────────────────────────────
# WY form + trailing update (same as v1, pure torch)
# ────────────────────────────────────────────────────────────────────────────

def _build_wy_and_update(H: torch.Tensor, tau_all: torch.Tensor, k: int, b: int, n: int):
    """
    Build Y, T and apply trailing update.

    NOTE (bug fix): T must be built from G = Y^T Y computed via a SINGLE
    batched GEMM over the full panel block (rows k..n-1), including the
    diagonal row of each column (Y[j,j]=1, Y[j,0:j]=H[k+j,k:k+j]). An
    earlier version started the inner-product at row k+j+1 and silently
    dropped that diagonal-row contribution, producing a wrong T for any
    matrix needing >1 outer block (n > _BLOCK). See submission_blocked_wy.py
    for the detailed derivation / regression test.
    """
    batch  = H.shape[0]
    device = H.device
    dtype  = H.dtype

    # Build Y: lower-triangular part of H[k:, k:k+b], with 1s on diagonal
    Y = torch.tril(H[:, k:, k:k+b])
    idx = torch.arange(b, device=device)
    Y[:, idx, idx] = 1.0

    # Gram matrix G = Y^T Y — single batched GEMM, correctly includes
    # the diagonal-row contribution for every column.
    G = torch.bmm(Y.transpose(-1, -2), Y)   # (batch, b, b)

    # Build T (b × b upper triangular) column by column
    T = torch.zeros(batch, b, b, device=device, dtype=dtype)
    for j in range(b):
        T[:, j, j] = tau_all[:, k + j]
        if j > 0:
            tj = tau_all[:, k + j].view(batch, 1, 1)
            Gj = G[:, :j, j:j+1]                       # (batch, j, 1)
            T[:, :j, j:j+1] = -tj * torch.bmm(T[:, :j, :j], Gj)

    # Trailing GEMM: A_trail -= Y @ T^T @ (Y^T @ A_trail)
    A_trail = H[:, k:, k+b:]
    C  = torch.bmm(Y.transpose(-1, -2), A_trail)
    TC = torch.bmm(T.transpose(-1, -2), C)
    H[:, k:, k+b:] -= torch.bmm(Y, TC)


# ────────────────────────────────────────────────────────────────────────────
# Main submission
# ────────────────────────────────────────────────────────────────────────────

_BLOCK = 64  # tune for B200 (try 32, 64, 96, 128)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def custom_kernel(data: input_t) -> output_t:
    """
    Triton-accelerated blocked WY-form Householder QR.

    Panel factorization: single Triton kernel launch per outer block.
    Trailing update: two batched GEMMs via torch.bmm (cuBLAS, tensor cores).
    """
    batch, n, _ = data.shape
    H       = data.clone()
    tau_all = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
    B       = _BLOCK

    for k in range(0, n, B):
        b     = min(B, n - k)
        k_end = k + b
        m     = n - k
        M_P2  = _next_pow2(m)  # M_POW2 must be constexpr in Triton

        # ── 1. Panel factorization (Triton kernel) ─────────────────────────
        grid = (batch,)
        _panel_qr_kernel[grid](
            H, tau_all,
            k_start=k, n=n,
            stride_Ab=H.stride(0), stride_Ar=H.stride(1), stride_Ac=H.stride(2),
            stride_tb=tau_all.stride(0), stride_tc=tau_all.stride(1),
            M_POW2=M_P2,
            B=b,
            UPDATE_ONLY_PANEL=True,
        )

        # ── 2 + 3. WY form + trailing GEMM ────────────────────────────────
        if k_end < n:
            _build_wy_and_update(H, tau_all, k=k, b=b, n=n)

    return H, tau_all
