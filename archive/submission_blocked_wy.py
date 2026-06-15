"""
Blocked WY-Form Householder QR — Competition Submission v1

Algorithm (Trefethen & Bau Lectures 10, 16 + Schreiber-Van Loan 1989):
  1. Panel factorization: b sequential Householder steps on H[k:, k:k+b].
  2. WY accumulation: build Y (Householder vectors) and T so that
       H_k H_{k+1} ... H_{k+b-1} = I - Y T Y^T
  3. Trailing update: A_trail -= Y T^T (Y^T A_trail)  [two batched GEMMs, tensor-core friendly]

Output format: (H, tau) matching torch.geqrf / LAPACK SGEQRF convention:
  H[..., j, j]    = beta_j   (new R diagonal)
  H[..., j+1:, j] = u_j[1:] (Householder vector, u_j[0]=1 implicit)
  tau[..., j]     = tau_j   (scalar: F_j = I - tau_j u_j u_j^T)

Correctness verified with torch.linalg.householder_product(H, tau).
"""

import torch
from task import input_t, output_t


# ── Householder step (LAPACK geqrf convention) ────────────────────────────────

def _householder_step(H: torch.Tensor, tau_all: torch.Tensor, col: int, col_end: int):
    """
    Compute and apply one Householder reflector to column `col` of H,
    updating H[col:, col+1 : col_end] in-place.

    Sign convention (Trefethen eq 10.5): v = sign(x_0)||x||e_1 + x
    avoids catastrophic cancellation in v_0.
    """
    n  = H.shape[1]
    x  = H[:, col:, col].clone()   # (batch, m)
    m  = x.shape[-1]
    if m <= 1:
        tau_all[:, col] = 0.0
        return

    alpha   = x[:, 0]
    norm_x  = x.norm(dim=-1)
    sign_a  = alpha.sign()
    sign_a[sign_a == 0] = 1.0
    beta    = -sign_a * norm_x          # new R diagonal
    v0      = alpha - beta              # v[0]

    v_norm_sq = v0**2 + (x[:, 1:] ** 2).sum(-1)
    tau_j     = torch.where(v_norm_sq > 0.0,
                             2.0 * v0**2 / v_norm_sq,
                             torch.zeros_like(v_norm_sq))
    safe_v0   = v0.clone(); safe_v0[safe_v0.abs() < 1e-30] = 1.0
    u_sub     = x[:, 1:] / safe_v0.unsqueeze(-1)   # u[1:] = x[1:] / v0

    # Apply reflector to trailing PANEL columns [col+1 .. col_end-1]
    end = min(col_end, n)
    if col + 1 < end:
        T    = H[:, col:, col+1:end]
        uTT  = T[:, 0, :] + torch.bmm(u_sub.unsqueeze(1), T[:, 1:, :]).squeeze(1)
        H[:, col,    col+1:end] -= tau_j.unsqueeze(-1) * uTT
        H[:, col+1:, col+1:end] -= (
            tau_j.unsqueeze(-1).unsqueeze(-1)
            * u_sub.unsqueeze(-1)
            * uTT.unsqueeze(-2)
        )

    H[:, col, col]   = beta
    if col + 1 < n:
        H[:, col+1:, col] = u_sub
    tau_all[:, col]  = tau_j


# ── WY form: Y matrix + T matrix ─────────────────────────────────────────────

def _build_wy(H: torch.Tensor, tau_all: torch.Tensor, k: int, b: int, n: int):
    """
    Build the WY representation for block starting at column k, width b:
        H_k H_{k+1} ... H_{k+b-1} = I - Y T Y^T   (Schreiber-Van Loan 1989)

    Y : (batch, n-k, b)  unit-lower-triangular, columns = Householder vectors
    T : (batch, b,   b)  upper triangular

    Key: we compute G = Y^T Y via a single batched GEMM, then build T
    column-by-column using G (avoids per-column inner-product kernel launches
    AND correctly includes the diagonal row contribution that a naive
    "start from row k+j+1" approach would miss).
    """
    batch  = H.shape[0]
    m      = n - k
    device = H.device
    dtype  = H.dtype

    # Build Y: lower triangular part of H[k:, k:k+b], with 1s on diagonal
    Y = torch.tril(H[:, k:, k:k+b])
    idx = torch.arange(b, device=device)
    Y[:, idx, idx] = 1.0

    # Gram matrix G = Y^T Y  —  single batched GEMM (b × m) × (m × b)
    G = torch.bmm(Y.transpose(-1, -2), Y)   # (batch, b, b)

    # Build T upper-triangular, column by column
    T = torch.zeros(batch, b, b, device=device, dtype=dtype)
    for j in range(b):
        T[:, j, j] = tau_all[:, k + j]
        if j > 0:
            # T[0:j, j] = -tau_j * T[0:j, 0:j] @ G[0:j, j]
            # G[0:j, j] = Y^T Y [0:j, j] = (Y[j:, 0:j]^T @ Y[j:, j])
            # This correctly includes the diagonal row (Y[j,j]=1, Y[j,0:j]=H[k+j,k:k+j])
            tj   = tau_all[:, k + j].view(batch, 1, 1)
            Gj   = G[:, :j, j:j+1]                      # (batch, j, 1)
            T[:, :j, j:j+1] = -tj * torch.bmm(T[:, :j, :j], Gj)

    return Y, T


# ── Trailing update: apply block reflector via two GEMMs ─────────────────────

def _trailing_update(H: torch.Tensor, Y: torch.Tensor, T: torch.Tensor, k: int, b: int):
    """
    Apply Q_block^T = (H_k ... H_{k+b-1})^T = I - Y T^T Y^T to trailing columns:
        A_trail  =  A_trail  -  Y  @  (T^T  @  (Y^T @ A_trail))
    Two batched GEMMs — fully tensor-core friendly.
    """
    A_trail = H[:, k:, k+b:]
    C  = torch.bmm(Y.transpose(-1, -2), A_trail)   # (batch, b, p)  ← GEMM 1
    TC = torch.bmm(T.transpose(-1, -2), C)         # (batch, b, p)  ← TRMM/GEMM 2
    H[:, k:, k+b:] -= torch.bmm(Y, TC)             #                ← GEMM 3


# ── Main entry point ─────────────────────────────────────────────────────────

_BLOCK = 64   # tune on B200: try 32, 64, 96, 128


def custom_kernel(data: input_t) -> output_t:
    """
    Blocked WY-form Householder QR.

    For each outer block k ∈ {0, B, 2B, ...}:
      1. Panel: b sequential Householder steps  →  O(n b²) work
      2. WY:   one GEMM for G = Y^T Y  +  b-step T recurrence
      3. GEMM: three batched GEMMs for trailing update  →  O(n² b) work

    Returns (H, tau) in torch.geqrf / LAPACK SGEQRF format.
    """
    batch, n, _ = data.shape
    H       = data.clone()
    tau_all = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
    B       = _BLOCK

    for k in range(0, n, B):
        b    = min(B, n - k)
        k_end = k + b

        # 1. Panel factorization
        for j in range(b):
            _householder_step(H, tau_all, k + j, k_end)

        # 2 + 3. WY form + trailing GEMM
        if k_end < n:
            Y, T = _build_wy(H, tau_all, k=k, b=b, n=n)
            _trailing_update(H, Y, T, k=k, b=b)

    return H, tau_all
