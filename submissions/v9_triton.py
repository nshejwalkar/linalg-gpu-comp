"""
v9_triton — Blocked WY Householder QR with the PANEL factorization fused into a
SINGLE Triton kernel (one program per batch element).

Why: blocked_wy (v1) is launch-bound — the `b` sequential Householder steps of
each column block issue ~10^4 tiny torch kernels/iter (copy_/mul/sub/norm/...).
Profiling (research/findings.md C1) shows bmm is only 15.8% of GPU time; the win
is collapsing the panel's per-step elementwise ops into ONE kernel launch/block.

This file:
  * PANEL  -> one Triton kernel per outer block (`_panel_qr_kernel`).
  * WY build + trailing update -> torch.bmm (3 GEMMs/block), same as v1.
  * Shape dispatch -> Triton panel where it helps; torch.geqrf elsewhere.

Numerics mirror submissions/v1.py (`_householder_step`) EXACTLY (verified 19/19):
  alpha = col[0]; sign_a = sign(alpha) (0 -> +1); beta = -sign_a*||col[0:]||;
  v0 = alpha - beta; v = [v0, col[1:]]; tau = 2*v0^2/||v||^2 (0 if ||v||^2==0);
  u_sub = col[1:]/v0 (safe_v0 guards |v0|<1e-30); H[diag]=beta; H[below]=u_sub;
  apply F = I - tau*[1;u_sub]*[1;u_sub]^T to trailing panel columns.

m-scaling strategy (avoids the archive bug of holding the whole (M_POW2,b) panel
in registers): the kernel reads the panel from global memory one column at a
time. Per step j it holds only ~one column of height m (M_POW2 lane vector), so
register pressure is O(M_POW2), independent of b. That scales to any panel height
(n=512 first block = 512 rows is fine), at the cost of more global traffic. With
one program per batch element the big-batch shapes still beat the launch-bound
torch path. tl.arange uses POWER-OF-2 constexpr bounds + masking (b can be
non-pow2 on the last block; m is arbitrary).
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t


_BLOCK = 32          # panel width; smaller block = shorter inner loop, more GEMMs
_EPS_V0 = 1e-30      # matches v1 safe_v0 guard


# ────────────────────────────────────────────────────────────────────────────
# Triton panel kernel: factor one (m x b) panel, one program per batch element.
# Reads columns from global memory (O(M_POW2) registers, any m).
# ────────────────────────────────────────────────────────────────────────────
@triton.jit
def _panel_qr_kernel(
    A_ptr,          # float32 (batch, n, n) — in/out (the panel is updated in place)
    tau_ptr,        # float32 (batch, n)    — out
    k_start,        # int: first row & column of this block (panel = A[k:, k:k+b])
    b,              # int: panel width (runtime; may be < BLOCK on last block)
    m,              # int: panel height = n - k_start (runtime)
    stride_Ab, stride_Ar, stride_Ac,
    stride_tb, stride_tc,
    M_POW2: tl.constexpr,   # next_pow2(m_max for this launch) — arange bound
    EPS_V0: tl.constexpr,   # safe_v0 guard (matches v1: 1e-30)
):
    bid = tl.program_id(0)
    rows = tl.arange(0, M_POW2)                       # lane = panel row index
    # Base pointer to A[bid, k_start, k_start] (top-left of the panel).
    base = A_ptr + bid * stride_Ab + k_start * stride_Ar + k_start * stride_Ac

    # Sequential Householder steps over the panel columns.
    for j in range(0, b):
        # ── Load active column j: rows [j, m) ────────────────────────────────
        col_mask = (rows >= j) & (rows < m)
        col_ptr = base + rows * stride_Ar + j * stride_Ac
        col = tl.load(col_ptr, mask=col_mask, other=0.0)     # (M_POW2,), 0 outside

        # alpha = col[j] (diagonal entry of this column).
        alpha = tl.sum(tl.where(rows == j, col, 0.0))
        norm_sq = tl.sum(col * col)                          # ||col[j:]||^2
        norm = tl.sqrt(norm_sq)

        sign_a = tl.where(alpha >= 0.0, 1.0, -1.0)           # sign(0) -> +1 (v1)
        beta = -sign_a * norm
        v0 = alpha - beta
        # ||v||^2 = v0^2 + sum_{i>j} col[i]^2 = v0^2 + (norm_sq - alpha^2)
        v_norm_sq = v0 * v0 + (norm_sq - alpha * alpha)
        tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)

        safe_v0 = tl.where(tl.abs(v0) < EPS_V0, 1.0, v0)
        # u_sub[i] = col[i]/v0 for i>j ; u[j] = 1 (implicit). 0 elsewhere.
        u_sub = tl.where(rows > j, col / safe_v0, 0.0)       # (M_POW2,)

        # ── Apply F = I - tau*u*u^T to trailing columns c in (j, b) ──────────
        # For column c: w = u^T col_c = col_c[j] + sum_{i>j} u_sub[i]*col_c[i]
        #               col_c -= tau * w * u   (u[j]=1, u[i>j]=u_sub[i])
        for c in range(j + 1, b):
            cc_ptr = base + rows * stride_Ar + c * stride_Ac
            cc = tl.load(cc_ptr, mask=col_mask, other=0.0)   # rows [j, m)
            cj = tl.sum(tl.where(rows == j, cc, 0.0))        # col_c[j]
            w = tau_j * (cj + tl.sum(u_sub * cc))            # tau * (u^T col_c)
            # new col_c: row j -= w*1 ; rows i>j -= w*u_sub[i]
            upd = tl.where(rows == j, cc - w, cc - w * u_sub)
            upd = tl.where(col_mask, upd, 0.0)
            tl.store(cc_ptr, upd, mask=col_mask)

        # ── Write back column j: H[diag]=beta, H[below]=u_sub, tau[col]=tau_j ─
        out_col = tl.where(rows == j, beta, u_sub)           # beta at j, u_sub below
        tl.store(col_ptr, out_col, mask=col_mask)
        tl.store(tau_ptr + bid * stride_tb + (k_start + j) * stride_tc, tau_j)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# ────────────────────────────────────────────────────────────────────────────
# WY build + trailing update — pure torch (same math as v1).
# ────────────────────────────────────────────────────────────────────────────
def _build_wy(H, tau_all, k, b, n):
    batch = H.shape[0]
    device = H.device
    dtype = H.dtype
    Y = torch.tril(H[:, k:, k:k + b])
    idx = torch.arange(b, device=device)
    Y[:, idx, idx] = 1.0
    G = torch.bmm(Y.transpose(-1, -2), Y)
    T = torch.zeros(batch, b, b, device=device, dtype=dtype)
    for j in range(b):
        T[:, j, j] = tau_all[:, k + j]
        if j > 0:
            tj = tau_all[:, k + j].view(batch, 1, 1)
            Gj = G[:, :j, j:j + 1]
            T[:, :j, j:j + 1] = -tj * torch.bmm(T[:, :j, :j], Gj)
    return Y, T


def _trailing_update(H, Y, T, k, b):
    A_trail = H[:, k:, k + b:]
    C = torch.bmm(Y.transpose(-1, -2), A_trail)
    TC = torch.bmm(T.transpose(-1, -2), C)
    H[:, k:, k + b:] -= torch.bmm(Y, TC)


def _blocked_wy_triton(data):
    batch, n, _ = data.shape
    H = data.clone()
    tau_all = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
    B = _BLOCK
    grid = (batch,)
    # M_POW2 is a Triton constexpr -> a new value triggers a recompile. Use ONE value
    # per call (next_pow2 of the full height) instead of next_pow2(m) per block, so each
    # matrix size compiles the panel kernel exactly once (masking covers shorter late
    # blocks). Per-block next_pow2(m) caused ~30 recompiles -> grader 300s timeout.
    MP = _next_pow2(n)
    for k in range(0, n, B):
        b = min(B, n - k)
        k_end = k + b
        m = n - k
        # Fused panel factorization: ONE Triton launch for the b sequential steps.
        _panel_qr_kernel[grid](
            H, tau_all,
            k, b, m,
            H.stride(0), H.stride(1), H.stride(2),
            tau_all.stride(0), tau_all.stride(1),
            M_POW2=MP,
            EPS_V0=_EPS_V0,
            num_warps=4,
        )
        if k_end < n:
            Y, T = _build_wy(H, tau_all, k=k, b=b, n=n)
            _trailing_update(H, Y, T, k=k, b=b)
    return H, tau_all


# ────────────────────────────────────────────────────────────────────────────
# Dispatch
# ────────────────────────────────────────────────────────────────────────────
def _use_triton(batch: int, n: int) -> bool:
    # Big-batch / medium-n is where the launch-bound torch panel hurts most and
    # the fused Triton panel + batched bmm wins (headline b640 n512). Tiny n=32
    # and small-batch huge-n (2048/4096) lose to cuSOLVER regardless (findings
    # E1), so route those to geqrf.
    return batch >= 32 and 128 <= n <= 1024


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if _use_triton(batch, n):
        return _blocked_wy_triton(data)
    return torch.geqrf(data)
