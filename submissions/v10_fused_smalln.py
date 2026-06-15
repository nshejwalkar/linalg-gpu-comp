"""
v10_fused_smalln — fully fused Householder QR for small n (one program per matrix).

Strategy (MAGMA batched-small-QR tactic): for n that fits an SM, run the ENTIRE
unblocked Householder QR of one n x n matrix inside a single Triton program. The
matrix is loaded once into the tile (registers/SRAM), all n reflector steps run
in-kernel, then H (R above diag, reflectors below) and tau are written back once.
This eliminates the ~10^4 tiny kernel launches that make the eager blocked-WY and
cuSOLVER's batched geqrf slow on the small-n shapes (b20 n32, b40 n176).

Numerics mirror submissions/v1.py::_householder_step EXACTLY (LAPACK SGEQRF sign
convention):
    alpha = x[0]; beta = -sign(alpha)*||x||;  v0 = alpha - beta
    tau   = 2 v0^2 / (v0^2 + ||x[1:]||^2)      (0 if the column is ~0)
    u     = x[1:] / v0   (reflector stored below the diagonal; u[0]=1 implicit)
Reflector applied to ALL trailing columns immediately (unblocked == same H,tau as
blocked WY / geqrf).

Dispatch: fused kernel for n the SM can hold (n <= _FUSED_NMAX); torch.geqrf for
everything else, so the full 19-test suite still passes (large-n falls back to the
correct cuSOLVER path). n=352 (495 KB/matrix) does NOT fit one SM's 228 KB shared
mem, so it goes to geqrf.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# Largest n handled by the fused single-program kernel. n=176 -> N_POW2=256 tile
# (256x256 f32 = 256 KB worth of tile, lives in registers+local backed by L1/SMEM).
# n=352 -> 512x512 tile is far too big for one program, so it falls back to geqrf.
_FUSED_NMAX = 256


@triton.jit
def _fused_qr_kernel(
    A_ptr,            # float32 (batch, n, n)  in/out: H
    tau_ptr,          # float32 (batch, n)     out: tau
    n,                # int  matrix size
    stride_ab, stride_ar, stride_ac,
    stride_tb, stride_tc,
    N_POW2: tl.constexpr,     # next_pow2(n)
):
    """One program == one matrix. program_id(0) = batch index."""
    bid = tl.program_id(0)

    rows = tl.arange(0, N_POW2)
    cols = tl.arange(0, N_POW2)
    row_valid = rows < n
    col_valid = cols < n

    base = A_ptr + bid * stride_ab
    ptrs = base + rows[:, None] * stride_ar + cols[None, :] * stride_ac
    mask = row_valid[:, None] & col_valid[None, :]

    # Full matrix tile.
    H = tl.load(ptrs, mask=mask, other=0.0)            # (N_POW2, N_POW2)
    tau_acc = tl.zeros([N_POW2], dtype=tl.float32)

    for j in range(0, n):
        # --- extract column j (rows j..n-1 are "active") -------------------
        col_j = tl.sum(tl.where(cols[None, :] == j, H, 0.0), axis=1)  # (N_POW2,)
        active = (rows >= j) & row_valid
        x = tl.where(active, col_j, 0.0)               # zero-padded subcolumn

        alpha = tl.sum(tl.where(rows == j, x, 0.0))    # scalar H[j,j]
        norm_sq = tl.sum(x * x)
        norm = tl.sqrt(norm_sq)

        s = tl.where(alpha >= 0.0, 1.0, -1.0)          # sign(alpha), sign(0)=+1
        beta = -s * norm
        v0 = alpha - beta
        # ||x[1:]||^2 = norm_sq - alpha^2  (rows > j part)
        tail_sq = tl.sum(tl.where(rows > j, x * x, 0.0))
        v_norm_sq = v0 * v0 + tail_sq
        tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)

        safe_v0 = tl.where(v0 != 0.0, v0, 1.0)
        # u_sub[i] = x[i]/v0 for i>j ; u[j]=1 implicit ; 0 elsewhere
        u = tl.where(rows > j, x / safe_v0, 0.0)
        u = tl.where(rows == j, 1.0, u)                # u[j] = 1

        # --- apply reflector to trailing columns c > j --------------------
        # w[c] = u^T H[:, c] = sum_i u[i] * H[i,c]    (i ranges over active rows)
        u_for_dot = tl.where(active, u, 0.0)
        w = tl.sum(u_for_dot[:, None] * H, axis=0)     # (N_POW2,)
        trailing = (cols > j) & col_valid
        w = tl.where(trailing, w, 0.0)
        # H[i,c] -= tau_j * u[i] * w[c]  for i in active rows, c > j
        H = H - (tau_j * u_for_dot[:, None]) * w[None, :]

        # --- write reflector + R diagonal into H column j -----------------
        # H[j,j] = beta ; H[i>j, j] = u_sub[i] = x[i]/v0
        new_colj = tl.where(rows == j, beta, tl.where(rows > j, x / safe_v0, col_j))
        H = tl.where(cols[None, :] == j, new_colj[:, None], H)

        tau_acc = tau_acc + tl.where(cols == j, tau_j, 0.0)

    tl.store(ptrs, H, mask=mask)
    tau_ptrs = tau_ptr + bid * stride_tb + cols * stride_tc
    tl.store(tau_ptrs, tau_acc, mask=col_valid)


def _num_warps_for(n_pow2: int) -> int:
    # Spread the big tile across more warps to cut per-thread register pressure.
    if n_pow2 >= 256:
        return 16
    if n_pow2 >= 128:
        return 4
    return 2


def _fused_qr(data: torch.Tensor):
    batch, n, _ = data.shape
    H = data.clone()
    tau = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
    N_P2 = _next_pow2(n)
    grid = (batch,)
    _fused_qr_kernel[grid](
        H, tau, n,
        H.stride(0), H.stride(1), H.stride(2),
        tau.stride(0), tau.stride(1),
        N_POW2=N_P2,
        num_warps=_num_warps_for(N_P2),
    )
    return H, tau


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if n <= _FUSED_NMAX:
        return _fused_qr(data)
    return torch.geqrf(data)
