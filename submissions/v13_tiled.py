"""
v13_tiled — Blocked WY Householder QR with a SHARED-MEMORY-RESIDENT panel kernel.

Base: submissions/v9_triton.py (champion, 2.38x). v9's `_panel_qr_kernel` is the
#1 GPU bottleneck (50-59% of GPU time, research/findings.md C3) because it
re-reads/re-writes the trailing panel columns from GLOBAL memory on every one of
its b sequential steps (O(b^2) global traffic, no on-chip residence).

THE FIX (this file): one Triton program per batch element. Load the WHOLE panel
block (rows [k:n], cols [k:k+b]) into an on-chip [M_POW2, B_POW2] tile ONCE, run
all b sequential Householder steps entirely on that resident tile (compute the
reflector, then apply F = I - tau*u*u^T to the in-panel trailing columns, all on
chip), then write the factored panel back to global ONCE. This removes the
per-step global re-read/re-write that dominated v9.

Residency / smem fit: dispatch range is n<=1024 with width _BLOCK=32, so the
largest tile is the first block of n=1024 -> 1024 x 32 x 4B = 128 KB, which fits
B200's 228 KB on-chip budget. (n=2048/4096 route to torch.geqrf, same as v9, so
the panel tile is always resident for the shapes we actually run.)

Numerics mirror submissions/v1.py (`_householder_step`) and v9 EXACTLY (19/19):
  alpha = col[0]; sign_a = sign(alpha) (0 -> +1); beta = -sign_a*||col[j:]||;
  v0 = alpha - beta; ||v||^2 = v0^2 + (||col[j:]||^2 - alpha^2);
  tau = 2*v0^2/||v||^2 (0 if ||v||^2 == 0); safe_v0 guards |v0| < 1e-30;
  u_sub[i>j] = col[i]/v0, u[j] = 1; H[diag] = beta, H[below] = u_sub.
WY build + trailing update stay torch.bmm (3 GEMMs/block), identical to v9.

constexpr budget: panel kernel keys on M_POW2 = next_pow2(n) (3 distinct values
across the dispatch range: 256, 512, 1024) and a fixed B_POW2 = 32 -> 3 compiles
total, same as v9. tl.arange needs power-of-2 bounds; masking covers the real
(m, b) inside the padded tile.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t


_BLOCK = 32          # panel width; tile column extent (B_POW2 = next_pow2(_BLOCK))
_EPS_V0 = 1e-30      # matches v1 safe_v0 guard


# ────────────────────────────────────────────────────────────────────────────
# Triton panel kernel: factor one (m x b) panel, one program per batch element.
# The panel is loaded into a resident [M_POW2, B_POW2] on-chip tile ONCE, all b
# steps run on that tile, and it is written back ONCE — no per-step global reads.
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
    M_POW2: tl.constexpr,   # next_pow2(n) — row extent of the resident tile
    B_POW2: tl.constexpr,   # next_pow2(_BLOCK) — column extent of the resident tile
    EPS_V0: tl.constexpr,   # safe_v0 guard (matches v1: 1e-30)
):
    bid = tl.program_id(0)
    rows = tl.arange(0, M_POW2)                       # lane = panel row index
    cols = tl.arange(0, B_POW2)                       # column index within panel
    # Base pointer to A[bid, k_start, k_start] (top-left of the panel).
    base = A_ptr + bid * stride_Ab + k_start * stride_Ar + k_start * stride_Ac

    # ── Load the whole panel into a resident on-chip tile ONCE ───────────────
    tile_ptr = base + rows[:, None] * stride_Ar + cols[None, :] * stride_Ac
    tile_mask = (rows[:, None] < m) & (cols[None, :] < b)
    panel = tl.load(tile_ptr, mask=tile_mask, other=0.0)   # (M_POW2, B_POW2)

    # ── b sequential Householder steps, all on the resident tile ─────────────
    for j in range(0, b):
        is_j_row = rows == j                          # (M_POW2,) selector for row j
        # Active column j of the (already-updated) panel, rows [j, m).
        col = tl.sum(tl.where(cols[None, :] == j, panel, 0.0), axis=1)  # (M_POW2,)
        col = tl.where(rows >= j, col, 0.0)           # zero rows above the diagonal

        alpha = tl.sum(tl.where(is_j_row, col, 0.0))  # col[j] (diagonal entry)
        norm_sq = tl.sum(col * col)                   # ||col[j:]||^2
        norm = tl.sqrt(norm_sq)

        sign_a = tl.where(alpha >= 0.0, 1.0, -1.0)    # sign(0) -> +1 (v1)
        beta = -sign_a * norm
        v0 = alpha - beta
        # ||v||^2 = v0^2 + sum_{i>j} col[i]^2 = v0^2 + (norm_sq - alpha^2)
        v_norm_sq = v0 * v0 + (norm_sq - alpha * alpha)
        tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)

        safe_v0 = tl.where(tl.abs(v0) < EPS_V0, 1.0, v0)
        # u_sub[i] = col[i]/v0 for i>j ; u[j] = 1 (implicit). 0 elsewhere.
        u_sub = tl.where(rows > j, col / safe_v0, 0.0)        # (M_POW2,)
        # Full reflector vector u with u[j] = 1 (used in the rank-1 apply below).
        u_full = tl.where(is_j_row, 1.0, u_sub)              # (M_POW2,)

        # ── Apply F = I - tau*u*u^T to the trailing columns c in (j, b) ──────
        # For every trailing column at once:
        #   w_c = u^T panel[:, c]   (over rows [j, m))
        #   panel[:, c] -= tau * w_c * u
        # w over rows >= j; u_full already encodes u[j]=1, u[i>j]=u_sub[i], 0 above.
        w = tl.sum(u_full[:, None] * panel, axis=0)          # (B_POW2,) = u^T @ panel
        update = (tau_j * u_full)[:, None] * w[None, :]       # (M_POW2, B_POW2)
        trailing = cols[None, :] > j                          # only cols after j
        panel = tl.where(trailing, panel - update, panel)

        # ── Write the factored column j into the tile: H[j,j]=beta, H[i>j,j]=u_sub
        # IMPORTANT: only touch rows >= j. Rows i<j of column j hold the R (upper-
        # triangle) entries produced by earlier steps' trailing updates and MUST be
        # preserved (v9 stores column j only over rows [j, m)).
        new_colj = tl.where(is_j_row, beta, u_sub)           # (M_POW2,), 0 for rows<j
        write_colj = (cols[None, :] == j) & (rows[:, None] >= j)
        panel = tl.where(write_colj, new_colj[:, None], panel)

        # tau for this column (scalar store).
        tl.store(tau_ptr + bid * stride_tb + (k_start + j) * stride_tc, tau_j)

    # ── Write the fully factored panel back to global ONCE ───────────────────
    tl.store(tile_ptr, panel, mask=tile_mask)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# ────────────────────────────────────────────────────────────────────────────
# WY build + trailing update — pure torch (same math as v1 / v9).
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
    # per call (next_pow2 of the full height) so each matrix size compiles the panel
    # kernel exactly once (masking covers shorter late blocks). 3 distinct values
    # across the dispatch range (256, 512, 1024) -> 3 compiles, same as v9.
    MP = _next_pow2(n)
    BP = _next_pow2(B)
    # More warps for taller resident tiles so fewer rows map to each thread
    # (the (M_POW2 x B_POW2) tile lives in registers/on chip; few warps over a
    # tall tile => high per-thread register pressure + slow reductions).
    nwarps = 4 if MP <= 256 else (8 if MP <= 512 else 16)
    for k in range(0, n, B):
        b = min(B, n - k)
        k_end = k + b
        m = n - k
        # Fused panel factorization: ONE Triton launch; the panel is resident on chip.
        _panel_qr_kernel[grid](
            H, tau_all,
            k, b, m,
            H.stride(0), H.stride(1), H.stride(2),
            tau_all.stride(0), tau_all.stride(1),
            M_POW2=MP,
            B_POW2=BP,
            EPS_V0=_EPS_V0,
            num_warps=nwarps,
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
    # the resident-panel Triton kernel + batched bmm wins (headline b640 n512).
    # Tiny n=32 and small-batch huge-n (2048/4096) lose to cuSOLVER regardless
    # (findings E1), so route those to torch.geqrf.
    return batch >= 32 and 128 <= n <= 1024


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if _use_triton(batch, n):
        return _blocked_wy_triton(data)
    return torch.geqrf(data)
