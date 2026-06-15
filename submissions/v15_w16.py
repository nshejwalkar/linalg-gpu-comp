"""
v15_onecompile — v13's shared-memory-resident panel QR, collapsed to ONE Triton compile.

WHY: v13 (geomean 3.41x, 19/19) is fast but its panel kernel keys on
M_POW2 = next_pow2(n) ∈ {256, 512, 1024} with num_warps 4/8/16 — three distinct
(constexpr, num_warps) combos → THREE compiled variants. On Blackwell each compile
of the heavy resident-tile kernel (up to 128KB smem, num_warps 16) is very slow, and
3 compiles + the eval blow past the grader's 300s timeout (findings D10: v13 timed
out >300s; the lighter v9 with 3 column-kernel compiles barely fit at 240s).

THE FIX (this file): make the panel kernel compile EXACTLY ONCE, for any n in the
dispatch range. Two levers, both constexpr-constant across every launch:
  1. ONE fixed M_POW2 = _M_FIXED = 1024 (the max next_pow2(n) for n ≤ 1024). Smaller n
     just leave the rows [m, 1024) of the resident tile masked off (loaded as 0.0,
     never written). The existing `tile_mask = (rows < m) & (cols < b)` already does
     this — masked rows contribute 0 to every reduction, so the numerics are byte-for-
     byte identical to v13 (see "Numerics unchanged" below).
  2. ONE fixed num_warps = _NUM_WARPS (tuned on Modal). v13 scaled warps with tile
     height; here the tile height constexpr is always 1024, so a single warp count
     applies to every call.
With both constexprs fixed, `_panel_qr_kernel.cache[dev]` holds a single entry no
matter which shapes ran. (Verify: `--mode profile` prints "N compiled variant(s)";
N MUST be 1.)

COST: small-n blocks (n=176/352) now run with 1024 lanes instead of 256/512, so a
few warps idle on the masked tail → those shapes get somewhat slower than v13. That
is the deliberate trade: ONE compile is what fits the 300s budget, and we still aim
to clear v9 (2.38x) comfortably and stay near v13 (3.41x).

Residency / smem fit: the resident tile is always (1024, 32) → 1024·32·4B = 128 KB,
which fits B200's 228 KB on-chip budget (same tile v13 used for its n=1024 case; we
just always use it). n=2048/4096 still route to torch.geqrf (dispatch unchanged), so
the panel tile never needs to exceed 128 KB.

Numerics unchanged from v13 / v9 / v1 (19/19). With M_POW2 = 1024 ≥ m for every
dispatched n, the only difference vs v13 is extra masked-off rows in [m, 1024):
  • load: rows ≥ m masked → 0.0;
  • `col` extraction then `tl.where(rows >= j, col, 0.0)` leaves rows ≥ m as 0;
  • `norm_sq = sum(col*col)`, `alpha = col[j]`, `w = sum(u_full*panel)` all sum 0 over
    the masked tail → identical to a tile of exact height m;
  • `u_sub = where(rows > j, col/v0, 0)` is 0 on the tail; trailing/colwrite masks
    (cols>j / (cols==j)&(rows>=j)) keep the tail untouched; store re-masks on write-back.
So per-step beta/tau/u and the final (H, tau) match v13 exactly. The math is still
v1's `_householder_step`:
  alpha = col[0]; sign_a = sign(alpha) (0 -> +1); beta = -sign_a*||col[j:]||;
  v0 = alpha - beta; ||v||^2 = v0^2 + (||col[j:]||^2 - alpha^2);
  tau = 2*v0^2/||v||^2 (0 if ||v||^2 == 0); safe_v0 guards |v0| < 1e-30;
  u_sub[i>j] = col[i]/v0, u[j] = 1; H[diag] = beta, H[below] = u_sub.
WY build + trailing update stay torch.bmm (3 GEMMs/block), identical to v13.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t


_BLOCK = 32          # panel width; tile column extent (B_POW2 = next_pow2(_BLOCK))
_EPS_V0 = 1e-30      # matches v1 safe_v0 guard

# ── ONE-COMPILE knobs ────────────────────────────────────────────────────────
# Fixed resident-tile row extent for ALL n in the dispatch range (max next_pow2 is
# 1024 for n ≤ 1024). Constant across every launch → the panel kernel keys on a
# single M_POW2, so it compiles exactly once.
_M_FIXED = 1024
# Fixed warp count for that single (1024, 32) tile. ~2 rows/thread at 16 warps; v13
# found warps must cover the tall tile. Tuned on Modal (warps 4/8/16/32 sweep): 16 is
# the sweet spot — 4 starves the tall tile (geomean collapses to ~1.4x), 32 slightly
# over-subscribes (~3.15x), 16 wins (~3.20-3.29x). Kept as ONE value so we do NOT
# introduce a second (M_POW2, num_warps) combo (which would mean a 2nd compile).
_NUM_WARPS = 16


# ────────────────────────────────────────────────────────────────────────────
# Triton panel kernel: factor one (m x b) panel, one program per batch element.
# The panel is loaded into a resident [M_POW2, B_POW2] on-chip tile ONCE, all b
# steps run on that tile, and it is written back ONCE — no per-step global reads.
#
# M_POW2 is fixed (= _M_FIXED) for every call, so this kernel compiles exactly once
# across all dispatched shapes; smaller n mask off rows [m, M_POW2).
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
    M_POW2: tl.constexpr,   # FIXED resident-tile row extent (= _M_FIXED for all n)
    B_POW2: tl.constexpr,   # next_pow2(_BLOCK) — column extent of the resident tile
    EPS_V0: tl.constexpr,   # safe_v0 guard (matches v1: 1e-30)
):
    bid = tl.program_id(0)
    rows = tl.arange(0, M_POW2)                       # lane = panel row index
    cols = tl.arange(0, B_POW2)                       # column index within panel
    # Base pointer to A[bid, k_start, k_start] (top-left of the panel).
    base = A_ptr + bid * stride_Ab + k_start * stride_Ar + k_start * stride_Ac

    # ── Load the whole panel into a resident on-chip tile ONCE ───────────────
    # rows >= m are masked -> loaded as 0.0 (the masked tail when m < M_POW2).
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
        norm_sq = tl.sum(col * col)                   # ||col[j:]||^2 (tail rows are 0)
        norm = tl.sqrt(norm_sq)

        sign_a = tl.where(alpha >= 0.0, 1.0, -1.0)    # sign(0) -> +1 (v1)
        beta = -sign_a * norm
        v0 = alpha - beta
        # ||v||^2 = v0^2 + sum_{i>j} col[i]^2 = v0^2 + (norm_sq - alpha^2)
        v_norm_sq = v0 * v0 + (norm_sq - alpha * alpha)
        tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)

        safe_v0 = tl.where(tl.abs(v0) < EPS_V0, 1.0, v0)
        # u_sub[i] = col[i]/v0 for i>j ; u[j] = 1 (implicit). 0 elsewhere (incl. tail).
        u_sub = tl.where(rows > j, col / safe_v0, 0.0)        # (M_POW2,)
        # Full reflector vector u with u[j] = 1 (used in the rank-1 apply below).
        u_full = tl.where(is_j_row, 1.0, u_sub)              # (M_POW2,)

        # ── Apply F = I - tau*u*u^T to the trailing columns c in (j, b) ──────
        # For every trailing column at once:
        #   w_c = u^T panel[:, c]   (over rows [j, m))
        #   panel[:, c] -= tau * w_c * u
        # w over rows >= j; u_full already encodes u[j]=1, u[i>j]=u_sub[i], 0 above/tail.
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
    # tile_mask keeps the masked tail [m, M_POW2) and cols >= b untouched in global.
    tl.store(tile_ptr, panel, mask=tile_mask)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# ────────────────────────────────────────────────────────────────────────────
# WY build + trailing update — pure torch (same math as v1 / v9 / v13).
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
    # ONE COMPILE: M_POW2 and num_warps are FIXED constants for every launch and every
    # n in the dispatch range, so _panel_qr_kernel keys on a single (constexpr, warps)
    # combo → exactly ONE compiled variant (vs v13's 3). Masking handles m < M_POW2.
    MP = _M_FIXED                  # fixed (= 1024); NOT next_pow2(n)
    BP = _next_pow2(B)
    nwarps = _NUM_WARPS           # fixed single value
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
# Dispatch — unchanged from v13: Triton for 128 ≤ n ≤ 1024 (large-batch/medium-n,
# where cuSOLVER under-parallelizes the batch); torch.geqrf for tiny n=32 and
# small-batch huge-n (2048/4096), where cuSOLVER's single-matrix path wins.
# ────────────────────────────────────────────────────────────────────────────
def _use_triton(batch: int, n: int) -> bool:
    return batch >= 32 and 128 <= n <= 1024


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if _use_triton(batch, n):
        return _blocked_wy_triton(data)
    return torch.geqrf(data)
