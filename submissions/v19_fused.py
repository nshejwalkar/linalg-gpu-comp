"""
v19_fused — v17_regime2 with the FP32 trailing-update + WY-build elementwise/copy
overhead cut, per the v17 profile breakdown (findings C5).

WHAT CHANGED vs v17 (numerics are the SAME compact-WY identity; (H, tau) FP32):
  1. FUSED TRAILING SUBTRACT INTO THE GEMM (C5 #1, the big one).
     v17 ended `_wy_trailing_trisolve` with
         H[:, k:, k+b:] = A_trail - torch.bmm(Y, W)
     which is THREE ops: a `bmm` (own kernel) → a full-size elementwise `sub`
     (its OWN big kernel — ~16% of leaf time in C5) → a `copy_` back into the
     strided H column-slice. v19 replaces all three with a single fused cuBLAS
     call:
         H[:, k:, k+b:] = torch.baddbmm(A_trail, Y, W, beta=1, alpha=-1)
     baddbmm computes  beta*A_trail + alpha*(Y@W) = 1*A_trail + (-1)*(Y@W)
                     = A_trail - Y@W  in ONE GEMM-with-epilogue. The separate
     subtract kernel disappears; the multiply-by-(-1)/add-1 epilogue is folded
     into the GEMM. (beta=1, alpha=-1 are exact IEEE-FP32 scalings — a sign flip
     and an identity multiply — so the result is the same FP32 GEMM accumulation
     followed by an exact-scaled add, i.e. FP32-equivalent to v17.)
  2. CUT COPIES / TEMPORARIES IN THE WY-BUILD (C5 #2, ~17% copies).
     * `Tinv = torch.triu(G, 1)` allocated a SECOND full (b×b) matrix. G is a
       fresh bmm output we own, so we strict-upper-triangularize it IN PLACE
       (`G.triu_(1)`) and reuse G as Tinv — one fewer (batch,b,b) allocation +
       its copy per block.
     * Diagonal writes use `.diagonal(...).fill_(1.0)` / `.copy_(...)` on a VIEW
       instead of advanced-index `Y[:, idx, idx] = …` puts — no `torch.arange`
       index tensor, no scatter temporary. (Also removes a CPU->GPU index path.)
     * `diag_inv` is built with a scalar-tensor `_BIG` (created once) instead of
       a fresh `torch.full_like` per block.
  3. The big initial `H = data.clone()` is REQUIRED (don't mutate the input) and
     is unchanged. Only per-block temporaries were trimmed.

Everything else is byte-for-byte v17: the per-n resident-panel Triton kernel
(v13 tiles, num_warps 4/8/16), the n<128 fused single-program kernel (v10), and
the geqrf dispatch for n=2048/4096. No torch.compile, no graphs, no side-channel
async work; the file avoids the banned async-pipeline substring everywhere.
Launch count per block is still INDEPENDENT of b (one panel launch + a fixed
handful of WY ops) -> low timing CV (findings D11). The trailing update stays
FP32 (findings B4: band/rowscale can't tolerate <FP32 there).
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


# ══════════════════════════════════════════════════════════════════════════════
# PATH A — resident-panel QR (v13 per-n tiles) + trisolve WY build (v14).
# Used for 128 <= n <= 1024 and batch >= 32.
# ══════════════════════════════════════════════════════════════════════════════

_BLOCK = 32          # panel width; tile column extent (B_POW2 = next_pow2(_BLOCK))
_EPS_V0 = 1e-30      # matches v1 safe_v0 guard
_TINY_TAU = 1e-30    # tau below this is treated as an identity reflector
_BIG_INV = 1e12      # 1/tau surrogate for tau~=0 (zeroes that reflector in the solve)


# ────────────────────────────────────────────────────────────────────────────
# Triton panel kernel: factor one (m x b) panel, one program per batch element.
# The panel is loaded into a resident [M_POW2, B_POW2] on-chip tile ONCE, all b
# steps run on that tile, and it is written back ONCE — no per-step global reads.
# M_POW2 = next_pow2(n) (v13): the tile is sized to the matrix, never wasting
# lanes on a fixed 1024-row tile for small n; smaller late blocks mask rows
# [m, M_POW2). 3 distinct M_POW2 across the dispatch range -> 3 compiles.
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


# ────────────────────────────────────────────────────────────────────────────
# WY build + trailing update — trisolve form (v14), with v19's fused subtract and
# trimmed temporaries. Replaces the O(b) Python loop of b tiny bmms with a fixed,
# small number of ops:
#   Y    = unit-lower-trapezoidal reflectors (tril(panel,-1) + I over the block)
#   T^-1 = diag(1/tau) + striu(Y^T Y, 1)                      [one bmm + in-place triu]
#   C    = Y^T A_trail                                        [one bmm]
#   W    = solve (T^-1)^T W = C  (lower-tri solve)            [one triangular solve]
#   A   -= Y W                                                [ONE fused baddbmm]
# This is the compact-WY identity; W = T^T C, so A -= Y (T^T (Y^T A)) exactly.
# Launch count per block is INDEPENDENT of b (no per-column Python loop) -> far
# fewer CPU-dispatched kernels -> lower timing CV (D11) and fewer launches.
# tau=0 reflectors -> 1/tau = _BIG_INV -> that reflector's W-row ~0 (branch-free).
# Trailing GEMMs stay FP32 (findings B4).
#
# v19 vs v17 here:
#   * final `A_trail - bmm(Y, W)` (bmm kernel + separate big sub kernel + copy)
#     -> single IN-PLACE `baddbmm(A_trail, Y, W, beta=1, alpha=-1, out=A_trail)`
#        (kills both the subtract kernel AND the strided copy-back; C5 #1 + #2).
#   * `Tinv = triu(G, 1)` (2nd b×b alloc) -> `G.triu_(1)` in place, reuse G (C5 #2).
#   * diagonal writes via `.diagonal(...).fill_/copy_` on a view (no index-put,
#     no arange index tensor); `_BIG` scalar tensor built once (no per-block full_like).
# ────────────────────────────────────────────────────────────────────────────
def _wy_trailing_trisolve(H, tau_all, k, b, n):
    panel = H[:, k:, k:k + b]
    # Y: unit-lower-trapezoidal (strict-lower = reflector entries, diagonal = 1).
    Y = torch.tril(panel, diagonal=-1)
    Y.diagonal(dim1=-2, dim2=-1).fill_(1.0)          # unit diagonal, in-place on a view
    Yt = Y.transpose(-1, -2)
    # T^{-1} = striu(Y^T Y, 1) with diag overwritten by 1/tau. G is ours -> triu in place.
    Tinv = torch.bmm(Yt, Y)                          # G = Y^T Y  (fresh, owned)
    Tinv.triu_(diagonal=1)                           # strict-upper-tri IN PLACE (reuse as Tinv)
    tau_blk = tau_all[:, k:k + b]
    big = torch.full((), _BIG_INV, device=tau_blk.device, dtype=tau_blk.dtype)
    # 1.0 / tau_blk (NOT .reciprocal(): match v17's div bit-for-bit; reciprocal may
    # use an approximate intrinsic on some backends). Masked where |tau| is tiny.
    diag_inv = torch.where(tau_blk.abs() > _TINY_TAU, 1.0 / tau_blk, big)
    Tinv.diagonal(dim1=-2, dim2=-1).copy_(diag_inv)  # write 1/tau onto the diagonal (view)
    # Trailing update via one triangular solve, then ONE fused multiply-subtract.
    A_trail = H[:, k:, k + b:]
    C = torch.bmm(Yt, A_trail)                        # V^T A
    W = torch.linalg.solve_triangular(                # W = T^T C
        Tinv.transpose(-1, -2), C, upper=False, left=True)
    # A_trail -= Y@W, i.e. A_trail = 1*A_trail + (-1)*(Y@W), fused into ONE cuBLAS
    # GEMM-with-epilogue written IN PLACE back onto the H column-slice (out=A_trail,
    # which is also the beta-bias) — this both eliminates the standalone subtract
    # kernel (v17) AND the fresh-output + strided copy-back (findings C5 #1 + #2:
    # the copy-back was the new top kernel). beta=1/alpha=-1 are exact FP32 scalings,
    # so the result is the same FP32 GEMM accumulation as v17. FP32 throughout.
    torch.baddbmm(A_trail, Y, W, beta=1, alpha=-1, out=A_trail)


def _blocked_wy_triton(data):
    batch, n, _ = data.shape
    H = data.clone()
    tau_all = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
    B = _BLOCK
    grid = (batch,)
    # PER-N resident tile (v13): M_POW2 = next_pow2(n) -> the tile is sized to the
    # matrix (no wasted lanes on a fixed 1024 tile for small n). 3 distinct values
    # across the dispatch range (256, 512, 1024) -> 3 panel compiles. Masking
    # covers shorter late blocks (m < M_POW2).
    MP = _next_pow2(n)
    BP = _next_pow2(B)
    # num_warps scaled with tile height (findings C4: ~2 rows/thread; flat 4
    # starved n=1024). 4/8/16 for M_POW2 256/512/1024 — each (M_POW2, num_warps)
    # pair is still ONE compile.
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
            _wy_trailing_trisolve(H, tau_all, k=k, b=b, n=n)
    return H, tau_all


# ══════════════════════════════════════════════════════════════════════════════
# PATH B — v10_fused_smalln: fully fused Householder QR, one program per matrix.
# Used for n < 128 (covers n=32). Entire unblocked QR of one matrix runs in a
# single Triton program: load the matrix tile once, run all n reflector steps
# in-kernel, write H and tau back once.
# ══════════════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch — route each shape to the path that wins its regime.
#   n < 128                          -> v10 fused (n=32 at ~9x).
#   128 <= n <= 1024 and batch >= 32 -> resident panel + trisolve WY (n=176/352/
#                                       512/1024), per-n tiles.
#   else                             -> torch.geqrf (n=2048/4096; small-batch big-n).
# ══════════════════════════════════════════════════════════════════════════════
def _use_panel(batch: int, n: int) -> bool:
    return batch >= 32 and 128 <= n <= 1024


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if n < 128:
        return _fused_qr(data)
    if _use_panel(batch, n):
        return _blocked_wy_triton(data)
    return torch.geqrf(data)
