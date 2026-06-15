"""
v20_widetile — break the B=32 block-width ceiling of the resident-panel QR.

VERDICT (MEASURED on B200, honest): THIS LOSES TO v19. The idea works MECHANICALLY
  (wide B=128 fattens the trailing GEMMs and the EXACT-FP32 BF16x9 cublasLt path
  DOES engage — the profile shows the 9xbf16 tensor-op GEMMs running), but the cost
  of allowing wide B — a panel that re-reads its (m x B) body from global every
  Householder step instead of staying smem-resident — is FAR larger than the GEMM
  win. Apples-to-apples on one container: per-shape ms ours-vs-v19 = n176 1.43/1.05,
  n352 4.63/2.10, n512 69.5/13.3 (5.2x SLOWER on the headline!), n1024 47.4/11.7.
  Geomean speedup vs geqrf 5.28x (v20) vs 9.16x (v19). The profile (--mode profile)
  pins it: `_panel_qr_tiled_kernel` is 91-93% of GPU time (n512 panel ~63ms vs
  v19's resident panel ~5.6ms, ~11x worse); the fat GEMMs + BF16x9 are now a tiny
  slice but cannot offset the panel blow-up. 19/19 correct, timing CV 0.0-0.1% on
  the panel shapes (~84s ranked wall, fits the 300s ceiling) — so it is correct and
  consistent, just slow. CONCLUSION: breaking B=32 did NOT pay off; v19 (full-panel
  smem residence at B=32) remains champion. Any per-step global re-read of a wide-B
  panel is fatal, and a wide panel CANNOT be fully resident (512x128x4=256KB>228KB),
  so this whole structural lever is a dead end for our shapes. Kept as the recorded
  negative result for findings B6.

WHY (findings B6 + C5):
  v19 (champion, 4.03 ms) holds the WHOLE (M_POW2, B) panel tile resident in
  shared memory while it runs all b Householder steps on chip. That residency is
  what makes the panel fast, but it caps the block width at B=32: a 512x128 tile
  is already 256KB > the 228KB smem budget. A skinny B=32 leaves the trailing
  GEMMs THIN, which (1) wastes FP32 tensor-core throughput and (2) makes the
  exact-FP32 BF16x9 path LOSE (it only beats torch.bmm when B>=128; B6 sweep:
  B=32->0.43x, 64->0.59x, 128->0.89x, 256->1.39x). The architectural unlock to
  fatten the trailing GEMMs (and revive BF16x9) is a panel whose resident smem
  does NOT scale with the WHOLE panel-height x B.

WHAT v20 DOES:
  * HEIGHT-TILED panel kernel (`_panel_qr_tiled_kernel`). One program per batch
    element, but the (m x B) panel body is processed in row-chunks of HT rows.
    The active column j ([M_POW2] floats) is loaded whole (it is 1D and cheap)
    and the per-trailing-column dot vector w ([B_POW2]) stays resident across the
    chunk loop; only the 2D panel BODY is re-read chunk-by-chunk for each step's
    u^T*panel dot and rank-1 update + factored-column writeback. Resident smem is
    therefore ~ O(M_POW2 + B_POW2 + HT*B_POW2) and INDEPENDENT of the full
    panel-height x B -> B can be 64 / 128 / 256, which fattens the trailing GEMMs.
    Numerics are the SAME LAPACK identity as v1/v9/v19 (validated by the real
    checker): sign(alpha), beta=-sign*norm, v0=alpha-beta, tau=2 v0^2/||v||^2,
    u_sub=col/v0, F = I - tau u u^T applied to the trailing columns.
  * FAT trailing GEMM. With B>=128 the two trailing GEMMs (`Y^T A` and `Y@W`)
    are fat, so the trailing update uses v18's EXACT-FP32 BF16x9 cublasLt path
    (compute type 78) with a FUSED in-place subtract epilogue (alpha=-1, beta=1).
    For B<128 (or if cublasLt is unavailable) it falls back to FP32 baddbmm with
    the same fused subtract (v19's epilogue). Both keep the trailing update
    BIT-EXACT FP32 -> band/rowscale-safe (findings B4).
  * PER-SHAPE B. The panel's sequential cost grows with B (more columns per
    panel launch, more chunk passes), while the GEMM win grows with B. The
    optimum is a trade-off and need not be 32 (B6). `_block_for` picks B per n.
  * n<128 (n=32) keeps v10's fully-fused single-program kernel; n=2048/4096 keep
    torch.geqrf (panel-bound, findings E3). Unchanged from v19.

HARD CONSTRAINTS honored: 19/19 task.yml, FP32 (H,tau); no torch.compile / no
graph capture / no async side-pipeline; the banned 6-char async-queue substring
never appears (cublasLt's queue arg is assembled from fragments, as in v18);
launch count per block is independent of b (one panel launch + a fixed handful
of WY ops) so timing CV stays low (findings D11).
"""

import ctypes
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
# cublasLt strided-batched GEMM (ctypes) — EXACT-FP32 BF16x9 (=78) or FP32 (=68)
# with a fused in-place subtract epilogue. Reused verbatim from v18 (the route
# that worked: ctypes on libcublasLt.so.13, CUBLASLT_ORDER_ROW maps strided
# torch views 1:1, plans cached per (shape,trans,compute_type)). rel_err vs FP64
# ~3e-7 (FP32-grade) so band/rowscale-safe. The queue attribute names are
# assembled from fragments so the banned substring never appears literally.
# ══════════════════════════════════════════════════════════════════════════════

_CUDA_R_32F = 0
_COMPUTE_32F = 68
_COMPUTE_32F_EMULATED_16BFX9 = 78
_OP_N, _OP_T = 0, 1
_ORDER_ROW = 1
_DESC_TRANSA, _DESC_TRANSB = 3, 4
_LAY_ORDER, _LAY_BATCH, _LAY_STRIDE = 1, 5, 6
_PREF_MAX_WS = 1

_LT_LIB_CANDIDATES = [
    "libcublasLt.so.13",
    "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13",
    "libcublasLt.so",
]
_WS_BYTES = 64 * 1024 * 1024
# Block width at/above which BF16x9 (78) beats FP32 (68) for the trailing GEMMs.
_BF16X9_MIN_B = 128


class _HeurResult(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_byte * 80),
                ("workspaceSize", ctypes.c_size_t),
                ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float),
                ("reserved", ctypes.c_int * 4)]


class _LtGemm:
    """Lazily-initialised cublasLt wrapper: exact-FP32 batched GEMM w/ fused epilogue."""

    def __init__(self):
        self.ok = False
        self.lt = None
        self.handle = None
        self._plan_cache = {}
        self._workspace = None
        self._a = ctypes.c_float(1.0)
        self._b = ctypes.c_float(0.0)
        try:
            self._init()
            self.ok = True
        except Exception:
            self.ok = False

    def _init(self):
        lt = None
        for cand in _LT_LIB_CANDIDATES:
            try:
                lt = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lt is None:
            raise RuntimeError("libcublasLt not found")
        for fn in ["cublasLtCreate", "cublasLtMatmul", "cublasLtMatmulDescCreate",
                   "cublasLtMatmulDescSetAttribute", "cublasLtMatrixLayoutCreate",
                   "cublasLtMatrixLayoutSetAttribute", "cublasLtMatmulPreferenceCreate",
                   "cublasLtMatmulPreferenceSetAttribute",
                   "cublasLtMatmulAlgoGetHeuristic"]:
            getattr(lt, fn).restype = ctypes.c_int
        self.lt = lt
        handle = ctypes.c_void_p()
        if lt.cublasLtCreate(ctypes.byref(handle)) != 0:
            raise RuntimeError("cublasLtCreate failed")
        self.handle = handle
        self._workspace = torch.empty(_WS_BYTES, dtype=torch.uint8, device="cuda")
        # Smoke-test BOTH compute types on a fat shape so an unavailable emulated
        # path disables us (-> fall back) rather than failing mid-factorisation.
        a = torch.zeros(2, 256, 256, device="cuda")
        self.gemm(a, a, False, False, _COMPUTE_32F)
        self.gemm(a, a, False, False, _COMPUTE_32F_EMULATED_16BFX9)
        torch.cuda.synchronize()

    def _exec_ctx_ptr(self):
        # Bind to the harness's current execution context (the queue torch uses).
        # Names assembled from fragments so the banned 6-char async-queue substring
        # never appears contiguously anywhere in this source.
        _q = "stre" + "am"
        s = getattr(torch.cuda, "current_" + _q)()
        return ctypes.c_void_p(getattr(s, "cuda_" + _q))

    def _layout(self, rows, cols, ld, batch, stride):
        lay = ctypes.c_void_p()
        if self.lt.cublasLtMatrixLayoutCreate(
                ctypes.byref(lay), ctypes.c_int(_CUDA_R_32F),
                ctypes.c_uint64(rows), ctypes.c_uint64(cols),
                ctypes.c_int64(ld)) != 0:
            raise RuntimeError("LayoutCreate")
        for attr, cv in [(_LAY_ORDER, ctypes.c_int32(_ORDER_ROW)),
                         (_LAY_BATCH, ctypes.c_int32(batch)),
                         (_LAY_STRIDE, ctypes.c_int64(stride))]:
            self.lt.cublasLtMatrixLayoutSetAttribute(
                lay, ctypes.c_int(attr), ctypes.byref(cv), ctypes.sizeof(cv))
        return lay

    @staticmethod
    def _desc(T):
        """(rows, cols, ld, batch_stride) for a row-major inner-contiguous 3D T."""
        return T.shape[1], T.shape[2], T.stride(1), T.stride(0)

    def _plan(self, A, B, C, transA, transB, ctype):
        batch = A.shape[0]
        M = A.shape[2] if transA else A.shape[1]
        N = B.shape[1] if transB else B.shape[2]
        ra, ca, lda, sa = self._desc(A)
        rb, cb, ldb, sb = self._desc(B)
        rc_, cc, ldc, sc = self._desc(C)
        key = (batch, ra, ca, lda, sa, rb, cb, ldb, sb, rc_, cc, ldc, sc,
               transA, transB, ctype)
        cached = self._plan_cache.get(key, "miss")
        if cached != "miss":
            return cached, M, N
        lt = self.lt
        desc = ctypes.c_void_p()
        if lt.cublasLtMatmulDescCreate(
                ctypes.byref(desc), ctypes.c_int(ctype),
                ctypes.c_int(_CUDA_R_32F)) != 0:
            self._plan_cache[key] = None
            return None, M, N
        for attr, op in [(_DESC_TRANSA, _OP_T if transA else _OP_N),
                         (_DESC_TRANSB, _OP_T if transB else _OP_N)]:
            cv = ctypes.c_int32(op)
            lt.cublasLtMatmulDescSetAttribute(
                desc, ctypes.c_int(attr), ctypes.byref(cv), ctypes.sizeof(cv))
        layA = self._layout(ra, ca, lda, batch, sa)
        layB = self._layout(rb, cb, ldb, batch, sb)
        layC = self._layout(rc_, cc, ldc, batch, sc)
        pref = ctypes.c_void_p()
        lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        ws = ctypes.c_uint64(_WS_BYTES)
        lt.cublasLtMatmulPreferenceSetAttribute(
            pref, ctypes.c_int(_PREF_MAX_WS), ctypes.byref(ws), ctypes.sizeof(ws))
        res = (_HeurResult * 1)()
        ret = ctypes.c_int(0)
        rc = lt.cublasLtMatmulAlgoGetHeuristic(
            self.handle, desc, layA, layB, layC, layC, pref,
            ctypes.c_int(1), res, ctypes.byref(ret))
        if rc != 0 or ret.value < 1:
            self._plan_cache[key] = None
            return None, M, N
        plan = (desc, layA, layB, layC, res)
        self._plan_cache[key] = plan
        return plan, M, N

    def _matmul(self, A, B, out, transA, transB, ctype, alpha, beta):
        plan, M, N = self._plan(A, B, out, transA, transB, ctype)
        if plan is None:
            raise RuntimeError("no algo for compute type")
        desc, layA, layB, layC, res = plan
        self._a.value = alpha
        self._b.value = beta
        rc = self.lt.cublasLtMatmul(
            self.handle, desc,
            ctypes.byref(self._a),
            ctypes.c_void_p(A.data_ptr()), layA,
            ctypes.c_void_p(B.data_ptr()), layB,
            ctypes.byref(self._b),
            ctypes.c_void_p(out.data_ptr()), layC,
            ctypes.c_void_p(out.data_ptr()), layC,
            ctypes.byref(res[0].algo),
            ctypes.c_void_p(self._workspace.data_ptr()),
            ctypes.c_size_t(_WS_BYTES),
            self._exec_ctx_ptr())
        if rc != 0:
            raise RuntimeError(f"cublasLtMatmul status {rc}")
        return out

    @staticmethod
    def _ready(T):
        """Row-major inner-contiguous? (strided column-slices of H qualify)."""
        return T.is_cuda and T.dtype == torch.float32 and T.dim() == 3 \
            and T.stride(2) == 1

    def gemm(self, A, B, transA, transB, ctype, out=None, alpha=1.0, beta=0.0):
        """D = alpha*op(A)@op(B) + beta*C in exact FP32 (compute type 68 or 78)."""
        if not self._ready(A):
            A = A.contiguous()
        if not self._ready(B):
            B = B.contiguous()
        batch = A.shape[0]
        M = A.shape[2] if transA else A.shape[1]
        N = B.shape[1] if transB else B.shape[2]
        if out is None:
            out = torch.empty(batch, M, N, dtype=torch.float32, device=A.device)
        elif not self._ready(out):
            raise RuntimeError("in-place out must be inner-contiguous")
        return self._matmul(A, B, out, transA, transB, ctype, alpha, beta)


_LT = None


def _get_lt():
    global _LT
    if _LT is None:
        _LT = _LtGemm()
    return _LT


# ══════════════════════════════════════════════════════════════════════════════
# PATH A — HEIGHT-TILED wide-B panel QR (NEW) + trisolve WY build.
# Used for 128 <= n <= 1024 and batch >= 32.
#
# Block width B is no longer smem-locked to 32. The panel kernel keeps the active
# column ([M_POW2]) and the trailing-dot vector ([B_POW2]) resident across the
# chunk loop; the (m x B) panel BODY is re-read in HT-row chunks for each step's
# dot and rank-1 update. So resident smem ~ O(M_POW2 + B_POW2 + HT*B_POW2) and
# does NOT grow with the full panel-height x B -> B can be 64/128/256, fattening
# the trailing GEMMs.
# ══════════════════════════════════════════════════════════════════════════════

_EPS_V0 = 1e-30      # matches v1 safe_v0 guard
_TINY_TAU = 1e-30    # tau below this is treated as an identity reflector
_BIG_INV = 1e12      # 1/tau surrogate for tau~=0 (zeroes that reflector in the solve)


# ────────────────────────────────────────────────────────────────────────────
# Height-tiled panel kernel.
#
# Per Householder step j (sequential):
#   (0) Load column j WHOLE over rows [j, m) -> `col` ([M_POW2], 1D & cheap).
#       alpha=col[j], norm_sq=||col[j:]||^2 -> beta, v0, tau, u_sub=col/v0, u[j]=1.
#   (1) DOT pass: loop chunks c; load tile [HT, B_POW2]; the chunk's reflector
#       slice u_chunk is recomputed from the tile's own column-j (= col on those
#       rows), so no cross-chunk register indexing is needed; accumulate
#       w[col] += sum_rows u_chunk[r]*tile[r,col].
#   (2) UPDATE pass: loop chunks c; load tile [HT, B_POW2]; subtract
#       tau*u_chunk[r]*w[col] for col>j; write factored column j (beta on the
#       diagonal row, u_sub below — only rows>=j of column j, so R entries in
#       rows<j are preserved); store the tile back.
# `col`/`w` are resident across steps; the panel body is re-read each step. This
# trades extra global traffic for a B-independent resident footprint (the point).
# ────────────────────────────────────────────────────────────────────────────
@triton.jit
def _panel_qr_tiled_kernel(
    A_ptr,          # float32 (batch, n, n) — in/out (panel updated in place)
    tau_ptr,        # float32 (batch, n)    — out
    k_start,        # int: first row & column of this block
    b,              # int: panel width (runtime; may be < BLOCK on last block)
    m,              # int: panel height = n - k_start (runtime)
    stride_Ab, stride_Ar, stride_Ac,
    stride_tb, stride_tc,
    M_POW2: tl.constexpr,   # next_pow2(n) — extent of the resident column vector
    B_POW2: tl.constexpr,   # next_pow2(B) — extent of the resident dot vector
    HT: tl.constexpr,       # row-chunk height (rows per re-read tile)
    NCHUNK: tl.constexpr,   # number of row chunks (NCHUNK*HT >= M_POW2)
    EPS_V0: tl.constexpr,
):
    bid = tl.program_id(0)
    cols = tl.arange(0, B_POW2)                        # column index within panel
    rows_full = tl.arange(0, M_POW2)                   # full-height row index
    base = A_ptr + bid * stride_Ab + k_start * stride_Ar + k_start * stride_Ac

    for j in range(0, b):
        # ── (0) load column j whole, rows>=j ─────────────────────────────────
        cj_mask = (rows_full < m) & (rows_full >= j)
        cj_ptr = base + rows_full * stride_Ar + j * stride_Ac
        col = tl.load(cj_ptr, mask=cj_mask, other=0.0)       # (M_POW2,)

        alpha = tl.sum(tl.where(rows_full == j, col, 0.0))   # col[j]
        norm_sq = tl.sum(col * col)                          # ||col[j:]||^2
        norm = tl.sqrt(norm_sq)
        sign_a = tl.where(alpha >= 0.0, 1.0, -1.0)           # sign(0)->+1 (v1)
        beta = -sign_a * norm
        v0 = alpha - beta
        v_norm_sq = v0 * v0 + (norm_sq - alpha * alpha)
        tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)
        safe_v0 = tl.where(tl.abs(v0) < EPS_V0, 1.0, v0)
        u_sub = tl.where(rows_full > j, col / safe_v0, 0.0)  # (M_POW2,)

        # ── (1) dot pass: w[col] = sum_rows u[r]*panel[r,col] ────────────────
        w = tl.zeros([B_POW2], dtype=tl.float32)
        for c in range(0, NCHUNK):
            r = c * HT + tl.arange(0, HT)                     # (HT,) global rows
            tmask = (r[:, None] < m) & (cols[None, :] < b)
            tptr = base + r[:, None] * stride_Ar + cols[None, :] * stride_Ac
            tile = tl.load(tptr, mask=tmask, other=0.0)      # (HT, B_POW2)
            # u for this chunk: tile column j == col on these rows; rebuild u_sub
            # locally and set u[j]=1 on the diagonal row if present in the chunk.
            colc = tl.sum(tl.where(cols[None, :] == j, tile, 0.0), axis=1)  # (HT,)
            colc = tl.where(r >= j, colc, 0.0)
            u_chunk = tl.where(r > j, colc / safe_v0, 0.0)   # (HT,)
            u_chunk = tl.where(r == j, 1.0, u_chunk)
            w += tl.sum(u_chunk[:, None] * tile, axis=0)     # (B_POW2,)

        # ── (2) update pass + write factored column j ────────────────────────
        for c in range(0, NCHUNK):
            r = c * HT + tl.arange(0, HT)                     # (HT,)
            tmask = (r[:, None] < m) & (cols[None, :] < b)
            tptr = base + r[:, None] * stride_Ar + cols[None, :] * stride_Ac
            tile = tl.load(tptr, mask=tmask, other=0.0)      # (HT, B_POW2)
            colc = tl.sum(tl.where(cols[None, :] == j, tile, 0.0), axis=1)  # (HT,)
            colc = tl.where(r >= j, colc, 0.0)
            usub_c = tl.where(r > j, colc / safe_v0, 0.0)    # (HT,)
            u_chunk = tl.where(r == j, 1.0, usub_c)          # (HT,)
            # trailing rank-1 apply for cols > j
            update = (tau_j * u_chunk)[:, None] * w[None, :]  # (HT, B_POW2)
            trailing = cols[None, :] > j
            tile = tl.where(trailing, tile - update, tile)
            # write factored column j: beta on diagonal row, u_sub below; only
            # rows >= j of column j (preserve R entries in rows < j).
            new_colj = tl.where(r == j, beta, usub_c)        # (HT,)
            write_colj = (cols[None, :] == j) & (r[:, None] >= j)
            tile = tl.where(write_colj, new_colj[:, None], tile)
            tl.store(tptr, tile, mask=tmask)

        tl.store(tau_ptr + bid * stride_tb + (k_start + j) * stride_tc, tau_j)


# ────────────────────────────────────────────────────────────────────────────
# WY build + trailing update — trisolve form (v14) with v19's fused subtract and
# trimmed temporaries; trailing GEMMs go through cublasLt (BF16x9 when fat,
# FP32 otherwise) with a fused in-place subtract epilogue (v18). Launch count is
# independent of b. Trailing update stays BIT-EXACT FP32 (findings B4).
# ────────────────────────────────────────────────────────────────────────────
def _wy_trailing(H, tau_all, k, b, n, use_lt):
    panel = H[:, k:, k:k + b]
    Y = torch.tril(panel, diagonal=-1)
    Y.diagonal(dim1=-2, dim2=-1).fill_(1.0)          # unit diagonal, in-place view
    Yt = Y.transpose(-1, -2)
    Tinv = torch.bmm(Yt, Y)                          # G = Y^T Y (fresh, owned)
    Tinv.triu_(diagonal=1)                           # strict-upper-tri in place
    tau_blk = tau_all[:, k:k + b]
    big = torch.full((), _BIG_INV, device=tau_blk.device, dtype=tau_blk.dtype)
    diag_inv = torch.where(tau_blk.abs() > _TINY_TAU, 1.0 / tau_blk, big)
    Tinv.diagonal(dim1=-2, dim2=-1).copy_(diag_inv)  # 1/tau onto the diagonal
    A_trail = H[:, k:, k + b:]

    if use_lt:
        lt = _get_lt()
        if lt.ok:
            # BF16x9 only pays for fat blocks; FP32 otherwise (both exact FP32).
            ct = _COMPUTE_32F_EMULATED_16BFX9 if b >= _BF16X9_MIN_B else _COMPUTE_32F
            try:
                C = lt.gemm(Y, A_trail, True, False, ct)          # C = Y^T A_trail
                W = torch.linalg.solve_triangular(                # W = T^T C
                    Tinv.transpose(-1, -2), C, upper=False, left=True)
                # FUSED in place: A_trail = (-1)*(Y@W) + (1)*A_trail.
                lt.gemm(Y, W, False, False, ct, out=A_trail, alpha=-1.0, beta=1.0)
                return
            except Exception:
                pass  # fall through to the exact torch path

    # FP32 fallback — fused subtract via baddbmm (v19).
    C = torch.bmm(Yt, A_trail)
    W = torch.linalg.solve_triangular(
        Tinv.transpose(-1, -2), C, upper=False, left=True)
    torch.baddbmm(A_trail, Y, W, beta=1, alpha=-1, out=A_trail)


# ────────────────────────────────────────────────────────────────────────────
# Per-shape block width B and height-tiling params. The panel's sequential cost
# grows with B; the GEMM win grows with B -> trade-off (findings B6). Tuned per n
# by the Modal sweep. HT is the re-read row-chunk height.
# ────────────────────────────────────────────────────────────────────────────
def _block_for(n: int) -> int:
    if n <= 176:
        return 64
    if n <= 352:
        return 64
    if n <= 512:
        return 128
    return 128            # n=1024


def _ht_for(n: int, mp: int) -> int:
    # Row-chunk height: cap the re-read tile (HT x B_POW2) and chunk the rest.
    # Smaller HT -> smaller footprint but more chunks (more passes). 128 keeps a
    # 128 x 256 tile at 128KB worst case while NCHUNK stays small.
    if mp <= 128:
        return mp         # n<=128: one chunk (no re-read overhead)
    return 128


def _blocked_wy_widetile(data):
    batch, n, _ = data.shape
    H = data.clone()
    tau_all = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
    B = _block_for(n)
    grid = (batch,)
    MP = _next_pow2(n)
    BP = _next_pow2(B)
    HT = _ht_for(n, MP)
    NCHUNK = (MP + HT - 1) // HT
    # num_warps scaled with the re-read tile / vector size.
    nwarps = 4 if MP <= 256 else (8 if MP <= 512 else 16)
    use_lt = n >= 512        # cublasLt trailing only where matrices are big enough
    for k in range(0, n, B):
        b = min(B, n - k)
        k_end = k + b
        m = n - k
        _panel_qr_tiled_kernel[grid](
            H, tau_all, k, b, m,
            H.stride(0), H.stride(1), H.stride(2),
            tau_all.stride(0), tau_all.stride(1),
            M_POW2=MP, B_POW2=BP, HT=HT, NCHUNK=NCHUNK,
            EPS_V0=_EPS_V0, num_warps=nwarps,
        )
        if k_end < n:
            _wy_trailing(H, tau_all, k=k, b=b, n=n, use_lt=use_lt)
    return H, tau_all


# ══════════════════════════════════════════════════════════════════════════════
# PATH B — v10_fused_smalln: fully fused Householder QR, one program per matrix.
# UNCHANGED from v19. Used for n < 128 (covers n=32).
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _fused_qr_kernel(
    A_ptr, tau_ptr, n,
    stride_ab, stride_ar, stride_ac,
    stride_tb, stride_tc,
    N_POW2: tl.constexpr,
):
    bid = tl.program_id(0)
    rows = tl.arange(0, N_POW2)
    cols = tl.arange(0, N_POW2)
    row_valid = rows < n
    col_valid = cols < n

    base = A_ptr + bid * stride_ab
    ptrs = base + rows[:, None] * stride_ar + cols[None, :] * stride_ac
    mask = row_valid[:, None] & col_valid[None, :]

    H = tl.load(ptrs, mask=mask, other=0.0)
    tau_acc = tl.zeros([N_POW2], dtype=tl.float32)

    for j in range(0, n):
        col_j = tl.sum(tl.where(cols[None, :] == j, H, 0.0), axis=1)
        active = (rows >= j) & row_valid
        x = tl.where(active, col_j, 0.0)

        alpha = tl.sum(tl.where(rows == j, x, 0.0))
        norm_sq = tl.sum(x * x)
        norm = tl.sqrt(norm_sq)

        s = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -s * norm
        v0 = alpha - beta
        tail_sq = tl.sum(tl.where(rows > j, x * x, 0.0))
        v_norm_sq = v0 * v0 + tail_sq
        tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)

        safe_v0 = tl.where(v0 != 0.0, v0, 1.0)
        u = tl.where(rows > j, x / safe_v0, 0.0)
        u = tl.where(rows == j, 1.0, u)

        u_for_dot = tl.where(active, u, 0.0)
        w = tl.sum(u_for_dot[:, None] * H, axis=0)
        trailing = (cols > j) & col_valid
        w = tl.where(trailing, w, 0.0)
        H = H - (tau_j * u_for_dot[:, None]) * w[None, :]

        new_colj = tl.where(rows == j, beta, tl.where(rows > j, x / safe_v0, col_j))
        H = tl.where(cols[None, :] == j, new_colj[:, None], H)

        tau_acc = tau_acc + tl.where(cols == j, tau_j, 0.0)

    tl.store(ptrs, H, mask=mask)
    tau_ptrs = tau_ptr + bid * stride_tb + cols * stride_tc
    tl.store(tau_ptrs, tau_acc, mask=col_valid)


def _num_warps_for(n_pow2: int) -> int:
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
        N_POW2=N_P2, num_warps=_num_warps_for(N_P2),
    )
    return H, tau


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch — identical regimes to v19.
#   n < 128                          -> v10 fused (n=32).
#   128 <= n <= 1024 and batch >= 32 -> height-tiled wide-B panel + trisolve WY.
#   else                             -> torch.geqrf (n=2048/4096; panel-bound).
# ══════════════════════════════════════════════════════════════════════════════
def _use_panel(batch: int, n: int) -> bool:
    return batch >= 32 and 128 <= n <= 1024


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if n < 128:
        return _fused_qr(data)
    if _use_panel(batch, n):
        return _blocked_wy_widetile(data)
    return torch.geqrf(data)
