"""
cute_qr_kernel.py — CuTe-DSL fused Householder QR kernels for the megakernel (v21).

Imported ONLY in the cutlass build image (modal_cute.py). Not imported on the
grader — the grader gets the embedded cubin + a cuda.bindings driver launch.

Numerics MIRROR v19's `_fused_qr_kernel` EXACTLY so the compact (H, tau) output is
bit-compatible with torch.geqrf / LAPACK SGEQRF (validated 19/19 incl band/rowscale).

DSL semantics learned the hard way:
  * The DSL reads function SOURCE via inspect — must live in a real .py file
    (not exec'd from a string).
  * Variables that are read after a dynamic if/while MUST be pre-initialized
    before the control-flow block ("Using variables defined in dynamic control
    flow is not supported"). So: init every accumulator/temp first, then mutate.
  * `for j in range(N)` becomes a dynamic scf loop (fine for correctness).
  * math: cute.math.sqrt(x, fastmath=False). smem: cutlass.utils.SmemAllocator.

Design (v1, "fully resident per-matrix"):
  * ONE CTA per matrix; the whole n×n tile lives in SMEM (n*n*4 <= ~200KB, so this
    variant targets small/medium n; n=512 -> 1MB needs the blocked variant).
  * Threads cooperatively run the n sequential Householder steps on the resident
    tile, then write H and tau back once.
  * All dims COMPILE-TIME STATIC (N constexpr) -> emitted GPU-kernel ABI is just
    raw data pointers (A, tau) — the simple ABI we proved we can driver-launch
    from a clean (no-cutlass) image.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def make_fused_qr(N: int, NTHREADS: int = 256, BATCH: int = 0):
    """Build a CuTe jit fn for unblocked per-matrix QR at size N (compiled later).

    If BATCH>0 the grid is a compile-time constant (so the emitted kernel ABI is
    just the two raw base pointers (A, tau) — the simple ABI we can driver-launch
    from a clean image). If BATCH==0 the grid is read from mA.shape[0] at launch
    (in-process use only)."""

    @cute.kernel
    def _kernel(mA: cute.Tensor, mTau: cute.Tensor):
        # mA: (batch, N, N) row-major f32 ; mTau: (batch, N) f32.
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        # --- smem: resident matrix tile + reduction scratch ---------------------
        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(cute.Float32, cute.make_layout((N, N)),
                                  byte_alignment=16)
        sRed = smem.allocate_tensor(cute.Float32, cute.make_layout((NTHREADS,)),
                                    byte_alignment=16)

        # --- load matrix global -> smem (each thread strided over N*N) ----------
        i = tid
        while i < N * N:
            r = i // N
            c = i % N
            sH[r, c] = mA[bid, r, c]
            i += NTHREADS
        cute.arch.sync_threads()

        # --- N sequential Householder steps ------------------------------------
        for j in range(N):
            # 1) partial sum of squares over rows i>j of column j (tail), plus we
            #    read alpha = H[j,j] directly. norm_sq = alpha^2 + tail_sq.
            part = cutlass.Float32(0.0)
            r = tid + j + 1                 # start at first row strictly below j
            while r < N:
                v = sH[r, j]
                part = part + v * v
                r += NTHREADS
            # warp-reduce the per-thread partials, then combine warps via smem.
            part = cute.arch.warp_reduction_sum(part, threads_in_group=32)
            lane = tid % 32
            warp = tid // 32
            if lane == 0:
                sRed[warp] = part
            cute.arch.sync_threads()
            tail_sq = cutlass.Float32(0.0)
            wcount = (NTHREADS + 31) // 32
            wi = 0
            while wi < wcount:
                tail_sq = tail_sq + sRed[wi]
                wi += 1
            cute.arch.sync_threads()

            alpha = sH[j, j]
            norm_sq = alpha * alpha + tail_sq
            norm = cute.math.sqrt(norm_sq, fastmath=False)

            sign_a = cutlass.Float32(1.0)
            if alpha < 0.0:
                sign_a = cutlass.Float32(-1.0)
            beta = -sign_a * norm
            v0 = alpha - beta
            v_norm_sq = v0 * v0 + tail_sq
            tau_j = cutlass.Float32(0.0)
            if v_norm_sq > 0.0:
                tau_j = 2.0 * v0 * v0 / v_norm_sq
            safe_v0 = cutlass.Float32(1.0)
            if v0 != 0.0:
                safe_v0 = v0

            # 2) write reflector into column j over rows > j: H[i>j,j] = x[i]/v0.
            #    H[j,j] = beta written once by thread 0.
            r = tid + j + 1
            while r < N:
                sH[r, j] = sH[r, j] / safe_v0
                r += NTHREADS
            cute.arch.sync_threads()
            if tid == 0:
                sH[j, j] = beta
                mTau[bid, j] = tau_j
            cute.arch.sync_threads()

            # 3) apply F = I - tau*u*u^T to trailing cols c>j. u[j]=1, u[i>j]=H[i,j]
            #    w = H[j,c] + sum_{i>j} H[i,j]*H[i,c]
            #    H[j,c] -= tau*w ; H[i>j,c] -= tau*w*H[i,j]
            c = j + 1 + tid
            while c < N:
                w = sH[j, c]                         # diagonal term (u[j]=1)
                ii = j + 1
                while ii < N:
                    w = w + sH[ii, j] * sH[ii, c]
                    ii += 1
                tw = tau_j * w
                sH[j, c] = sH[j, c] - tw             # u[j]=1
                ii = j + 1
                while ii < N:
                    sH[ii, c] = sH[ii, c] - tw * sH[ii, j]
                    ii += 1
                c += NTHREADS
            cute.arch.sync_threads()

        # --- write H back to global --------------------------------------------
        i = tid
        while i < N * N:
            r = i // N
            c = i % N
            mA[bid, r, c] = sH[r, c]
            i += NTHREADS
        cute.arch.sync_threads()

    @cute.jit
    def _entry(mA: cute.Tensor, mTau: cute.Tensor):
        if BATCH > 0:
            _kernel(mA, mTau).launch(grid=(BATCH, 1, 1), block=(NTHREADS, 1, 1))
        else:
            batch = mA.shape[0]
            _kernel(mA, mTau).launch(grid=(batch, 1, 1), block=(NTHREADS, 1, 1))

    return _entry
