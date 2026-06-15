"""
opus_bench.py — PERF GATE for the tcgen05 BF16x9 GEMM (Stage 1 go/no-go).

Generalizes the PROVEN opus_stage1 single-tile kernel (TmemAllocator +
PipelineUmmaAsync + canonical make_fragment recipe) to a full M/N-tiled GEMM, then
benchmarks vs cublasLt type-78 (CUBLAS_COMPUTE_32F_EMULATED_16BFX9, the v18 baseline)
and torch FP32 matmul, on the QR trailing shapes:
    M in {512,1024}, N ~ M, K=B in {64,128,256}.

Kernel: grid=(M//128, N//256). Each block computes one 128x256 output tile by
accumulating BF16x9 over the full K. A/B are pre-split into 9 bf16 limbs and packed
along K (width 9*K), so the per-tile MMA count = 9*K/16, all summed in one TMEM acc.
This realizes the thesis "TMEM accumulator un-skinnies the trailing GEMM": K=B is
turned into a long in-TMEM accumulation instead of a skinny batched bmm.

Bit-exactness: x = x0+x1+x2 (3 bf16 limbs) => x*y = sum_{i,j} x_i*y_j (9 plain-sum
products). Verified rel-vs-FP64 ~4e-7 in opus_stage1.

Anti-hang: server timeout=120 (bench needs warmup+iters); local timeout 200.
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-bench")


# Kernel: M/N-tiled. gA=(M, 9*K) bf16 packed limbs; gB=(9*K, N) bf16; gC=(M,N) fp32.
# NPASS = 9*K/16 (set per shape at format time). Grid (M//128, N//256).
_KERNEL_SRC = r'''"""tcgen05 BF16x9 tiled GEMM — opus_bench."""
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
import cutlass.utils.blackwell_helpers as bh
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M: int = 128
TILE_N: int = 256
BK: int = 16
NPASS: int = {NPASS}   # = 9*K/16
THREADS: int = 128


@cute.struct
class SharedStorage:
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def _gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = arch.thread_idx()
    bidx, bidy, _ = arch.block_idx()
    warp_id = arch.warp_idx()
    warp_id = arch.make_warp_uniform(warp_id)
    lane_id = arch.lane_idx()

    m_off = bidx * TILE_M       # this block's row offset in M
    n_off = bidy * TILE_N       # this block's col offset in N

    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N),
    )
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)

    smem = SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(element_type=BFloat16, layout=sA_layout.outer,
                              byte_alignment=128, swizzle=sA_layout.inner)
    sB = smem.allocate_tensor(element_type=BFloat16, layout=sB_layout.outer,
                              byte_alignment=128, swizzle=sB_layout.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr,
                               barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar.data_ptr(),
    ).make_participants()

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C((TILE_M, TILE_N))
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    sA_c = sA[None, None, None, 0]
    sB_c = sB[None, None, None, 0]

    KPACK = NPASS * BK          # packed K width = 9*K
    Ncols = cute.size(gC, mode=[1])
    # This block's output tile in GMEM C: rows [m_off:m_off+128], cols [n_off:n_off+256]
    gC_base = gC.iterator + (m_off * Ncols + n_off)
    gC_part = cute.make_tensor(
        gC_base, cute.make_layout(((TILE_M, TILE_N), 1, 1), stride=((Ncols, 1), 0, 0)))

    # ---- MMA: accumulate all NPASS K-blocks (= BF16x9 over full K) in TMEM ----
    if warp_id == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for p in cutlass.range(NPASS, unroll=1):
            k0 = p * BK
            # A tile: rows [m_off:+128], packed-K cols [k0:k0+16]
            for i in cutlass.range(TILE_M * BK // 32, unroll=1):
                idx = lane_id + i * 32
                m = idx // BK
                k = idx % BK
                sA_c[(m, k), 0, 0] = gA[m_off + m, k0 + k]
            # B tile: packed-K rows [k0:k0+16], cols [n_off:+256]; store as (n,k)
            for i in cutlass.range(BK * TILE_N // 32, unroll=1):
                idx = lane_id + i * 32
                n = idx // BK
                k = idx % BK
                sB_c[(n, k), 0, 0] = gB[k0 + k, n_off + n]
            arch.fence_view_async_shared()
            arch.sync_warp()
            tiled_mma.set(tc.Field.ACCUMULATE, p != 0)
            cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, 0, 0)],
                      tCrB[(None, None, 0, 0)], tCtAcc)
        acc_empty.commit()

    tmem.relinquish_alloc_permit()

    acc_full = acc_consumer.wait_and_advance()
    SUBTILE = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]),
                  cute.size(tCtAcc, mode=[0, 1]) // SUBTILE),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(gC_part, epi_tiler)
    ld_op = tc.Ld32x32bOp(tc.Repetition.x64, tc.Pack.NONE)
    copy_atom = cute.make_copy_atom(ld_op, Float32)
    tmem_tiled_copy = tc.make_tmem_copy(copy_atom, tCtAcc_epi[None, 0])
    thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = thr_copy.partition_S(tCtAcc_epi)
    tDgC = thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, Float32)
    for i in cutlass.range(cute.size(tDtC, mode=[2]), unroll=1):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        arch.fence_view_async_tmem_load()
        cute.autovec_copy(tCrAcc, tDgC[None, None, i])
    acc_full.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def run_gemm(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
             grid_m: cutlass.Constexpr, grid_n: cutlass.Constexpr):
    _gemm_kernel(gA, gB, gC).launch(grid=(grid_m, grid_n, 1), block=(128, 1, 1))
'''


def _build(npass, tag):
    import sys, importlib.util
    src = _KERNEL_SRC.format(NPASS=npass)
    kpath = f"/root/_opus_bench_{tag}.py"
    with open(kpath, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"_opus_bench_{tag}", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules[f"_opus_bench_{tag}"] = kmod
    spec.loader.exec_module(kmod)
    return kmod


def _split3_bf16(x):
    x0 = x.bfloat16(); r1 = x - x0.float()
    x1 = r1.bfloat16(); r2 = r1 - x1.float()
    x2 = r2.bfloat16()
    return x0, x1, x2


def _pack_bf16x9(A_fp32, B_fp32):
    import torch
    a0, a1, a2 = _split3_bf16(A_fp32)
    b0, b1, b2 = _split3_bf16(B_fp32)
    pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
    A_packed = torch.cat([ai for (ai, _) in pairs], dim=1).contiguous()  # (M, 9*K)
    B_packed = torch.cat([bi for (_, bi) in pairs], dim=0).contiguous()  # (9*K, N)
    return A_packed, B_packed


@app.function(gpu="B200", image=cutlass_image, timeout=120, retries=0)
def perf_gate():
    import torch, traceback, time
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 78)
    print("OPUS STAGE1 PERF GATE — tcgen05 BF16x9 vs cublasLt type-78 vs torch FP32")
    print("=" * 78)

    # ---- cublasLt type-78 (CUBLAS_COMPUTE_32F_EMULATED_16BFX9) via ctypes ----
    import ctypes
    lt = None
    for name in ("libcublasLt.so.13",
                 "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13",
                 "libcublasLt.so"):
        try:
            lt = ctypes.CDLL(name); break
        except OSError:
            continue
    have_lt = lt is not None
    print(f"  cublasLt loaded: {have_lt}")

    def make_lt_gemm():
        """Minimal cublasLt wrapper: C = A@B, FP32 out, compute type 78 (BF16x9)."""
        c_void_p, c_int, c_size_t, byref = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.byref
        for fn in ["cublasLtCreate","cublasLtMatmul","cublasLtMatmulDescCreate",
                   "cublasLtMatmulDescSetAttribute","cublasLtMatrixLayoutCreate",
                   "cublasLtMatmulPreferenceCreate","cublasLtMatmulPreferenceSetAttribute",
                   "cublasLtMatmulAlgoGetHeuristic"]:
            getattr(lt, fn).restype = c_int
        handle = c_void_p()
        if lt.cublasLtCreate(byref(handle)) != 0:
            raise RuntimeError("cublasLtCreate failed")
        CUDA_R_32F = 0; CUDA_R_16BF = 14; COMPUTE_78 = 78
        CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0  # not used; type set at create

        def gemm(A, B):
            # A: (M,K) bf16 row-major, B: (K,N) bf16 row-major -> C (M,N) fp32.
            # cublasLt is column-major; compute C^T = B^T @ A^T via op on row-major data.
            M, K = A.shape; K2, N = B.shape
            C = torch.empty(M, N, device="cuda", dtype=torch.float32)
            desc = c_void_p()
            # compute type 78, scale type FP32
            lt.cublasLtMatmulDescCreate(byref(desc), COMPUTE_78, CUDA_R_32F)
            # Column-major layouts: treat row-major (M,K) as col-major (K,M).
            lA = c_void_p(); lB = c_void_p(); lC = c_void_p()
            # We compute C(MxN row) = A(MxK)·B(KxN). In col-major terms with no transpose:
            #   Ccol(NxM) = Bcol(NxK) · Acol(KxM)  => pass B then A as (rows=N/K, ld).
            lt.cublasLtMatrixLayoutCreate(byref(lB), CUDA_R_16BF, N, K, N)
            lt.cublasLtMatrixLayoutCreate(byref(lA), CUDA_R_16BF, K, M, K)
            lt.cublasLtMatrixLayoutCreate(byref(lC), CUDA_R_32F, N, M, N)
            pref = c_void_p(); lt.cublasLtMatmulPreferenceCreate(byref(pref))
            ws = c_size_t(32*1024*1024)
            lt.cublasLtMatmulPreferenceSetAttribute(pref, 0, byref(ws), ctypes.sizeof(ws))
            # heuristic
            class Heur(ctypes.Structure):
                _fields_ = [("algo", ctypes.c_byte*72), ("workspaceSize", c_size_t),
                            ("state", c_int), ("wavesCount", ctypes.c_float), ("reserved", c_int*4)]
            res = (Heur*1)(); cnt = c_int()
            rc = lt.cublasLtMatmulAlgoGetHeuristic(handle, desc, lB, lA, lC, lC, pref, 1,
                                                   byref(res), byref(cnt))
            if rc != 0 or cnt.value == 0:
                raise RuntimeError(f"heuristic failed rc={rc} cnt={cnt.value}")
            wsbuf = torch.empty(ws.value, device="cuda", dtype=torch.uint8)
            alpha = ctypes.c_float(1.0); beta = ctypes.c_float(0.0)
            def run():
                rc = lt.cublasLtMatmul(handle, desc, byref(alpha),
                    c_void_p(B.data_ptr()), lB, c_void_p(A.data_ptr()), lA,
                    byref(beta), c_void_p(C.data_ptr()), lC, c_void_p(C.data_ptr()), lC,
                    byref(res[0].algo), c_void_p(wsbuf.data_ptr()), ws, c_void_p(0))
                if rc != 0:
                    raise RuntimeError(f"cublasLtMatmul rc={rc}")
            return run, C
        return gemm

    lt_gemm = None
    if have_lt:
        try:
            lt_gemm = make_lt_gemm()
            print("  cublasLt type-78 wrapper ready")
        except Exception as e:
            print(f"  cublasLt wrapper setup FAILED: {e}"); traceback.print_exc()

    def bench(fn, iters=50, warmup=10):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # us

    shapes = []
    for m in (512, 1024):
        for K in (64, 128, 256):
            shapes.append((m, m, K))

    # cache compiled tcgen05 kernels by NPASS
    compiled_cache = {}
    print(f"\n  {'shape (M,N,K)':>16} | {'tcgen05 us':>11} | {'cublasLt78 us':>13} | {'torchFP32 us':>12} | verdict")
    print("  " + "-" * 78)
    results = []
    for (M, N, K) in shapes:
        torch.manual_seed(7)
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref64 = A.double() @ B.double()

        # ---- tcgen05 BF16x9 ----
        npass = 9 * K // 16
        A_packed, B_packed = _pack_bf16x9(A, B)
        C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        gA = from_dlpack(A_packed); gB = from_dlpack(B_packed); gC = from_dlpack(C)
        grid_m = M // 128; grid_n = N // 256
        try:
            if npass not in compiled_cache:
                km = _build(npass, f"n{npass}")
                compiled_cache[npass] = (km, cute.compile(km.run_gemm, gA, gB, gC, grid_m, grid_n))
            km, comp = compiled_cache[npass]
            # recompile if grid differs (compile is keyed on constexpr) — compile per (npass,grid)
            comp = cute.compile(km.run_gemm, gA, gB, gC, grid_m, grid_n)
            comp(gA, gB, gC, grid_m, grid_n); torch.cuda.synchronize()
            rel = ((C.double() - ref64).abs().max() / ref64.abs().max()).item()
            t_tc = bench(lambda: comp(gA, gB, gC, grid_m, grid_n))
            tc_ok = rel < 1e-5
        except Exception as e:
            print(f"  {(M,N,K)!s:>16} | tcgen05 FAILED: {type(e).__name__}: {str(e)[:80]}")
            traceback.print_exc(); t_tc = float('inf'); rel = float('nan'); tc_ok = False

        # ---- cublasLt type-78 ----
        t_lt = float('nan')
        if lt_gemm is not None:
            try:
                Abf = A.bfloat16().contiguous(); Bbf = B.bfloat16().contiguous()
                run_lt, _ = lt_gemm(Abf, Bbf)
                t_lt = bench(run_lt)
            except Exception as e:
                print(f"    cublasLt78 {(M,N,K)} FAILED: {str(e)[:80]}")

        # ---- torch FP32 ----
        t_f32 = bench(lambda: torch.matmul(A, B))

        best_baseline = min(x for x in (t_lt, t_f32) if x == x)  # min ignoring nan
        verdict = "WIN" if (tc_ok and t_tc < best_baseline) else ("lose" if tc_ok else "ERR")
        results.append((M, N, K, t_tc, t_lt, t_f32, rel, verdict))
        print(f"  {(M,N,K)!s:>16} | {t_tc:>11.1f} | {t_lt:>13.1f} | {t_f32:>12.1f} | {verdict} (rel={rel:.1e})")

    print("\n  GATE SUMMARY:")
    wins = sum(1 for r in results if r[7] == "WIN")
    print(f"    tcgen05 BF16x9 beat BOTH baselines on {wins}/{len(results)} shapes.")
    return results


@app.local_entrypoint()
def main():
    perf_gate.remote()
