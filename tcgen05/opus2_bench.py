"""
opus2_bench.py — CORRECTED GATE: v19's REAL trailing update is a BATCHED SKINNY bmm (K=B=32).

The prior B9 gate (opus_bench.py) benched single fat GEMMs M×M, K∈{64,128,256} where
cublasLt-78 is a tuned tcgen05+Ozaki kernel (~9µs, unbeatable). But v19's ACTUAL trailing
update — `torch.baddbmm(A_trail, Y, W)` in _wy_trailing_trisolve — is a BATCHED bmm:
per matrix [M×K]@[K×N] with **K = panel width b = 32**, batch∈{640 (n512), 60 (n1024)},
M up to n, N up to ~M. This is the shape the gate skipped.

Compare THREE methods on the batched-K32 shape:
  (a) torch.bmm  FP32          — what v19 uses (baddbmm = bmm + fused affine add; we time bmm core)
  (b) cublasLt-78 batched      — CUBLAS_COMPUTE_32F_EMULATED_16BFX9 strided-batched, FP32-exact
  (c) tcgen05 BF16x9 batched   — proven opus_bench kernel generalized over batch via grid.z

VERDICT: does (c) beat (a) on batched-K32 → drop-in exact-FP32 trailing replacement Y/N?

tcgen05 kernel: TILE_M=128, TILE_N=256, NPASS=9*K/16. Grid=(M//128, ceil(N/256), batch);
the batch index (bidz) selects the matrix. A/B are pre-split into 9 bf16 limbs packed along K
(width 9*K) per matrix, all 9 products summed in ONE TMEM accumulator per output tile.
M is padded to a multiple of 128 and N to a multiple of 256 for the proven full-tile kernel;
the padded tcgen05 µs is thus an UPPER BOUND on the true-shape time (it does >= the real FLOPs).
Baselines (torch.bmm / cublasLt) run on the TRUE (M,N,K) shapes. FLOP/s uses TRUE M,N,K.

Bit-exactness: x = x0+x1+x2 (3 bf16 limbs) => x*y = Σ_{i,j} x_i*y_j (9 plain-sum products).
opus_stage1 verified rel-vs-FP64 ~4e-7. Per-shape rel-vs-FP64 reported here too.

Anti-hang: server timeout=90; every `modal run` under local `timeout 120`, FOREGROUND.
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus2-bench")


# Batched tcgen05 BF16x9 GEMM. mA=(BATCH, M, 9*K) bf16; mB=(BATCH, 9*K, N) bf16;
# mC=(BATCH, M, N) fp32. NPASS = 9*K/16. Grid=(M//128, N//256, BATCH); bidz = matrix.
# Each block computes one 128x256 output tile of matrix bidz by accumulating all NPASS
# K-blocks (= BF16x9 over full K) in ONE TMEM accumulator. Identical per-tile recipe to
# the PROVEN opus_bench single-batch kernel; only the leading batch index is added.
_KERNEL_SRC = r'''"""tcgen05 BF16x9 BATCHED GEMM — opus2_bench (grid.z over batch)."""
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
def _gemm_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tidx, _, _ = arch.thread_idx()
    bidx, bidy, bidz = arch.block_idx()
    warp_id = arch.warp_idx()
    warp_id = arch.make_warp_uniform(warp_id)
    lane_id = arch.lane_idx()

    KPACK = NPASS * BK          # packed K width = 9*K
    # Select this matrix (bidz) then tile it. mA=(BATCH,M,9K) -> per-matrix (M,9K).
    aM = mA[bidz, None, None]
    bM = mB[bidz, None, None]
    cM = mC[bidz, None, None]
    gA = cute.local_tile(aM, (TILE_M, KPACK), (bidx, 0))      # (128, 9K)
    gB = cute.local_tile(bM, (KPACK, TILE_N), (0, bidy))      # (9K, 256)
    gC = cute.local_tile(cM, (TILE_M, TILE_N), (bidx, bidy))  # (128, 256)
    Ncols = cute.size(cM, mode=[1])

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
    gC_part = cute.make_tensor(
        gC.iterator, cute.make_layout(((TILE_M, TILE_N), 1, 1), stride=((Ncols, 1), 0, 0)))

    if warp_id == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for p in cutlass.range(NPASS, unroll=1):
            k0 = p * BK
            for i in cutlass.range(TILE_M * BK // 32, unroll=1):
                idx = lane_id + i * 32
                m = idx // BK
                k = idx % BK
                sA_c[(m, k), 0, 0] = gA[m, k0 + k]
            for i in cutlass.range(BK * TILE_N // 32, unroll=1):
                idx = lane_id + i * 32
                n = idx // BK
                k = idx % BK
                sB_c[(n, k), 0, 0] = gB[k0 + k, n]
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
def run_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    BATCH = cute.size(mC, mode=[0])
    M = cute.size(mC, mode=[1]); N = cute.size(mC, mode=[2])
    _gemm_kernel(mA, mB, mC).launch(
        grid=(M // TILE_M, N // TILE_N, BATCH), block=(128, 1, 1))
'''


def _build(npass, tag):
    import sys, importlib.util
    src = _KERNEL_SRC.format(NPASS=npass)
    kpath = f"/root/_opus2_bench_{tag}.py"
    with open(kpath, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"_opus2_bench_{tag}", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules[f"_opus2_bench_{tag}"] = kmod
    spec.loader.exec_module(kmod)
    return kmod


def _split3_bf16(x):
    x0 = x.bfloat16(); r1 = x - x0.float()
    x1 = r1.bfloat16(); r2 = r1 - x1.float()
    x2 = r2.bfloat16()
    return x0, x1, x2


def _pack_bf16x9_batched(A_fp32, B_fp32):
    # A_fp32 (BATCH,M,K), B_fp32 (BATCH,K,N) -> packed (BATCH,M,9K),(BATCH,9K,N) bf16.
    import torch
    a0, a1, a2 = _split3_bf16(A_fp32)
    b0, b1, b2 = _split3_bf16(B_fp32)
    pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
    A_packed = torch.cat([ai for (ai, _) in pairs], dim=2).contiguous()  # (B,M,9K)
    B_packed = torch.cat([bi for (_, bi) in pairs], dim=1).contiguous()  # (B,9K,N)
    return A_packed, B_packed


def _ceil_mult(x, m):
    return ((x + m - 1) // m) * m


@app.function(gpu="B200", image=cutlass_image, timeout=90, retries=0)
def perf_gate(use_lt: bool = True):
    import torch, traceback, time
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 92)
    print(f"OPUS2 CORRECTED GATE — BATCHED-skinny K=32 trailing GEMM (v19's REAL shape)  use_lt={use_lt}")
    print("  (a) torch.bmm FP32   (b) cublasLt-78 batched   (c) tcgen05 BF16x9 batched")
    print("=" * 92)

    # ---- cublasLt type-78 STRIDED-BATCHED via ctypes ----
    import ctypes
    lt = None
    if use_lt:
        for name in ("libcublasLt.so.13",
                     "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13",
                     "libcublasLt.so"):
            try:
                lt = ctypes.CDLL(name); break
            except OSError:
                continue
    have_lt = lt is not None
    print(f"  cublasLt loaded: {have_lt}")

    c_void_p, c_int, c_size_t, c_int64, byref = (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_int64, ctypes.byref)
    lt_handle = None
    if have_lt:
        for fn in ["cublasLtCreate","cublasLtMatmul","cublasLtMatmulDescCreate",
                   "cublasLtMatmulDescSetAttribute","cublasLtMatrixLayoutCreate",
                   "cublasLtMatrixLayoutSetAttribute",
                   "cublasLtMatmulPreferenceCreate","cublasLtMatmulPreferenceSetAttribute",
                   "cublasLtMatmulAlgoGetHeuristic"]:
            getattr(lt, fn).restype = c_int
        lt_handle = c_void_p()
        if lt.cublasLtCreate(byref(lt_handle)) != 0:
            print("  cublasLtCreate failed"); have_lt = False

    # cublasLt enums
    CUDA_R_32F = 0; CUDA_R_16BF = 14; COMPUTE_78 = 78
    # MatrixLayout attrs: BATCH_COUNT=4, STRIDED_BATCH_OFFSET=5 (cublasLtMatrixLayoutAttribute_t)
    LAYOUT_BATCH_COUNT = 4; LAYOUT_BATCH_STRIDE = 5

    def lt_batched_gemm(Abf, Bbf):
        """Strided-batched: Abf (BATCH,M,K) bf16, Bbf (BATCH,K,N) bf16 -> C (BATCH,M,N) fp32.
        cublasLt is column-major; compute per-batch C(MxN row)=A(MxK)·B(KxN) via the
        col-major identity Ccol(NxM)=Bcol(NxK)·Acol(KxM) (pass B then A, no transpose),
        with batch strides set on each layout. Returns (run_callable, C)."""
        BATCH, M, K = Abf.shape
        _, _, N = Bbf.shape
        C = torch.empty(BATCH, M, N, device="cuda", dtype=torch.float32)
        desc = c_void_p()
        lt.cublasLtMatmulDescCreate(byref(desc), COMPUTE_78, CUDA_R_32F)
        lB = c_void_p(); lA = c_void_p(); lC = c_void_p()
        lt.cublasLtMatrixLayoutCreate(byref(lB), CUDA_R_16BF, N, K, N)
        lt.cublasLtMatrixLayoutCreate(byref(lA), CUDA_R_16BF, K, M, K)
        lt.cublasLtMatrixLayoutCreate(byref(lC), CUDA_R_32F, N, M, N)
        for lay, stride in ((lB, N * K), (lA, M * K), (lC, M * N)):
            bc = c_int(BATCH)
            lt.cublasLtMatrixLayoutSetAttribute(lay, LAYOUT_BATCH_COUNT, byref(bc), ctypes.sizeof(bc))
            bs = c_int64(stride)
            lt.cublasLtMatrixLayoutSetAttribute(lay, LAYOUT_BATCH_STRIDE, byref(bs), ctypes.sizeof(bs))
        pref = c_void_p(); lt.cublasLtMatmulPreferenceCreate(byref(pref))
        ws = c_size_t(64 * 1024 * 1024)
        lt.cublasLtMatmulPreferenceSetAttribute(pref, 0, byref(ws), ctypes.sizeof(ws))
        class Heur(ctypes.Structure):
            _fields_ = [("algo", ctypes.c_byte*72), ("workspaceSize", c_size_t),
                        ("state", c_int), ("wavesCount", ctypes.c_float), ("reserved", c_int*4)]
        res = (Heur*1)(); cnt = c_int()
        rc = lt.cublasLtMatmulAlgoGetHeuristic(lt_handle, desc, lB, lA, lC, lC, pref, 1,
                                               byref(res), byref(cnt))
        if rc != 0 or cnt.value == 0:
            raise RuntimeError(f"heuristic failed rc={rc} cnt={cnt.value}")
        wsbuf = torch.empty(ws.value, device="cuda", dtype=torch.uint8)
        alpha = ctypes.c_float(1.0); beta = ctypes.c_float(0.0)
        Bp = c_void_p(Bbf.data_ptr()); Ap = c_void_p(Abf.data_ptr()); Cp = c_void_p(C.data_ptr())
        wsp = c_void_p(wsbuf.data_ptr())
        def run():
            rc = lt.cublasLtMatmul(lt_handle, desc, byref(alpha),
                Bp, lB, Ap, lA, byref(beta), Cp, lC, Cp, lC,
                byref(res[0].algo), wsp, ws, c_void_p(0))
            if rc != 0:
                raise RuntimeError(f"cublasLtMatmul rc={rc}")
        return run, C

    def bench(fn, iters=50, warmup=10):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # us

    # ── Representative batched-K32 trailing shapes from the QR sweep ──
    # (label, BATCH, M, N, K=32). first block: N_trail=n-32; mid block: M=n/2, N=n/2-32.
    K = 32
    shapes = [
        ("n512  first ", 640, 512, 480, K),
        ("n512  mid   ", 640, 256, 224, K),
        ("n1024 first ", 60, 1024, 992, K),
        ("n1024 mid   ", 60, 512, 480, K),
    ]

    npass = 9 * K // 16   # = 18
    compiled_cache = {}
    hdr = f"  {'shape':>13} {'B':>4} {'M':>5} {'N':>5} K | {'bmmFP32':>9} | {'cuBLAS78':>9} | {'tcgen05':>9} | {'tc_pad':>8} | rel(tc) | verdict"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    results = []
    for (label, BATCH, M, N, Kk) in shapes:
        torch.manual_seed(7)
        A = torch.randn(BATCH, M, Kk, device="cuda", dtype=torch.float32)
        B = torch.randn(BATCH, Kk, N, device="cuda", dtype=torch.float32)
        ref64 = (A.double() @ B.double())

        # ---- (a) torch.bmm FP32 (the v19 path's GEMM core) ----
        t_bmm = bench(lambda: torch.bmm(A, B))

        # ---- (b) cublasLt-78 batched (FP32-exact via BF16x9) ----
        t_lt = float('nan'); rel_lt = float('nan')
        if have_lt:
            try:
                Abf = A.bfloat16().contiguous(); Bbf = B.bfloat16().contiguous()
                run_lt, Clt = lt_batched_gemm(Abf, Bbf)
                run_lt(); torch.cuda.synchronize()
                rel_lt = ((Clt.double() - ref64).abs().max() / ref64.abs().max()).item()
                t_lt = bench(run_lt)
            except Exception as e:
                print(f"    cuBLAS78 {label} FAILED: {str(e)[:90]}")

        # ---- (c) tcgen05 BF16x9 batched (padded to full 128x256 tiles) ----
        Mp = _ceil_mult(M, 128); Np = _ceil_mult(N, 256)
        t_tc = float('inf'); rel_tc = float('nan'); tc_ok = False
        try:
            Apad = torch.zeros(BATCH, Mp, Kk, device="cuda", dtype=torch.float32)
            Bpad = torch.zeros(BATCH, Kk, Np, device="cuda", dtype=torch.float32)
            Apad[:, :M, :] = A; Bpad[:, :, :N] = B
            A_packed, B_packed = _pack_bf16x9_batched(Apad, Bpad)
            C = torch.zeros(BATCH, Mp, Np, device="cuda", dtype=torch.float32)
            gA = from_dlpack(A_packed); gB = from_dlpack(B_packed); gC = from_dlpack(C)
            grid_m = Mp // 128; grid_n = Np // 256
            ck = (npass, BATCH, Mp, Np)
            print(f"    [{label}] tc grid=({grid_m},{grid_n},{BATCH}) npass={npass} compiling...", flush=True)
            if ck not in compiled_cache:
                km = _build(npass, f"b{BATCH}_{Mp}_{Np}")
                compiled_cache[ck] = (km, cute.compile(km.run_gemm, gA, gB, gC))
            km, comp = compiled_cache[ck]
            comp(gA, gB, gC); torch.cuda.synchronize()
            C_true = C[:, :M, :N]
            rel_tc = ((C_true.double() - ref64).abs().max() / ref64.abs().max()).item()
            tc_ok = rel_tc < 1e-5
            print(f"    [{label}] single-run OK rel={rel_tc:.2e}; benching...", flush=True)
            t_tc = bench(lambda: comp(gA, gB, gC))
        except Exception as e:
            print(f"  {label} tcgen05 FAILED: {type(e).__name__}: {str(e)[:90]}")
            traceback.print_exc()

        # tc_pad note: padded GEMM does Mp*Np work vs true M*N; ratio for honesty.
        pad_ratio = (Mp * Np) / (M * N)
        baselines = [x for x in (t_bmm, t_lt) if x == x]
        best_base = min(baselines) if baselines else float('inf')
        verdict = ("WIN" if (tc_ok and t_tc < best_base)
                   else ("lose" if tc_ok else "ERR"))
        results.append((label, BATCH, M, N, t_bmm, t_lt, t_tc, rel_tc, rel_lt, pad_ratio, verdict))
        print(f"  {label:>13} {BATCH:>4} {M:>5} {N:>5} {Kk:>1} | {t_bmm:>9.1f} | {t_lt:>9.1f} |"
              f" {t_tc:>9.1f} | x{pad_ratio:>5.2f} | {rel_tc:>6.1e} | {verdict}")

    print("\n  GATE SUMMARY (batched-K32 trailing — v19's REAL shape):")
    for r in results:
        label, BATCH, M, N, t_bmm, t_lt, t_tc, rel_tc, rel_lt, pad, v = r
        vs_bmm = (t_tc / t_bmm) if t_bmm > 0 else float('nan')
        vs_lt = (t_tc / t_lt) if (t_lt == t_lt and t_lt > 0) else float('nan')
        print(f"    {label}: tcgen05 = {vs_bmm:6.2f}x torch.bmm,  {vs_lt:6.2f}x cuBLAS78"
              f"  (tc {t_tc:.1f}µs / bmm {t_bmm:.1f}µs / lt {t_lt:.1f}µs; rel_lt={rel_lt:.1e})")
    wins = sum(1 for r in results if r[10] == "WIN")
    print(f"\n    >>> tcgen05 BF16x9 beat the BEST baseline on {wins}/{len(results)} batched-K32 shapes.")
    print(f"    >>> DROP-IN trailing replacement for v19: {'YES' if wins == len(results) else 'NO'}"
          f" (need to beat torch.bmm — the v19 path — on ALL).")
    return results


@app.local_entrypoint()
def main(use_lt: str = "1"):
    perf_gate.remote(use_lt=(use_lt == "1"))
