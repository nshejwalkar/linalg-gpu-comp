"""
stage1_v4.py — tcgen05 BF16x9 GEMM via CuTe-DSL, v4.

Key fixes from v3 failures:
  1. make_umma_smem_desc major arg: lowercase "k" NOT "K" (MLIR enum is lowercase)
  2. TMEM epilogue: retrieve_tmem_ptr + make_tensor + make_tmem_copy + cute.copy
     (NOT raw bitwise-OR integer address; Ld32x32bOp takes a TMEM Pointer, not int)
  3. No Boolean kernel arg (causes 'BFloat16 has no attribute type' error) — split
     into two kernels: _mma_kernel_clear (first pass) and _mma_kernel_accum (subsequent)
  4. alloc_tmem writes to smem_tmem_ptr (SmemAllocator.allocate(Uint32) → Pointer)
  5. dealloc_tmem takes the smem Pointer (same one passed to alloc_tmem), NOT tmem addr
  6. retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr) gives a typed TMEM Pointer
  7. elect_one() is a context manager: `with arch.elect_one():`

Anti-hang: Modal timeout=90s; local `timeout 120 modal run ...`

Usage:
  conda activate modal && set PYTHONUTF8=1
  timeout 120 modal run tcgen05/stage1_v4.py --stage smoke 2>&1 | tail -60
  timeout 120 modal run tcgen05/stage1_v4.py --stage bf16x9 2>&1 | tail -60
  timeout 120 modal run tcgen05/stage1_v4.py --stage bench 2>&1 | tail -120
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v4")


# ─────────────────────────────────────────────────────────────────────────────
# Kernel source — written to a real .py file on the container (DSL needs source).
# One tile: (128 x 256 x 64) BF16->FP32 MMA.
# Two entrypoints:
#   run_mma_clear(gA, gB, gC)  — clears TMEM acc, does MMA, writes to gC (overwrite)
#   run_mma_accum(gA, gB, gC)  — accumulates into TMEM, then ADDS result to gC
# For BF16x9: call run_mma_clear for pass 0, run_mma_accum for passes 1..8.
# ─────────────────────────────────────────────────────────────────────────────

_KERNEL_SRC = r'''"""
tcgen05 BF16->FP32 one-tile MMA kernel (CuTe-DSL, v4).

Two entrypoints:
  run_mma_clear(gA, gB, gC) : CLEAR TMEM then MMA, OVERWRITE gC
  run_mma_accum(gA, gB, gC) : ACCUMULATE into TMEM, ADD to gC

BF16x9 outer loop calls clear on pass 0, accum on passes 1-8.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32, Int32

TILE_M: int = 128
TILE_N: int = 256
TILE_K: int = 64
BK: int = 16        # K per MMA instruction for BF16
N_K_ITERS: int = TILE_K // BK  # = 4


def _build_kernel(first_pass: bool):
    """
    Returns a @cute.kernel function. first_pass=True clears TMEM then does MMA
    (output overwrites gC). first_pass=False accumulates into existing TMEM,
    then ADDS result to gC.
    We build two closures because the CuTe DSL can't take Python bool args.
    """
    ACCUMULATE_FIRST = not first_pass  # whether to use ACCUMULATE on k_iter==0

    @cute.kernel
    def _mma_kernel(
        gA: cute.Tensor,   # (TILE_M, TILE_K) BF16
        gB: cute.Tensor,   # (TILE_K, TILE_N) BF16
        gC: cute.Tensor,   # (TILE_M, TILE_N) FP32
    ):
        warp_id = arch.warp_idx()
        lane_id = arch.lane_idx()

        # ── Shared memory ────────────────────────────────────────────────────
        smem = SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)))
        sB = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1)))
        smem_tmem_ptr = smem.allocate(Uint32)   # uint32 slot; alloc_tmem writes TMEM addr here
        smem_mbar = smem.allocate(cutlass.Uint64)  # mbarrier: MMA->epilogue handoff

        # ── Phase 0: init mbarrier (warp 0 lane 0) ──────────────────────────
        if warp_id == 0 and lane_id == 0:
            arch.mbarrier_init(smem_mbar, 1)

        # ── Phase 1: load A + B from gmem -> smem (warp 0) ──────────────────
        if warp_id == 0:
            for i in range(TILE_M * TILE_K // 32):
                idx = lane_id + i * 32
                sA[idx // TILE_K, idx % TILE_K] = gA[idx // TILE_K, idx % TILE_K]
            for i in range(TILE_K * TILE_N // 32):
                idx = lane_id + i * 32
                sB[idx // TILE_N, idx % TILE_N] = gB[idx // TILE_N, idx % TILE_N]

        arch.fence_view_async_shared()

        # ── Phase 2: alloc TMEM (warp 1 lane 0) ─────────────────────────────
        if warp_id == 1 and lane_id == 0:
            arch.alloc_tmem(TILE_N, smem_tmem_ptr)
            arch.relinquish_tmem_alloc_permit()

        arch.sync_threads()

        # ── Phase 3: MMA (warp 2, single thread via elect_one) ──────────────
        if warp_id == 2:
            with arch.elect_one():
                mma_op = tc.MmaF16BF16Op(
                    BFloat16, Float32,
                    (TILE_M, TILE_N, BK),
                    tc.CtaGroup.ONE,
                    tc.OperandSource.SMEM,
                    tc.OperandMajorMode.K,
                    tc.OperandMajorMode.K,
                )

                # Build descriptors for A and B tiles.
                # KEY FIX: major must be lowercase "k" (MLIR enum; "K" fails with parse error).
                desc_a = tc.make_umma_smem_desc(
                    sA.iterator,
                    cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)),
                    "k",
                    None
                )
                desc_b = tc.make_umma_smem_desc(
                    sB.iterator,
                    cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1)),
                    "k",
                    None
                )

                # K-loop: 4 steps of BK=16
                for k_it in range(N_K_ITERS):
                    if k_it == 0 and not ACCUMULATE_FIRST:
                        mma_op(smem_tmem_ptr, desc_a, desc_b, [])  # clear TMEM
                    else:
                        mma_op(smem_tmem_ptr, desc_a, desc_b, [tc.Field.ACCUMULATE])

                tc.commit(smem_mbar)

        arch.sync_threads()

        # ── Phase 4: epilogue (warps 4-7) ────────────────────────────────────
        # Each epilogue warp handles 32 of the 128 TMEM lanes (= M rows).
        if warp_id >= 4:
            warp_epi = warp_id - 4   # 0..3
            lane_base = warp_epi * 32

            if lane_id == 0:
                arch.mbarrier_wait(smem_mbar, 0)
            arch.sync_warp()

            # Get a typed TMEM Pointer via retrieve_tmem_ptr.
            # signature: retrieve_tmem_ptr(element_type, alignment, smem_ptr)
            tmem_fp32_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)

            # TMEM layout: (TILE_M lanes, TILE_N cols), stride=(lane_stride, 1).
            # TMEM addressing: addr = (lane << 16) | col, so lane_stride = 65536.
            tAcc = cute.make_tensor(
                tmem_fp32_ptr,
                cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
            )

            # Partition tAcc for the current warp using make_tmem_copy.
            ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
            tiled_copy = tc.make_tmem_copy(ld_op, tAcc)
            tidx, _, _ = arch.thread_idx()
            thr_copy = tiled_copy.get_slice(tidx)
            tSrc = thr_copy.partition_S(tAcc)
            tDst = cute.make_fragment_like(tSrc)

            cute.copy(tiled_copy, tSrc, tDst)
            arch.fence_view_async_tmem_load()

            # Write accumulated result to gC (overwrite on first pass, add on accum).
            # Build a gmem tensor with same shape for the output partition.
            gC_tile = cute.make_tensor(
                gC.iterator,
                cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
            )
            tGOut = thr_copy.partition_D(gC_tile)
            if first_pass:
                # Overwrite mode: gC = TMEM result
                for i in range(cute.size(tDst)):
                    tGOut[i] = tDst[i]
            else:
                # Accumulate mode: gC += TMEM result (ADD for BF16x9 outer loop)
                for i in range(cute.size(tDst)):
                    tGOut[i] = tGOut[i] + tDst[i]

        arch.sync_threads()

        # ── Phase 5: dealloc TMEM (warp 1 lane 0) ───────────────────────────
        if warp_id == 1 and lane_id == 0:
            arch.dealloc_tmem(smem_tmem_ptr, TILE_N)

    return _mma_kernel


# Build the two kernel variants at module import time.
_mma_kernel_clear = _build_kernel(first_pass=True)
_mma_kernel_accum = _build_kernel(first_pass=False)


@cute.jit
def run_mma_clear(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    """First MMA pass: clear TMEM accumulator, compute, overwrite gC."""
    _mma_kernel_clear(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))


@cute.jit
def run_mma_accum(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    """Subsequent MMA passes: accumulate into TMEM, add result to gC."""
    _mma_kernel_accum(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    """Smoke: one BF16->FP32 MMA tile (clear mode), verify against torch FP32."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V4 SMOKE — tcgen05 128x256x64 BF16->FP32 (clear mode)")
    print("=" * 72)

    kpath = "/root/_stage1_v4_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v4_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v4_kernel"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    M, N, K = 128, 256, 64
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    gA = from_dlpack(A.contiguous())
    gB = from_dlpack(B.contiguous())
    gC = from_dlpack(C)

    print("  Compiling run_mma_clear...")
    try:
        compiled_clear = cute.compile(kmod.run_mma_clear, gA, gB, gC)
        print("  cute.compile(run_mma_clear) OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running run_mma_clear...")
    try:
        compiled_clear(gA, gB, gC)
        torch.cuda.synchronize()
        print("  Kernel returned OK.")
    except Exception as e:
        print(f"  EXEC FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    ref = A.float() @ B.float()
    err = (C - ref).abs().max().item()
    denom = ref.abs().max().item()
    rel = err / denom if denom > 0 else err
    print(f"  max abs err={err:.5f}  rel={rel:.2e}  (BF16 single MMA: expect ~1e-2)")
    print(f"  C[0,:4]   = {C[0,:4].tolist()}")
    print(f"  ref[0,:4] = {ref[0,:4].tolist()}")
    ok = rel < 5e-2
    print(f"  >>> SMOKE {'PASS' if ok else 'FAIL (rel err too high — check layout)'}")
    return ok


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def bf16x9_test():
    """BF16x9: 9 passes (clear+8 accum), verify FP32-exact vs FP64."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V4 BF16x9 — 9-pass Ozaki split verification")
    print("=" * 72)

    kpath = "/root/_stage1_v4_bf16x9.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v4_bf16x9", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v4_bf16x9"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    M, N, K = 128, 256, 64
    torch.manual_seed(123)
    A_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_fp32 = torch.randn(K, N, device="cuda", dtype=torch.float32)

    def split_bf16(x):
        x0 = x.bfloat16()
        r1 = x - x0.float()
        x1 = r1.bfloat16()
        r2 = r1 - x1.float()
        x2 = r2.bfloat16()
        return x0, x1, x2

    a0, a1, a2 = split_bf16(A_fp32)
    b0, b1, b2 = split_bf16(B_fp32)
    pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]

    C_out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    print("  Compiling run_mma_clear + run_mma_accum...")
    gA0 = from_dlpack(pairs[0][0].contiguous())
    gB0 = from_dlpack(pairs[0][1].contiguous())
    gC = from_dlpack(C_out)
    try:
        compiled_clear = cute.compile(kmod.run_mma_clear, gA0, gB0, gC)
        compiled_accum = cute.compile(kmod.run_mma_accum, gA0, gB0, gC)
        print("  cute.compile OK (both variants)")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running 9 passes (1 clear + 8 accum)...")
    for i, (ai, bi) in enumerate(pairs):
        try:
            gAi = from_dlpack(ai.contiguous())
            gBi = from_dlpack(bi.contiguous())
            if i == 0:
                compiled_clear(gAi, gBi, gC)
            else:
                compiled_accum(gAi, gBi, gC)
        except Exception as e:
            print(f"  PASS {i} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    torch.cuda.synchronize()

    ref_f64 = A_fp32.double() @ B_fp32.double()
    ref_f32 = A_fp32 @ B_fp32
    err_f64 = (C_out.double() - ref_f64).abs().max().item()
    denom = ref_f64.abs().max().item()
    rel_f64 = err_f64 / denom if denom > 0 else 0
    err_f32 = (C_out - ref_f32).abs().max().item()
    print(f"  err vs torch FP32 = {err_f32:.3e}")
    print(f"  err vs FP64       = {err_f64:.3e}  rel = {rel_f64:.3e}  (target ~3e-7)")
    ok = rel_f64 < 1e-5
    print(f"  >>> BF16x9 {'PASS' if ok else 'FAIL'}")
    return ok


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def bench_test():
    """
    Perf gate: tcgen05 BF16x9 vs cublasLt type-78 vs torch FP32 bmm.
    Trailing shapes: m in {512, 1024}, N=m, K=B in {64, 128, 256}.
    Uses batched versions (batch=4 for the perf comparison since
    our kernels are single-tile; scale b to match cublasLt).
    """
    import sys
    import time
    import torch
    import traceback
    import importlib.util
    import ctypes
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V4 BENCH — tcgen05 BF16x9 vs cublasLt type-78 vs torch FP32")
    print("=" * 72)

    kpath = "/root/_stage1_v4_bench.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v4_bench", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v4_bench"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # ── cublasLt setup (same approach as v18) ─────────────────────────────
    _COMPUTE_32F = 68
    _COMPUTE_32F_EMULATED_16BFX9 = 78
    _CUDA_R_32F = 0
    _OP_N, _OP_T = 0, 1
    _ORDER_ROW = 1

    _lt_lib = None
    for name in ["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"]:
        try:
            _lt_lib = ctypes.CDLL(name)
            break
        except OSError:
            pass
    if _lt_lib is None:
        print("  WARNING: cublasLt not found, skipping cublasLt benchmark")

    def _do_bench(fn, warmup=5, rep=20):
        """Simple GPU timer: warmup runs, then median of rep runs."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        return times[len(times)//2]  # median ms

    # Compile once with representative tensor.
    M0, N0, K0 = 128, 256, 64
    torch.manual_seed(0)
    A0 = torch.randn(M0, K0, device="cuda", dtype=torch.bfloat16)
    B0 = torch.randn(K0, N0, device="cuda", dtype=torch.bfloat16)
    C0 = torch.zeros(M0, N0, device="cuda", dtype=torch.float32)
    gA0 = from_dlpack(A0); gB0 = from_dlpack(B0); gC0 = from_dlpack(C0)

    try:
        compiled_clear = cute.compile(kmod.run_mma_clear, gA0, gB0, gC0)
        compiled_accum = cute.compile(kmod.run_mma_accum, gA0, gB0, gC0)
        print("  Kernels compiled OK.")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    def split_bf16(x):
        x0 = x.bfloat16()
        r1 = x - x0.float()
        x1 = r1.bfloat16()
        r2 = r1 - x1.float()
        x2 = r2.bfloat16()
        return x0, x1, x2

    results = []

    # Benchmark shapes (single tile 128x256x64 is our primitive).
    # For fair multi-tile comparison: run NTILES tiles sequentially.
    # trailing shapes: m=n, K=B. We map: TILE calls = ceil(m/128) * ceil(n/256) * 9 passes.
    trailing_shapes = [
        # (m, n, k=B)  - these are the actual trailing GEMM shapes from our QR
        (512,  512,  64),
        (512,  512,  128),
        (512,  512,  256),
        (1024, 1024, 64),
        (1024, 1024, 128),
        (1024, 1024, 256),
    ]

    print(f"\n{'Shape':>25} | {'tcgen05 BF16x9':>16} | {'torch FP32 bmm':>16} | {'Ratio':>8}")
    print("-" * 75)

    for (m, n, k) in trailing_shapes:
        # Use 1 matrix at a time for baseline comparison (like v18).
        # tcgen05 tiles: m_tiles * n_tiles tiles total; k_tiles handled inside kernel.
        m_tiles = m // 128
        n_tiles = n // 256
        k_iter_per_tile = k // 64  # how many full TILE_K=64 chunks

        # Prepare random BF16 inputs.
        torch.manual_seed(1)
        A_fp32 = torch.randn(m, k, device="cuda", dtype=torch.float32)
        B_fp32 = torch.randn(k, n, device="cuda", dtype=torch.float32)
        C_fp32 = torch.zeros(m, n, device="cuda", dtype=torch.float32)

        # BF16x9 splits.
        a0, a1, a2 = split_bf16(A_fp32)
        b0, b1, b2 = split_bf16(B_fp32)
        pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]

        def run_tcgen05():
            """Run tcgen05 BF16x9 for one (m, n, k) GEMM."""
            C_out = torch.zeros(m, n, device="cuda", dtype=torch.float32)
            gC_run = from_dlpack(C_out)
            first = True
            for ai, bi in pairs:
                # Tile over m and n.
                for mt in range(m_tiles):
                    for nt in range(n_tiles):
                        for kt in range(k_iter_per_tile):
                            # Slice 128x64 from A, 64x256 from B.
                            A_tile = ai[mt*128:(mt+1)*128, kt*64:(kt+1)*64].contiguous()
                            B_tile = bi[kt*64:(kt+1)*64, nt*256:(nt+1)*256].contiguous()
                            gA_t = from_dlpack(A_tile)
                            gB_t = from_dlpack(B_tile)
                            # Use C slice.
                            C_slice = C_out[mt*128:(mt+1)*128, nt*256:(nt+1)*256]
                            gC_t = from_dlpack(C_slice)
                            if first and kt == 0:
                                compiled_clear(gA_t, gB_t, gC_t)
                            else:
                                compiled_accum(gA_t, gB_t, gC_t)
                first = False

        def run_torch_fp32():
            """torch FP32 matmul for comparison."""
            _ = torch.mm(A_fp32, B_fp32)

        # Time tcgen05.
        try:
            t_tc = _do_bench(run_tcgen05)
        except Exception as e:
            print(f"  tcgen05 FAILED for {m}x{n}x{k}: {e}")
            t_tc = float("inf")

        # Time torch FP32.
        t_torch = _do_bench(run_torch_fp32)

        ratio = t_tc / t_torch if t_torch > 0 else float("inf")
        shape_str = f"m={m} n={n} k={k}"
        print(f"  {shape_str:>23} | {t_tc:>14.3f}ms | {t_torch:>14.3f}ms | {ratio:>7.2f}x")
        results.append((shape_str, t_tc, t_torch, ratio))

    print()
    print("NOTE: tcgen05 timing above includes Python-level tiling overhead.")
    print("For a FAIR comparison, the real Stage-2 megakernel would tile inside the GPU.")
    print("These numbers show the UPPER BOUND overhead of the current single-tile primitive.")

    wins = sum(1 for r in results if r[3] < 1.0)
    print(f"\nGATE: tcgen05 beats torch FP32 in {wins}/{len(results)} shapes")
    return results


@app.local_entrypoint()
def main(stage: str = "smoke"):
    print(f">>> stage1_v4 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    elif stage == "bench":
        results = bench_test.remote()
        print(">>> BENCH DONE")
    else:
        print(f"Unknown stage: {stage}")
