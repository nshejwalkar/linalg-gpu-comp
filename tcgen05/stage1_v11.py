"""
stage1_v11.py — tcgen05 BF16x9 GEMM, v11.

FIX v10 ERROR:
  'cute_nvgpu.make_umma_smem_desc' op Expect rank-2 vec_mode layout but got 128:16

  Root cause: flat stride layout (128,16):(16,1) is not a "vec_mode" layout.
  Need: ComposedLayout with swizzle, created via:
    1. tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
       -> ComposedLayout: sw=S<1,4,3>, outer=(8,16):(16,1)  [for BF16, 256-bit contiguous]
    2. tc.tile_to_mma_shape(layout_atom, (TILE_M, BK)) -> tiled ComposedLayout
    3. smem.allocate_tensor(BFloat16, tiled_layout) -> SMEM pointer with swizzle
    4. tc.make_umma_smem_desc(sA.iterator, tiled_layout, "k") -> works

For TILE_M=128, BK=16, BF16:
  K_SW32: num_contiguous=256b/16b=16 BF16, atom=(8,16)
  Tile to (128,16): M-repeats=128/8=16, K-repeats=16/16=1

Anti-hang: Modal timeout=90s
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v11")


_KERNEL_SRC = r'''"""
tcgen05 BF16->FP32 v11.
Fix: use make_smem_layout_atom + tile_to_mma_shape for swizzled SMEM layout.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32
import cutlass._mlir.dialects.cute as _cute_ir

TILE_M: int = 128
TILE_N: int = 256
BK: int = 16   # one BF16 MMA instruction = K=16


@cute.kernel
def _mma_kernel_clear(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    """Single tcgen05 MMA K=16, TMEM cleared, overwrite gC."""
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    # Swizzled SMEM layout for A: K-major, K_SW32 (256b/16b=16 BF16 contiguous)
    # Atom: (8,16):(16,1) with S<1,4,3> swizzle -> ComposedLayout
    # Tiled to (128,16): 16 repeats in M, 1 in K
    smem_layout_a_atom = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    smem_layout_a = tc.tile_to_mma_shape(smem_layout_a_atom, (TILE_M, BK))

    # Swizzled SMEM layout for B: K-major, K_SW32
    # For B (K,N)=(16,256): atom=(8,16), tile to (16,256): K-repeats=16/8=2, N-repeats=256/16=16
    # Wait — B is (K=16, N=256). K_SW32 K-major atom = (8, 16):(16,1).
    # tile_to_mma_shape((8,16), (16, 256)): shape = (16/8=2, 256/16=16) * atom = correct
    # But major_str for B in MMA: B is N-major (K dimension is the "K" mode).
    # Use "k" for K-major (row-major for B since B is KxN).
    smem_layout_b_atom = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    smem_layout_b = tc.tile_to_mma_shape(smem_layout_b_atom, (BK, TILE_N))

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, smem_layout_a)
    sB = smem.allocate_tensor(BFloat16, smem_layout_b)
    smem_tmem_ptr = smem.allocate(Uint32)
    smem_mbar = smem.allocate(cutlass.Uint64)

    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(smem_mbar, 1)

    # Load A: TILE_M * BK = 128*16 = 2048 BF16 elements, 32 threads x 64 iters
    if warp_id == 0:
        for i in range(TILE_M * BK // 32):
            idx = lane_id + i * 32
            sA[idx // BK, idx % BK] = gA[idx // BK, idx % BK]
        for i in range(BK * TILE_N // 32):
            idx = lane_id + i * 32
            sB[idx // TILE_N, idx % TILE_N] = gB[idx // TILE_N, idx % TILE_N]

    arch.fence_view_async_shared()

    if warp_id == 1 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smem_tmem_ptr)
        arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()

    if warp_id == 2:
        with arch.elect_one():
            mma_op = tc.MmaF16BF16Op(
                BFloat16, Float32, (TILE_M, TILE_N, BK),
                tc.CtaGroup.ONE, tc.OperandSource.SMEM,
                OperandMajorMode.K, OperandMajorMode.K,
            )
            mma_atom = cute.make_mma_atom(mma_op)

            tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
            tCtAcc = cute.make_tensor(
                tmem_ptr,
                cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
            )

            # Build SMEM descriptors using swizzled layout
            desc_a_val = tc.make_umma_smem_desc(
                sA.iterator,
                smem_layout_a,
                "k", None
            )
            desc_b_val = tc.make_umma_smem_desc(
                sB.iterator,
                smem_layout_b,
                "k", None
            )

            # Bypass cute.gemm Python wrapper — call _cute_ir.gemm directly
            atom_val = mma_atom._unpack()
            _cute_ir.gemm(
                atom_val,
                tCtAcc.value,    # d: TMEM tensor ir.Value
                [desc_a_val],    # a: SmemDescViewType ir.Value
                [desc_b_val],    # b
                tCtAcc.value,    # c: same as d (fresh TMEM each kernel call)
            )

            tc.commit(smem_mbar)

    arch.sync_threads()

    if warp_id >= 4:
        if lane_id == 0:
            arch.mbarrier_wait(smem_mbar, 0)
        arch.sync_warp()

        tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
        tAcc = cute.make_tensor(
            tmem_ptr,
            cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
        )

        ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
        copy_atom = cute.make_copy_atom(ld_op, Float32)
        tiled_copy = tc.make_tmem_copy(copy_atom, tAcc)

        tidx, _, _ = arch.thread_idx()
        thr_copy = tiled_copy.get_slice(tidx)
        tSrc = thr_copy.partition_S(tAcc)
        tDst = cute.make_fragment_like(tSrc)

        cute.copy(tiled_copy, tSrc, tDst)
        arch.fence_view_async_tmem_load()

        gC_tile = cute.make_tensor(
            gC.iterator,
            cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
        )
        tGOut = thr_copy.partition_D(gC_tile)
        for i in range(cute.size(tDst)):
            tGOut[i] = tDst[i]   # overwrite gC

    arch.sync_threads()

    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(smem_tmem_ptr, TILE_N)


@cute.kernel
def _mma_kernel_accum(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    """Compute fresh TMEM MMA, ADD result to gC (for BF16x9 passes 1-8)."""
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    smem_layout_a_atom = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    smem_layout_a = tc.tile_to_mma_shape(smem_layout_a_atom, (TILE_M, BK))
    smem_layout_b_atom = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    smem_layout_b = tc.tile_to_mma_shape(smem_layout_b_atom, (BK, TILE_N))

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, smem_layout_a)
    sB = smem.allocate_tensor(BFloat16, smem_layout_b)
    smem_tmem_ptr = smem.allocate(Uint32)
    smem_mbar = smem.allocate(cutlass.Uint64)

    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(smem_mbar, 1)

    if warp_id == 0:
        for i in range(TILE_M * BK // 32):
            idx = lane_id + i * 32
            sA[idx // BK, idx % BK] = gA[idx // BK, idx % BK]
        for i in range(BK * TILE_N // 32):
            idx = lane_id + i * 32
            sB[idx // TILE_N, idx % TILE_N] = gB[idx // TILE_N, idx % TILE_N]

    arch.fence_view_async_shared()

    if warp_id == 1 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smem_tmem_ptr)
        arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()

    if warp_id == 2:
        with arch.elect_one():
            mma_op = tc.MmaF16BF16Op(
                BFloat16, Float32, (TILE_M, TILE_N, BK),
                tc.CtaGroup.ONE, tc.OperandSource.SMEM,
                OperandMajorMode.K, OperandMajorMode.K,
            )
            mma_atom = cute.make_mma_atom(mma_op)

            tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
            tCtAcc = cute.make_tensor(
                tmem_ptr,
                cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
            )

            desc_a_val = tc.make_umma_smem_desc(
                sA.iterator,
                smem_layout_a,
                "k", None
            )
            desc_b_val = tc.make_umma_smem_desc(
                sB.iterator,
                smem_layout_b,
                "k", None
            )
            atom_val = mma_atom._unpack()
            _cute_ir.gemm(
                atom_val,
                tCtAcc.value,
                [desc_a_val],
                [desc_b_val],
                tCtAcc.value,
            )
            tc.commit(smem_mbar)

    arch.sync_threads()

    if warp_id >= 4:
        if lane_id == 0:
            arch.mbarrier_wait(smem_mbar, 0)
        arch.sync_warp()

        tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
        tAcc = cute.make_tensor(
            tmem_ptr,
            cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
        )

        ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
        copy_atom = cute.make_copy_atom(ld_op, Float32)
        tiled_copy = tc.make_tmem_copy(copy_atom, tAcc)

        tidx, _, _ = arch.thread_idx()
        thr_copy = tiled_copy.get_slice(tidx)
        tSrc = thr_copy.partition_S(tAcc)
        tDst = cute.make_fragment_like(tSrc)

        cute.copy(tiled_copy, tSrc, tDst)
        arch.fence_view_async_tmem_load()

        gC_tile = cute.make_tensor(
            gC.iterator,
            cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
        )
        tGOut = thr_copy.partition_D(gC_tile)
        for i in range(cute.size(tDst)):
            tGOut[i] = tGOut[i] + tDst[i]   # accumulate

    arch.sync_threads()

    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(smem_tmem_ptr, TILE_N)


@cute.jit
def run_mma_clear(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _mma_kernel_clear(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))


@cute.jit
def run_mma_accum(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _mma_kernel_accum(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    """Smoke: 128x256x16 BF16->FP32 single MMA."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V11 SMOKE — swizzled SMEM + make_copy_atom")
    print("=" * 72)

    kpath = "/root/_stage1_v11_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v11_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v11_kernel"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    M, N, K = 128, 256, 16
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
        print("  cute.compile OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running smoke...")
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
    print(f"  max abs err={err:.5f}  rel={rel:.2e}")
    print(f"  C[0,:4]   = {C[0,:4].tolist()}")
    print(f"  ref[0,:4] = {ref[0,:4].tolist()}")
    ok = rel < 5e-2
    print(f"  >>> SMOKE {'PASS' if ok else 'FAIL'}")
    return ok


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def bf16x9_test():
    """BF16x9 Ozaki split: 9 passes, K=16 per pass."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V11 BF16x9 — Ozaki split, 9 passes, K=16")
    print("=" * 72)

    kpath = "/root/_stage1_v11_bf16x9.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v11_bf16x9", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v11_bf16x9"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    M, N, K = 128, 256, 16
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
    gA0 = from_dlpack(pairs[0][0].contiguous())
    gB0 = from_dlpack(pairs[0][1].contiguous())
    gC = from_dlpack(C_out)

    print("  Compiling...")
    try:
        compiled_clear = cute.compile(kmod.run_mma_clear, gA0, gB0, gC)
        compiled_accum = cute.compile(kmod.run_mma_accum, gA0, gB0, gC)
        print("  OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    for i, (ai, bi) in enumerate(pairs):
        gAi = from_dlpack(ai.contiguous())
        gBi = from_dlpack(bi.contiguous())
        try:
            if i == 0:
                compiled_clear(gAi, gBi, gC)
            else:
                compiled_accum(gAi, gBi, gC)
        except Exception as e:
            print(f"  PASS {i} FAILED: {e}")
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
    print(f"  err vs FP64       = {err_f64:.3e}  rel = {rel_f64:.3e}")
    ok = rel_f64 < 1e-5
    print(f"  >>> BF16x9 {'PASS' if ok else 'FAIL'}")
    return ok


@app.local_entrypoint()
def main(stage: str = "smoke"):
    print(f">>> stage1_v11 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown stage: {stage}")
