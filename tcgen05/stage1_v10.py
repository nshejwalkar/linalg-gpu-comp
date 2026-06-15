"""
stage1_v10.py — tcgen05 BF16x9 GEMM, v10.

TWO KEY FIXES:
1. MMA: call _cute_ir.gemm directly with raw ir.Values, bypassing Python isinstance checks
   atom._unpack() -> atom_val (ir.Value)
   _cute_ir.gemm(atom_val, d.value, [desc_a_val], [desc_b_val], c.value)

2. EPILOGUE: use cute.make_copy_atom(ld_op, Float32) to create CopyAtom from Ld32x32bOp,
   THEN pass CopyAtom to tc.make_tmem_copy(copy_atom, tAcc)

Also: for BF16x9 with K=64 (4 instructions), use kernel with 4 MMA iterations.
But for SMOKE: K=16 (single instruction), simplest case.

ACCUMULATE field: for 9-pass BF16x9, each kernel computes fresh TMEM (no K-loop needed
since K=BK=16 per pass). The ADD to gC is done in epilogue (not in TMEM).
So no ACCUMULATE flag needed within a single kernel call.

Anti-hang: Modal timeout=90s
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v10")


_KERNEL_SRC = r'''"""
tcgen05 BF16->FP32 v10.
Fix 1: _cute_ir.gemm direct call (bypass Python isinstance check).
Fix 2: cute.make_copy_atom(ld_op, Float32) to create CopyAtom.
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

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_M, BK), stride=(BK, 1)))
    sB = smem.allocate_tensor(BFloat16, cute.make_layout((BK, TILE_N), stride=(TILE_N, 1)))
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

            # Build SMEM descriptors (return SmemDescViewType ir.Value)
            desc_a_val = tc.make_umma_smem_desc(
                sA.iterator,
                cute.make_layout((TILE_M, BK), stride=(BK, 1)),
                "k", None
            )
            desc_b_val = tc.make_umma_smem_desc(
                sB.iterator,
                cute.make_layout((BK, TILE_N), stride=(TILE_N, 1)),
                "k", None
            )

            # Bypass cute.gemm Python wrapper (isinstance check fails for SmemDescViewType).
            # Call _cute_ir.gemm directly with raw ir.Values.
            atom_val = mma_atom._unpack()
            _cute_ir.gemm(
                atom_val,
                tCtAcc.value,    # d: TMEM tensor ir.Value
                [desc_a_val],    # a: list of ir.Values (SmemDescViewType)
                [desc_b_val],    # b: list of ir.Values
                tCtAcc.value,    # c: same as d (accumulate into TMEM; clear on first)
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

        # FIX: use make_copy_atom to create a CopyAtom from Ld32x32bOp
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

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_M, BK), stride=(BK, 1)))
    sB = smem.allocate_tensor(BFloat16, cute.make_layout((BK, TILE_N), stride=(TILE_N, 1)))
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
                cute.make_layout((TILE_M, BK), stride=(BK, 1)),
                "k", None
            )
            desc_b_val = tc.make_umma_smem_desc(
                sB.iterator,
                cute.make_layout((BK, TILE_N), stride=(TILE_N, 1)),
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
    print("STAGE 1 V10 SMOKE — _cute_ir.gemm + make_copy_atom fix")
    print("=" * 72)

    kpath = "/root/_stage1_v10_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v10_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v10_kernel"] = kmod
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
    print("STAGE 1 V10 BF16x9 — Ozaki split, 9 passes, K=16")
    print("=" * 72)

    kpath = "/root/_stage1_v10_bf16x9.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v10_bf16x9", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v10_bf16x9"] = kmod
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
    print(f">>> stage1_v10 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown stage: {stage}")
