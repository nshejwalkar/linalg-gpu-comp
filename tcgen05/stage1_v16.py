"""
stage1_v16.py — tcgen05 BF16x9 GEMM, v16.

FIX v15+ errors:

1. cute.gemm path now works! Use:
   tiled_mma = cute.make_tiled_mma(mma_op)
   thr_mma = tiled_mma.get_slice(0)
   tSA = thr_mma.partition_A(sA)   # ((128,16),1,1)
   tSB = thr_mma.partition_B(sB)   # ((256,16),1,1)
   tCC = thr_mma.partition_C(tCtAcc)
   frag_a = thr_mma.make_fragment_A(tSA)
   frag_b = thr_mma.make_fragment_B(tSB)
   cute.gemm(tiled_mma, tCC, frag_a, frag_b, tCC)

2. dealloc_tmem error: smem_tmem_ptr is SMEM ptr, but dealloc needs TMEM ptr.
   arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr) returns the TMEM ptr.
   Must save it and pass to dealloc.
   BUT: dealloc_tmem needs Uint32 TMEM ptr, not Float32 TMEM ptr.
   Wait -- from earlier probes: arch.alloc_tmem(TILE_N, smem_tmem_ptr) stores
   the TMEM address in smem_tmem_ptr (a SMEM Uint32).
   arch.dealloc_tmem(smem_tmem_ptr, TILE_N) should use the smem ptr as address storage.
   But the error says it gets a smem ptr, expects tmem ptr.

   ALTERNATIVE: Use arch.dealloc_tmem on the retrieved TMEM ptr.
   OR: Read arch.dealloc_tmem signature -- maybe it needs the tmem ptr directly.

   From _mlir_helpers/arch.py or similar: dealloc_tmem(tmem_ptr, num_cols)
   where tmem_ptr is the actual TMEM pointer (retrieved via retrieve_tmem_ptr).

Anti-hang: Modal timeout=90s
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v16")


_KERNEL_SRC = r'''"""
tcgen05 BF16->FP32 v16.
Fix: use tiled_mma + partition_A/B + make_fragment_A/B + cute.gemm.
Fix: dealloc_tmem needs TMEM ptr (from retrieve_tmem_ptr), not SMEM ptr.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M: int = 128
TILE_N: int = 256
BK: int = 16


@cute.kernel
def _mma_kernel_clear(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    smem_atom_a = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    sw_a = smem_atom_a.inner
    smem_atom_b = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    sw_b = smem_atom_b.inner

    affine_a = cute.make_layout((TILE_M, BK), stride=(BK, 1))
    affine_b = cute.make_layout((BK, TILE_N), stride=(TILE_N, 1))

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, affine_a, swizzle=sw_a)
    sB = smem.allocate_tensor(BFloat16, affine_b, swizzle=sw_b)
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
        mma_op = tc.MmaF16BF16Op(
            BFloat16, Float32, (TILE_M, TILE_N, BK),
            tc.CtaGroup.ONE, tc.OperandSource.SMEM,
            OperandMajorMode.K, OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)
        thr_mma = tiled_mma.get_slice(0)

        tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
        tCtAcc = cute.make_tensor(
            tmem_ptr,
            cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
        )

        tSA = thr_mma.partition_A(sA)
        tSB = thr_mma.partition_B(sB)
        tCC = thr_mma.partition_C(tCtAcc)

        frag_a = thr_mma.make_fragment_A(tSA)
        frag_b = thr_mma.make_fragment_B(tSB)

        cute.gemm(tiled_mma, tCC, frag_a, frag_b, tCC)
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
            tGOut[i] = tDst[i]

    arch.sync_threads()

    # dealloc_tmem needs the TMEM ptr (retrieved from smem), not smem ptr
    if warp_id == 1 and lane_id == 0:
        tmem_ptr_dealloc = arch.retrieve_tmem_ptr(Uint32, 128, smem_tmem_ptr)
        arch.dealloc_tmem(tmem_ptr_dealloc, TILE_N)


@cute.kernel
def _mma_kernel_accum(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    smem_atom_a = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    sw_a = smem_atom_a.inner
    smem_atom_b = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, BFloat16)
    sw_b = smem_atom_b.inner

    affine_a = cute.make_layout((TILE_M, BK), stride=(BK, 1))
    affine_b = cute.make_layout((BK, TILE_N), stride=(TILE_N, 1))

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, affine_a, swizzle=sw_a)
    sB = smem.allocate_tensor(BFloat16, affine_b, swizzle=sw_b)
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
        mma_op = tc.MmaF16BF16Op(
            BFloat16, Float32, (TILE_M, TILE_N, BK),
            tc.CtaGroup.ONE, tc.OperandSource.SMEM,
            OperandMajorMode.K, OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)
        thr_mma = tiled_mma.get_slice(0)

        tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
        tCtAcc = cute.make_tensor(
            tmem_ptr,
            cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
        )

        tSA = thr_mma.partition_A(sA)
        tSB = thr_mma.partition_B(sB)
        tCC = thr_mma.partition_C(tCtAcc)

        frag_a = thr_mma.make_fragment_A(tSA)
        frag_b = thr_mma.make_fragment_B(tSB)

        cute.gemm(tiled_mma, tCC, frag_a, frag_b, tCC)
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
            tGOut[i] = tGOut[i] + tDst[i]

    arch.sync_threads()

    if warp_id == 1 and lane_id == 0:
        tmem_ptr_dealloc = arch.retrieve_tmem_ptr(Uint32, 128, smem_tmem_ptr)
        arch.dealloc_tmem(tmem_ptr_dealloc, TILE_N)


@cute.jit
def run_mma_clear(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _mma_kernel_clear(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))


@cute.jit
def run_mma_accum(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _mma_kernel_accum(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V16 SMOKE — tiled_mma + partition + make_fragment + cute.gemm")
    print("=" * 72)

    kpath = "/root/_stage1_v16_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v16_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v16_kernel"] = kmod
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
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V16 BF16x9 — Ozaki split, 9 passes, K=16")
    print("=" * 72)

    kpath = "/root/_stage1_v16_bf16x9.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v16_bf16x9", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v16_bf16x9"] = kmod
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
    print(f">>> stage1_v16 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown stage: {stage}")
