"""
stage1_v6.py — tcgen05 BF16x9 GEMM, v6.

Key fix from v5: cute.gemm(atom, d, a, b, c) needs PARTITIONED tensors.
The correct pattern (from algorithm.py) is:
  tiled_mma = cute.make_tiled_mma(mma_atom)  (or make_tiled_mma(op))
  thr_mma   = tiled_mma.get_slice(tidx)
  tCrA = thr_mma.partition_A(sA)    -- SMEM tensor A (viewed through MMA layout)
  tCrB = thr_mma.partition_B(sB)
  tCtC = thr_mma.partition_C(tCtAcc) -- TMEM tensor
  cute.gemm(mma_atom, tCtC, tCrA, tCrB, tCtC)   -- D=C accumulate

Also: cute.gemm is issued per K-block; after first: atom.accumulate_ needed.
From mma.py Field enum: Field.ACCUMULATE = "accum_c"
The atom.accumulate_ may be set on the atom instance.

BUT: in DSL context, atom fields must be set via cute.arch.mma_atom_call or
equivalent. Let's try setting it via atom itself: unclear. Alternative:
pass extra kwargs to cute.gemm?

Actually from algorithm.py: "some MMA Atoms (e.g. warpgroup-wide or tcgen05 MMAs)
require manually setting an 'accumulate' boolean field." The note says
manually set — i.e., the atom object has an attribute that controls this.
In CuTe C++: `tiled_mma.accumulate_ = UMMA::ScaleOut::One;`
In DSL: likely atom.accumulate_ = tc.Field.ACCUMULATE (or True?)

SIMPLEST APPROACH FOR SMOKE: use only ONE k-iteration (no k-loop), BK=TILE_K=64.
This avoids the accumulate field issue entirely. Then verify smoke is correct.
For the full BF16x9, we'll keep 9 separate kernel calls (one per limb-pair),
each doing the full K in one shot (no accumulate needed within a kernel call).

Anti-hang: Modal timeout=90s; local `timeout 120 modal run ...`
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v6")


_KERNEL_SRC = r'''"""
tcgen05 BF16->FP32 v6: TiledMma partitions + cute.gemm.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

# For BF16: K per MMA instruction is 16. But let's try K=64 (4 instrs).
# Actually check: can MmaF16BF16Op have instruction_shape (128, 256, 64)?
# mma.py line 260: m in [64, 128], n: not shown. K is likely only 16.
# So BK must be 16 for the instruction shape.
# BUT cute.gemm handles K-loop internally via the shape of tCrA/tCrB.
# We partition sA as (TILE_M, TILE_K) and the MMA atom sees the K-mode.
# The tiled_mma with instruction shape (128,256,16) + input A (128,64):
# K mode size = 64/16 = 4 steps -> cute.gemm loops 4 times internally.
# So we just call cute.gemm ONCE and it handles K internally!

TILE_M: int = 128
TILE_N: int = 256
TILE_K: int = 64
BK: int = 16   # instruction K


@cute.kernel
def _mma_kernel_clear(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    """First pass: CLEAR TMEM, compute, OVERWRITE gC."""
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)))
    sB = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1)))
    smem_tmem_ptr = smem.allocate(Uint32)
    smem_mbar = smem.allocate(cutlass.Uint64)

    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(smem_mbar, 1)

    # Load A + B
    if warp_id == 0:
        for i in range(TILE_M * TILE_K // 32):
            idx = lane_id + i * 32
            sA[idx // TILE_K, idx % TILE_K] = gA[idx // TILE_K, idx % TILE_K]
        for i in range(TILE_K * TILE_N // 32):
            idx = lane_id + i * 32
            sB[idx // TILE_N, idx % TILE_N] = gB[idx // TILE_N, idx % TILE_N]

    arch.fence_view_async_shared()

    # Alloc TMEM
    if warp_id == 1 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smem_tmem_ptr)
        arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()

    # MMA warp: build TiledMma, partition, call cute.gemm
    if warp_id == 2:
        with arch.elect_one():
            mma_op = tc.MmaF16BF16Op(
                BFloat16, Float32, (TILE_M, TILE_N, BK),
                tc.CtaGroup.ONE, tc.OperandSource.SMEM,
                OperandMajorMode.K, OperandMajorMode.K,
            )
            mma_atom = cute.make_mma_atom(mma_op)
            tiled_mma = cute.make_tiled_mma(mma_atom)

            # Get the TMEM accumulator tensor
            tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
            tCtAcc = cute.make_tensor(
                tmem_ptr,
                cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
            )

            # Get thread index for partitioning
            tidx, _, _ = arch.thread_idx()

            # Partition the tensors according to the TiledMma
            thr_mma = tiled_mma.get_slice(tidx)

            # Partition A (sA: SMEM) and B (sB: SMEM) for this thread
            tCrA = thr_mma.partition_A(sA)   # (V, M, K) shaped
            tCrB = thr_mma.partition_B(sB)   # (V, N, K) shaped
            tCtC = thr_mma.partition_C(tCtAcc)  # (V, M, N) shaped TMEM

            # Issue GEMM: D = A * B + C (clear on first pass)
            # For first pass (clear TMEM): pass c as tCtC but hardware
            # will clear because ACCUMULATE field is NOT set.
            # Actually: cute.gemm with c=tCtC means "accumulate from c".
            # To CLEAR: we need to set accumulate_=False on the atom.
            # Let's try: just call without accumulate and see what happens.
            cute.gemm(mma_atom, tCtC, tCrA, tCrB, tCtC)

            tc.commit(smem_mbar)

    arch.sync_threads()

    # Epilogue: read TMEM -> gC
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
        tiled_copy = tc.make_tmem_copy(ld_op, tAcc)
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

    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(smem_tmem_ptr, TILE_N)


@cute.kernel
def _mma_kernel_accum(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    """Subsequent passes: ACCUMULATE into TMEM, ADD result to gC."""
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)))
    sB = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1)))
    smem_tmem_ptr = smem.allocate(Uint32)
    smem_mbar = smem.allocate(cutlass.Uint64)

    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(smem_mbar, 1)

    if warp_id == 0:
        for i in range(TILE_M * TILE_K // 32):
            idx = lane_id + i * 32
            sA[idx // TILE_K, idx % TILE_K] = gA[idx // TILE_K, idx % TILE_K]
        for i in range(TILE_K * TILE_N // 32):
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
            tiled_mma = cute.make_tiled_mma(mma_atom)

            tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
            tCtAcc = cute.make_tensor(
                tmem_ptr,
                cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
            )

            tidx, _, _ = arch.thread_idx()
            thr_mma = tiled_mma.get_slice(tidx)
            tCrA = thr_mma.partition_A(sA)
            tCrB = thr_mma.partition_B(sB)
            tCtC = thr_mma.partition_C(tCtAcc)

            # Accumulate mode: same call, but hardware accumulates because d==c
            cute.gemm(mma_atom, tCtC, tCrA, tCrB, tCtC)

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
        tiled_copy = tc.make_tmem_copy(ld_op, tAcc)
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
    """Smoke: one BF16->FP32 MMA tile."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V6 SMOKE — TiledMma partition + cute.gemm")
    print("=" * 72)

    kpath = "/root/_stage1_v6_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v6_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v6_kernel"] = kmod
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
    """BF16x9: 9 passes."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V6 BF16x9")
    print("=" * 72)

    kpath = "/root/_stage1_v6_bf16x9.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v6_bf16x9", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v6_bf16x9"] = kmod
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
    print(f">>> stage1_v6 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown stage: {stage}")
