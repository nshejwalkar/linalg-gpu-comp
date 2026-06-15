"""
stage1_v2.py — Corrected CuTe-DSL tcgen05 BF16x9 GEMM using the REAL API.

Key API corrections from probing the installed wheel:
- SmemAllocator() with .allocate_tensor() for SMEM tensors
- arch.alloc_tmem(num_cols, smem_ptr) where smem_ptr is a Pointer
- elect_one() is a CONTEXT MANAGER: `with arch.elect_one():`
- mbarrier_init(mbar_ptr, cnt) where mbar_ptr is a Pointer
- make_umma_smem_desc(src_ptr, layout, major_str, next_src_ptr)
  where major_str is "K" or "MN" (string, not enum)
- tcgen05.commit(mbar_ptr, mask=None, cta_group=CtaGroup.ONE)
- SmemAllocator auto-tracks usage -> kernel launch needs sharedMemBytes

Anti-hang: Modal timeout=90s; local caller uses `timeout 120 modal run ...`

Usage:
  timeout 120 modal run tcgen05/stage1_v2.py --stage smoke 2>&1 | tail -50
  timeout 120 modal run tcgen05/stage1_v2.py --stage bf16x9 2>&1 | tail -60
  timeout 120 modal run tcgen05/stage1_v2.py --stage bench 2>&1 | tail -100
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v2")

# ─────────────────────────────────────────────────────────────────────────────
# Kernel source (written to a real file before compile, since DSL needs it).
# One tile: (TILE_M x TILE_N x TILE_K) BF16->FP32 MMA.
# gC is the accumulator OUTPUT (ADD mode to allow 9-pass BF16x9).
# ─────────────────────────────────────────────────────────────────────────────

_KERNEL_SRC = '''"""
tcgen05 BF16->FP32 single-tile GEMM kernel (CuTe-DSL).
Tile: TILE_M=128, TILE_N=256, TILE_K=64 (4 MMA steps of BK=16).
gC = gC + gA @ gB  (accumulate mode, for BF16x9 9-pass outer loop).
Pass accumulate=True to ADD, False to CLEAR then write.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import BFloat16, Float32, Int32, Boolean

TILE_M: int = 128
TILE_N: int = 256
TILE_K: int = 64   # total K dimension
BK: int = 16       # K per MMA instruction (BF16)
NWARPS: int = 8    # 256 threads total
MMA_WARP: int = 2  # warp that issues tcgen05.mma (single thread via elect_one)
EPI_WARP_START: int = 4  # warps 4..7 = epilogue warpgroup


@cute.kernel
def _mma_kernel(
    gA: cute.Tensor,         # (TILE_M, TILE_K) BF16 gmem
    gB: cute.Tensor,         # (TILE_K, TILE_N) BF16 gmem
    gC: cute.Tensor,         # (TILE_M, TILE_N) FP32 gmem (input+output, ADD)
    accumulate: Boolean,     # True = add to gC; False = overwrite gC
):
    """One tcgen05 BF16->FP32 MMA tile."""
    tidx, _, _ = arch.thread_idx()
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    # ── Shared memory via SmemAllocator ────────────────────────────────────
    smem = SmemAllocator()

    # A tile: (128, 64) BF16, K-major layout (row-major)
    sA = smem.allocate_tensor(
        BFloat16,
        cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1))
    )
    # B tile: (64, 256) BF16, K-major layout
    sB = smem.allocate_tensor(
        BFloat16,
        cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1))
    )
    # TMEM address slot: 1 x Uint32
    tmem_addr_slot = smem.allocate_tensor(
        cutlass.Uint32,
        cute.make_layout((1,))
    )
    # MMA->epilogue mbarrier: 1 x 64-bit
    mbar_mma = smem.allocate_tensor(
        cutlass.Uint64,
        cute.make_layout((1,))
    )

    # ── Phase 1: Warp 0 loads A and B tiles from gmem ─────────────────────
    # Simple cooperative copy: 32 threads, iterate over elements.
    # A: 128*64=8192 BF16 elements, 32 threads -> 256 per thread
    if warp_id == 0:
        n_a = TILE_M * TILE_K  # 8192
        for i in range(n_a // 32):
            idx = lane_id + i * 32
            r = idx // TILE_K
            c = idx % TILE_K
            sA[r, c] = gA[r, c]
        # B: 64*256=16384 BF16 elements, 32 threads -> 512 per thread
        n_b = TILE_K * TILE_N
        for i in range(n_b // 32):
            idx = lane_id + i * 32
            r = idx // TILE_N
            c = idx % TILE_N
            sB[r, c] = gB[r, c]

    # ── Phase 2: Warp 1 allocs TMEM; warp 0 inits mbarrier ────────────────
    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(mbar_mma.iterator, 1)

    if warp_id == 1:
        with arch.elect_one():
            arch.alloc_tmem(TILE_N, tmem_addr_slot.iterator)
            arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()

    # ── Phase 3: MMA warp (warp 2, single thread) ─────────────────────────
    if warp_id == MMA_WARP:
        with arch.elect_one():
            # Build MMA op: BF16 inputs, FP32 accumulator, 1-SM
            mma_op = tc.MmaF16BF16Op(
                BFloat16,          # ab_dtype
                Float32,           # acc_dtype
                (TILE_M, TILE_N, BK),  # instruction shape
                tc.CtaGroup.ONE,
                tc.OperandSource.SMEM,
                tc.OperandMajorMode.K,
                tc.OperandMajorMode.K,
            )

            # Issue K-loop: TILE_K // BK iterations
            for k_it in range(TILE_K // BK):
                k_off_a = k_it * BK * TILE_M  # byte offset into sA for this k-block
                k_off_b = k_it * BK * TILE_N  # byte offset into sB for this k-block

                # Build SMEM descriptors for this k-block.
                # make_umma_smem_desc(src_ptr, layout, major_str, next_src_ptr)
                # major_str is "K" for K-major (row-major A, col-major B from MMA view).
                sA_k = cute.make_tensor(
                    cute.recast_ptr(sA.iterator + k_it * BK, BFloat16),
                    cute.make_layout((TILE_M, BK), stride=(TILE_K, 1))
                )
                sB_k = cute.make_tensor(
                    cute.recast_ptr(sB.iterator + k_it * BK * TILE_N, BFloat16),
                    cute.make_layout((BK, TILE_N), stride=(TILE_N, 1))
                )

                desc_a = tc.make_umma_smem_desc(sA_k.iterator, sA_k.layout, "K", None)
                desc_b = tc.make_umma_smem_desc(sB_k.iterator, sB_k.layout, "K", None)

                # Fields: ACCUMULATE for k_it>0 or when accumulate=True (9-pass BF16x9)
                acc_field = [tc.Field.ACCUMULATE] if (k_it > 0 or accumulate) else []
                mma_op(tmem_addr_slot[0], desc_a, desc_b, acc_field)

            # Commit: signal epilogue that accumulator is ready.
            tc.commit(mbar_mma.iterator)

    arch.sync_threads()

    # ── Phase 4: Epilogue warpgroup (warps 4-7) ────────────────────────────
    # Each warp covers 32 of the 128 TMEM lanes.
    if warp_id >= EPI_WARP_START:
        warp_epi = warp_id - EPI_WARP_START  # 0..3
        lane_base = warp_epi * 32

        # Wait for MMA to finish.
        if lane_id == 0:
            arch.mbarrier_wait(mbar_mma.iterator, 0)
        arch.sync_warp()

        # Load from TMEM using Ld32x32bOp.
        # Each call returns one FP32 for one thread (one lane, one column group).
        # We iterate over TILE_N columns in groups of 32.
        ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
        my_lane = lane_base + lane_id

        for n_blk in range(TILE_N // 32):
            my_col = n_blk * 32 + lane_id
            # TMEM addr = (lane << 16) | col
            tmem_a = tmem_addr_slot[0] | (my_lane << 16) | my_col
            val = ld_op(tmem_a)

            # fence after all loads in this warp
            if n_blk == TILE_N // 32 - 1:
                arch.fence_view_async_tmem_load()

            # Write to output (ADD mode vs OVERWRITE mode doesn\'t matter here
            # because TMEM already has the full accumulated result; we always
            # ADD to gC from Python side by initializing gC=0 before first pass).
            # For simplicity: gC stores the TMEM result additively.
            gC[my_lane, my_col] = gC[my_lane, my_col] + val

    arch.sync_threads()

    # ── Phase 5: Warp 1 deallocates TMEM ──────────────────────────────────
    if warp_id == 1:
        with arch.elect_one():
            arch.dealloc_tmem(tmem_addr_slot[0], TILE_N)


@cute.jit
def run_mma(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, accumulate: Boolean):
    """Launch one 128x256x64 BF16->FP32 MMA tile."""
    _mma_kernel(gA, gB, gC, accumulate).launch(
        grid=(1, 1, 1),
        block=(NWARPS * 32, 1, 1),
    )
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    """Smoke: one 128x256x64 BF16->FP32 MMA, verify correctness."""
    import os
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import BFloat16, Float32, Boolean

    print("=" * 72)
    print("STAGE 1 V2 SMOKE — CuTe-DSL tcgen05 128x256x64 BF16->FP32")
    print("=" * 72)

    kpath = "/root/_stage1_v2_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v2_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v2_kernel"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  KERNEL LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    M, N, K = 128, 256, 64
    torch.manual_seed(42)
    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_bf16 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C_out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    gA = from_dlpack(A_bf16.contiguous())
    gB = from_dlpack(B_bf16.contiguous())
    gC = from_dlpack(C_out)

    print("  Compiling (first call, may take ~30s)...")
    try:
        compiled = cute.compile(kmod.run_mma, gA, gB, gC, Boolean(False))
        print("  cute.compile OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running kernel (accumulate=False = overwrite mode)...")
    try:
        compiled(gA, gB, gC, Boolean(False))
        torch.cuda.synchronize()
        print("  Kernel returned (no hang).")
    except Exception as e:
        print(f"  KERNEL EXEC FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    ref = A_bf16.float() @ B_bf16.float()
    err = (C_out - ref).abs().max().item()
    denom = ref.abs().max().item()
    rel_err = err / denom if denom > 0 else err
    print(f"  max abs err={err:.5f}  rel={rel_err:.2e}  (expect ~1e-2 for BF16 MMA)")
    print(f"  C_out[0,:4] = {C_out[0,:4].tolist()}")
    print(f"  ref  [0,:4] = {ref[0,:4].tolist()}")
    ok = rel_err < 5e-2
    print(f"  >>> SMOKE {'PASS' if ok else 'FAIL (check values above)'}")
    return ok


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def bf16x9_test():
    """BF16x9: 9 MMA passes -> FP32-exact result. Verify rel_err ~3e-7 vs FP64."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import BFloat16, Float32, Boolean

    print("=" * 72)
    print("STAGE 1 V2 BF16x9 — 9-pass Ozaki FP32-exact verification")
    print("=" * 72)

    kpath = "/root/_stage1_v2_kernel_bf16x9.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v2_kernel_bf16x9", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v2_kernel_bf16x9"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  KERNEL LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    M, N, K = 128, 256, 64
    torch.manual_seed(123)
    A_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_fp32 = torch.randn(K, N, device="cuda", dtype=torch.float32)

    def split_bf16(x):
        """Split FP32 -> 3 BF16 limbs (Ozaki 3-split)."""
        x0 = x.bfloat16()
        r1 = x - x0.float()
        x1 = r1.bfloat16()
        r2 = r1 - x1.float()
        x2 = r2.bfloat16()
        return x0, x1, x2

    a0, a1, a2 = split_bf16(A_fp32)
    b0, b1, b2 = split_bf16(B_fp32)
    # 9 pairs descending magnitude: (0,0), (0,1), (1,0), (0,2), (2,0), (1,1), (1,2), (2,1), (2,2)
    pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]

    C_out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    print("  Compiling...")
    gA0 = from_dlpack(pairs[0][0].contiguous())
    gB0 = from_dlpack(pairs[0][1].contiguous())
    gC = from_dlpack(C_out)
    try:
        compiled = cute.compile(kmod.run_mma, gA0, gB0, gC, Boolean(False))
        print("  cute.compile OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running 9 MMA passes (BF16x9)...")
    for i, (ai, bi) in enumerate(pairs):
        ai_c = ai.contiguous()
        bi_c = bi.contiguous()
        gAi = from_dlpack(ai_c)
        gBi = from_dlpack(bi_c)
        # First pass: accumulate=False (C starts at 0, kernel adds to it)
        # Subsequent passes: accumulate=True (add to existing C_out)
        # Actually we accumulate ON THE CPU SIDE via gC being updated each call.
        # The kernel always ADDS val to gC (gC[r,c] = gC[r,c] + val).
        # So we pass Boolean(False) which only affects the TMEM internal behavior.
        acc = Boolean(i > 0)
        try:
            compiled(gAi, gBi, gC, acc)
        except Exception as e:
            print(f"  PASS {i} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    torch.cuda.synchronize()
    print("  All 9 passes done.")

    ref_f32 = A_fp32 @ B_fp32
    ref_f64 = A_fp32.double() @ B_fp32.double()
    err_vs_f32 = (C_out - ref_f32).abs().max().item()
    err_vs_f64 = (C_out.double() - ref_f64).abs().max().item()
    denom_f64 = ref_f64.abs().max().item()
    rel_f64 = err_vs_f64 / denom_f64 if denom_f64 > 0 else 0
    print(f"  err vs torch FP32 = {err_vs_f32:.3e}")
    print(f"  err vs FP64       = {err_vs_f64:.3e}  rel = {rel_f64:.3e}  (target ~3e-7)")
    ok = rel_f64 < 1e-5
    print(f"  >>> BF16x9 accuracy {'PASS' if ok else 'FAIL'}")
    return ok


@app.local_entrypoint()
def main(stage: str = "smoke"):
    print(f">>> Running stage1_v2 stage: {stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown stage: {stage}")
