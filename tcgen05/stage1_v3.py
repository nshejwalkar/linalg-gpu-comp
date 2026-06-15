"""
stage1_v3.py — tcgen05 BF16 GEMM via CuTe-DSL, v3.

Key changes from v2:
- Use static constexpr loop for K tiles (range(N) at Python level = unrolled)
- Use proper SmemAllocator + tensor indexing via sA[r*TILE_K + c] flat access
- Use make_umma_smem_desc correctly with a proper layout+pointer
- Simplify to bare minimum: one MMA tile, overwrite mode only (no accumulate arg)
  to eliminate the Boolean DSL type issue.

Anti-hang: Modal timeout=90s; local caller uses `timeout 120 modal run ...`

Usage:
  timeout 120 modal run tcgen05/stage1_v3.py --stage smoke 2>&1 | tail -50
  timeout 120 modal run tcgen05/stage1_v3.py --stage bf16x9 2>&1 | tail -60
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v3")


# Write kernel source to a file container-side.
# This kernel does ONE 128x256x64 BF16->FP32 MMA and ADDS the result to gC.
# For BF16x9: call 9 times (gC starts at 0, each call ADDS one product).
_KERNEL_SRC = r'''"""
tcgen05 BF16->FP32 single-tile MMA.
Computes: gC += gA @ gB  (ADD mode always; caller zeroes gC before first pass).
gA: (128, 64) BF16 row-major
gB: (64, 256) BF16 row-major
gC: (128, 256) FP32 row-major (input/output, ADDS to existing values)
Block: 8 warps (256 threads)
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32, Uint64

TILE_M: int = 128
TILE_N: int = 256
TILE_K: int = 64
BK: int = 16        # K per MMA instruction
N_K_ITERS: int = TILE_K // BK  # = 4


@cute.kernel
def _mma_kernel(
    gA: cute.Tensor,   # (128, 64) BF16
    gB: cute.Tensor,   # (64, 256) BF16
    gC: cute.Tensor,   # (128, 256) FP32
):
    """BF16->FP32 MMA: gC += gA @ gB."""
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    # ── Shared memory ───────────────────────────────────────────────────────
    smem = SmemAllocator()

    # A: (128, 64) BF16, row-major
    sA = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)))
    # B: (64, 256) BF16, row-major
    sB = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1)))
    # TMEM address slot (Uint32 scalar)
    smem_tmem_ptr = smem.allocate(Uint32)
    # mbarrier for MMA->epilogue handoff (64-bit, aligned)
    smem_mbar = smem.allocate(Uint64)

    # ── Phase 0: init mbarrier (warp 0 thread 0) ───────────────────────────
    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(smem_mbar, 1)

    # ── Phase 1: load A and B into smem (warp 0) ───────────────────────────
    # A: 128*64 = 8192 BF16 elements; 32 threads -> 256 per thread
    if warp_id == 0:
        for i in range(TILE_M * TILE_K // 32):
            idx = lane_id + i * 32
            r = idx // TILE_K
            c = idx % TILE_K
            sA[r, c] = gA[r, c]
        for i in range(TILE_K * TILE_N // 32):
            idx = lane_id + i * 32
            r = idx // TILE_N
            c = idx % TILE_N
            sB[r, c] = gB[r, c]

    arch.fence_view_async_shared()

    # ── Phase 2: alloc TMEM (warp 1) ───────────────────────────────────────
    if warp_id == 1 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smem_tmem_ptr)
        arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()

    # ── Phase 3: MMA warp (warp 2, single thread via elect_one) ───────────
    if warp_id == 2:
        with arch.elect_one():
            mma_op = tc.MmaF16BF16Op(
                BFloat16,
                Float32,
                (TILE_M, TILE_N, BK),
                tc.CtaGroup.ONE,
                tc.OperandSource.SMEM,
                tc.OperandMajorMode.K,
                tc.OperandMajorMode.K,
            )

            # K-loop: 4 iterations of BK=16
            # Use static unrolling by constructing sub-tensors of sA, sB
            # for each k-block using make_tensor with offset layout.
            # IMPORTANT: make_umma_smem_desc takes the SMEM pointer + layout
            # for the full tile and uses the stride to navigate k-blocks.
            # We use the whole sA layout and let the MMA descriptor stride.
            # Actually for BF16x9 we need to handle this carefully:
            # The descriptor describes the full (M, K) tile; the MMA unit
            # handles the K dimension internally with the instruction_shape K.

            # Build descriptors over the full tiles with K-major layout.
            # major="K" = K is the contiguous dimension in the SMEM layout.
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

            # Issue MMA: first pass clears TMEM (no ACCUMULATE field),
            # subsequent passes add (ACCUMULATE field).
            mma_op(smem_tmem_ptr, desc_a, desc_b, [])  # first: clear
            for k_it in range(N_K_ITERS - 1):
                mma_op(smem_tmem_ptr, desc_a, desc_b, [tc.Field.ACCUMULATE])

            tc.commit(smem_mbar)

    arch.sync_threads()

    # ── Phase 4: epilogue warpgroup (warps 4-7) ────────────────────────────
    if warp_id >= 4:
        warp_epi = warp_id - 4
        lane_base = warp_epi * 32

        if lane_id == 0:
            arch.mbarrier_wait(smem_mbar, 0)
        arch.sync_warp()

        # Load from TMEM and ADD to gC.
        ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
        my_lane = lane_base + lane_id

        for n_blk in range(TILE_N // 32):
            my_col = n_blk * 32 + lane_id
            tmem_a = smem_tmem_ptr | (my_lane << 16) | my_col
            val = ld_op(tmem_a)
            if n_blk == TILE_N // 32 - 1:
                arch.fence_view_async_tmem_load()
            # ADD to gC (accumulate multiple BF16x9 passes via multiple kernel calls)
            gC[my_lane, my_col] = gC[my_lane, my_col] + val

    arch.sync_threads()

    # ── Phase 5: dealloc TMEM (warp 1) ─────────────────────────────────────
    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(smem_tmem_ptr, TILE_N)


@cute.jit
def run_mma(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    """Launch one BF16->FP32 tile (128x256x64), adds result to gC."""
    _mma_kernel(gA, gB, gC).launch(
        grid=(1, 1, 1),
        block=(256, 1, 1),
    )
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    """Smoke: one BF16->FP32 MMA tile, verify correctness."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V3 SMOKE — tcgen05 128x256x64 BF16->FP32")
    print("=" * 72)

    kpath = "/root/_stage1_v3_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v3_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v3_kernel"] = kmod
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

    print("  Compiling...")
    try:
        compiled = cute.compile(kmod.run_mma, gA, gB, gC)
        print("  cute.compile OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running...")
    try:
        compiled(gA, gB, gC)
        torch.cuda.synchronize()
        print("  Kernel returned.")
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
    print(f"  >>> SMOKE {'PASS' if ok else 'FAIL'}")
    return ok


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def bf16x9_test():
    """BF16x9: 9 passes, verify FP32-exact output vs FP64."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("STAGE 1 V3 BF16x9 — 9-pass Ozaki verification")
    print("=" * 72)

    kpath = "/root/_stage1_v3_bf16x9.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v3_bf16x9", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v3_bf16x9"] = kmod
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

    print("  Compiling...")
    gA0 = from_dlpack(pairs[0][0].contiguous())
    gB0 = from_dlpack(pairs[0][1].contiguous())
    gC = from_dlpack(C_out)
    try:
        compiled = cute.compile(kmod.run_mma, gA0, gB0, gC)
        print("  cute.compile OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running 9 passes...")
    for i, (ai, bi) in enumerate(pairs):
        try:
            gAi = from_dlpack(ai.contiguous())
            gBi = from_dlpack(bi.contiguous())
            compiled(gAi, gBi, gC)
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


@app.local_entrypoint()
def main(stage: str = "smoke"):
    print(f">>> stage1_v3 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown: {stage}")
