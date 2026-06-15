"""
stage1_cutedsl.py — Stage-1 tcgen05 BF16x9 GEMM via CuTe-DSL (NOT Gluon).

Route: CuTe-DSL (nvidia-cutlass-dsl 4.5.2, confirmed API from stage0).
The Gluon route hung on mbarrier.wait in stage1_smoke.py; we use the
confirmed CuTe-DSL path whose compile->cubin->driver pipeline is proven (B8).

Plan:
  smoke   : minimal 128x256x64 single-tile BF16->FP32 MMA, verify correctness
  bf16x9  : full BF16x9 (9 MMA passes), verify rel_err vs FP64 ~3e-7
  bench   : perf gate — our trailing shapes (m,n,k) vs cublasLt type-78 vs torch bmm

Anti-hang rules (enforced by caller via `timeout 120 modal run ...`):
  - Modal function timeout=90s (server-side kill)
  - All runs are foreground with local timeout
  - A correct smoke returns in seconds; hang => kernel bug => fix first

Usage:
  conda activate modal
  export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  timeout 120 modal run tcgen05/stage1_cutedsl.py --stage smoke 2>&1 | tail -40
  timeout 120 modal run tcgen05/stage1_cutedsl.py --stage bf16x9 2>&1 | tail -50
  timeout 120 modal run tcgen05/stage1_cutedsl.py --stage bench 2>&1 | tail -80
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-cutedsl")

# ─────────────────────────────────────────────────────────────────────────────
# The kernel source must live in a real .py file on the container filesystem
# (CuTe DSL inspects source via inspect.getsource — REPL strings fail).
# We write it inside the function before importing.
# ─────────────────────────────────────────────────────────────────────────────

# Single-MMA smoke kernel: one 128x256x64 BF16->FP32 tile.
_SMOKE_KERNEL_SRC = '''
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.nvgpu.tcgen05 as tcgen05_ops
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.arch as arch
import cutlass.utils as utils
from cutlass.utils.blackwell_helpers import (
    make_trivial_tiled_mma, make_smem_layout_a, make_smem_layout_b,
    get_num_tmem_alloc_cols, get_tmem_load_op,
)
import cutlass.pipeline as pipeline

import torch

# Tile sizes (constexpr for the DSL).
M: int = 128
N: int = 256
K: int = 64  # total K; BK=16 for BF16 (one MMA step)
BK: int = 16  # K per MMA instruction (BF16 = 16 elements)


@cute.kernel
def _mma_kernel_smoke(
    gA: cute.Tensor,   # (M, K) BF16 gmem row-major
    gB: cute.Tensor,   # (K, N) BF16 gmem row-major
    gC: cute.Tensor,   # (M, N) FP32 gmem row-major, output
):
    """Minimal tcgen05 BF16->FP32: one tile, no pipelining."""
    tidx, _, _ = arch.thread_idx()
    bidx, _, _ = arch.block_idx()
    warp_id = tidx // 32
    lane_id = tidx % 32

    # Shared memory for A and B operand tiles.
    # Layout must match what the MMA descriptor expects: K-major (row-major for A,
    # col-major for B in the SMEM sense), with 128B swizzle.
    # A: (M, K) = (128, 64), BF16 -> 128*64*2 = 16KB
    # B: (K, N) = (64, 256), BF16 -> 64*256*2 = 32KB
    # Total: 48KB — fits in SMEM.
    sA = cute.make_tensor(
        cute.make_smem_ptr(cute.smem_alloc(M * K * 2, cute.bfloat16)),
        cute.make_layout((M, K), stride=(K, 1))
    )
    sB = cute.make_tensor(
        cute.make_smem_ptr(cute.smem_alloc(K * N * 2, cute.bfloat16)),
        cute.make_layout((K, N), stride=(N, 1))
    )

    # Shared memory for the TMEM pointer (alloc writes the address here).
    tmem_ptr_smem = cute.make_tensor(
        cute.make_smem_ptr(cute.smem_alloc(4, cute.uint32)),
        cute.make_layout((1,))
    )

    # ── Step 1: warp 0 loads A tile from gmem -> smem (simple copy) ──────────
    if warp_id == 0:
        # Each warp-0 thread copies a slice of A and B.
        # Simple elementwise copy: thread i handles element i.
        elems_a = M * K  # 8192 elements, 128 threads -> 64 each
        for i in range(elems_a // 128):
            idx = tidx + i * 128
            r = idx // K
            c = idx % K
            sA[r, c] = gA[r, c]
        elems_b = K * N  # 16384 elements, 128 threads -> 128 each
        for i in range(elems_b // 128):
            idx = tidx + i * 128
            r = idx // N
            c = idx % N
            sB[r, c] = gB[r, c]
    arch.sync_threads()

    # ── Step 2: alloc TMEM (one thread in warp 1) ────────────────────────────
    if warp_id == 1 and lane_id == 0:
        arch.alloc_tmem(N, tmem_ptr_smem.data_ptr(), arch="sm_100")
        arch.relinquish_tmem_alloc_permit(arch="sm_100")
    arch.sync_threads()
    tmem_addr = tmem_ptr_smem[0]

    # ── Step 3: MMA (warp 2, single thread via elect_one) ────────────────────
    # mbarrier for MMA -> epilogue sync
    mbar_smem = cute.make_tensor(
        cute.make_smem_ptr(cute.smem_alloc(8, cute.uint64)),
        cute.make_layout((1,))
    )
    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(mbar_smem.data_ptr(), 1)
    arch.sync_threads()

    if warp_id == 2:
        elected = arch.elect_one()
        if elected:
            # Build the MmaF16BF16Op atom for BF16->FP32.
            # instruction_shape = (M, N, BK) for one MMA instruction.
            atom = tcgen05_ops.MmaF16BF16Op(
                cute.bfloat16,   # ab_dtype
                cute.float32,    # acc_dtype
                (M, N, BK),      # instruction_shape (one tile)
                tcgen05_ops.CtaGroup.ONE,
                tcgen05_ops.OperandSource.SMEM,
                tcgen05_ops.OperandMajorMode.K,
                tcgen05_ops.OperandMajorMode.K,
            )
            # Build SMEM descriptors for A and B.
            desc_a = tcgen05_ops.make_umma_smem_desc(sA, cute.make_layout((M, K), stride=(K, 1)), tcgen05_ops.OperandMajorMode.K, None)
            desc_b = tcgen05_ops.make_umma_smem_desc(sB, cute.make_layout((K, N), stride=(N, 1)), tcgen05_ops.OperandMajorMode.K, None)

            # Issue MMA (K//BK iterations, first pass clear acc, rest accumulate).
            n_k_iters = K // BK
            fields = [tcgen05_ops.Field.ACCUMULATE] if n_k_iters > 1 else []
            for k_it in range(n_k_iters):
                acc_fields = fields if k_it > 0 else []
                atom(tmem_addr, desc_a, desc_b, acc_fields)

            # Commit: signal the mbarrier that MMA is done.
            tcgen05_ops.commit(mbar_smem.data_ptr(), 0xFFFFFFFF, tcgen05_ops.CtaGroup.ONE)
    arch.sync_threads()

    # ── Step 4: epilogue warpgroup (warps 4-7 = threads 128-255) ─────────────
    # Wait for MMA completion.
    if warp_id >= 4:
        if lane_id == 0:
            arch.mbarrier_wait(mbar_smem.data_ptr(), 0)

        arch.sync_threads()  # within epilogue group

        # Each warp reads 32 TMEM lanes (rows) x N columns.
        # tcgen05.ld: each warp covers lane_warp_base..+31.
        warp_in_epi = warp_id - 4  # 0..3
        # tmem_addr encodes: addr = (lane << 16) | col_offset
        # Each thread in a 32-thread warp covers one lane.
        lane_base = warp_in_epi * 32

        # Load N FP32 values for my lane (row) from TMEM.
        # Use Ld32x32bOp(Repetition.x1) to load one 32-bit word per call.
        # We do N // 32 calls to cover all N columns in groups of 32.
        ld_atom = tcgen05_ops.Ld32x32bOp(tcgen05_ops.Repetition.x1, tcgen05_ops.Pack.x1)
        for n_blk in range(N // 32):
            col_off = n_blk * 32 + lane_id
            src_addr = tmem_addr | ((lane_base + lane_id) << 16) | col_off
            val = ld_atom(src_addr)
            # fence after last load
            if n_blk == N // 32 - 1:
                arch.fence_view_async_tmem_load()
            row = lane_base + lane_id
            col = col_off
            gC[row, col] = val

    # ── Step 5: dealloc TMEM ─────────────────────────────────────────────────
    arch.sync_threads()
    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(tmem_addr, N, arch="sm_100")


@cute.jit
def mma_smoke(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _mma_kernel_smoke(gA, gB, gC).launch(
        grid=(1, 1, 1),
        block=(256, 1, 1),  # 8 warps
    )
'''


# ─────────────────────────────────────────────────────────────────────────────
# Approach 2: Use the higher-level make_trivial_tiled_mma approach from the
# proven CUTLASS examples. This is cleaner and matches the actual API patterns.
# ─────────────────────────────────────────────────────────────────────────────

_TILED_MMA_KERNEL_SRC = '''
"""
tcgen05 BF16x9 batched GEMM via CuTe DSL.

This kernel implements: C = A @ B  (FP32 output, BF16 inputs via 9-pass Ozaki split)
for a single (M, N, K) tile, using:
  - make_trivial_tiled_mma from cutlass.utils.blackwell_helpers
  - TmemAllocator for TMEM management
  - PipelineUmmaAsync for MMA->epilogue sync
  - Manual BF16 operand split (FP32->BF16 hi/lo/mid per operand)

This is a SIMPLE non-persistent kernel (one CTA per tile) to establish correctness
and baseline perf before adding TMA pipelining.

Entry point: run_gemm(A, B, C) where A:(M,K) BF16, B:(K,N) BF16, C:(M,N) FP32
The 9-pass outer loop lives in Python-side dispatch; each pass does one
tcgen05.mma into the FP32 TMEM accumulator.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.nvgpu.tcgen05 as tc
import cutlass.cute.arch as arch
import cutlass.utils as utils
import torch

# These are constexpr so the ABI is just pointers.
TILE_M: int = 128
TILE_N: int = 256
TILE_K: int = 64   # K per kernel launch
MMA_K: int = 16    # K per MMA instruction (BF16)
NWARPS: int = 8    # 256 threads


@cute.kernel
def _bf16_mma_kernel(
    gA: cute.Tensor,    # (TILE_M, TILE_K) BF16 smem-ready
    gB: cute.Tensor,    # (TILE_K, TILE_N) BF16 smem-ready
    gC: cute.Tensor,    # (TILE_M, TILE_N) FP32 gmem OUT (accumulated ADD)
    clear_acc: cute.Int32,  # 1 = clear accumulator, 0 = accumulate into existing
):
    tidx, _, _ = arch.thread_idx()
    warp_id = tidx // 32
    lane_id = tidx % 32

    # Shared memory layout.
    # A tile: (TILE_M=128, TILE_K=64) BF16 = 16KB  (K-major: stride=(K,1))
    # B tile: (TILE_K=64, TILE_N=256) BF16 = 32KB  (N-major: stride=(N,1))
    # tmem ptr: 4 bytes
    # mbar: 8 bytes
    # Total static smem: ~48KB (fits within 48KB limit without dynamic smem opt-in)
    sA_flat = cute.smem_alloc(TILE_M * TILE_K * 2, cute.bfloat16)
    sB_flat = cute.smem_alloc(TILE_K * TILE_N * 2, cute.bfloat16)
    smem_tmem_ptr = cute.smem_alloc(4, cute.uint32)
    smem_mbar = cute.smem_alloc(8, cute.uint64)

    sA = cute.make_tensor(cute.make_smem_ptr(sA_flat),
                          cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)))
    sB = cute.make_tensor(cute.make_smem_ptr(sB_flat),
                          cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1)))

    # Warp 0: init mbarrier + load A/B tiles from gmem.
    if warp_id == 0:
        if lane_id == 0:
            arch.mbarrier_init(smem_mbar, 1)
        # Load A (128*64=8192 BF16 elements, 32 threads -> 256 each).
        n_a = TILE_M * TILE_K
        for i in range(n_a // 32):
            idx = lane_id + i * 32
            r = idx // TILE_K
            c = idx % TILE_K
            sA[r, c] = gA[r, c]
        n_b = TILE_K * TILE_N
        for i in range(n_b // 32):
            idx = lane_id + i * 32
            r = idx // TILE_N
            c = idx % TILE_N
            sB[r, c] = gB[r, c]

    # Warp 1: alloc TMEM.
    if warp_id == 1 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smem_tmem_ptr, arch="sm_100")
        arch.relinquish_tmem_alloc_permit(arch="sm_100")

    arch.sync_threads()
    tmem_addr = cute.make_tensor(cute.make_smem_ptr(smem_tmem_ptr),
                                 cute.make_layout((1,)))[0]

    # Warp 2: MMA loop.
    if warp_id == 2:
        if arch.elect_one():
            n_k = TILE_K // MMA_K
            for k_it in range(n_k):
                # Offset into sA/sB for this k-block.
                k_off = k_it * MMA_K
                sA_k = cute.make_tensor(
                    cute.make_smem_ptr(sA_flat + TILE_M * k_off * 2),
                    cute.make_layout((TILE_M, MMA_K), stride=(TILE_K, 1))
                )
                sB_k = cute.make_tensor(
                    cute.make_smem_ptr(sB_flat + k_off * TILE_N * 2),
                    cute.make_layout((MMA_K, TILE_N), stride=(TILE_N, 1))
                )
                desc_a = tc.make_umma_smem_desc(sA_k, None, tc.OperandMajorMode.K, None)
                desc_b = tc.make_umma_smem_desc(sB_k, None, tc.OperandMajorMode.K, None)

                # Fields: ACCUMULATE after first k-block; also for subsequent
                # BF16x9 passes (clear_acc controls the very first pass).
                acc_fields = []
                if k_it > 0 or clear_acc == 0:
                    acc_fields = [tc.Field.ACCUMULATE]

                mma_op = tc.MmaF16BF16Op(
                    cute.bfloat16, cute.float32,
                    (TILE_M, TILE_N, MMA_K),
                    tc.CtaGroup.ONE,
                    tc.OperandSource.SMEM,
                    tc.OperandMajorMode.K,
                    tc.OperandMajorMode.K,
                )
                mma_op(tmem_addr, desc_a, desc_b, acc_fields)

            # Signal epilogue.
            tc.commit(smem_mbar, 0xFFFFFFFF, tc.CtaGroup.ONE)

    arch.sync_threads()

    # Warps 4-7: epilogue (each covers 32 of 128 TMEM lanes).
    if warp_id >= 4:
        warp_epi = warp_id - 4  # 0..3
        lane_base = warp_epi * 32

        if lane_id == 0:
            arch.mbarrier_wait(smem_mbar, 0)
        arch.sync_warp()

        # Load TILE_N FP32 values for this warp's 32 lanes from TMEM.
        # Ld32x32bOp loads one 32-bit value from TMEM for one thread.
        # Each thread loads N columns in groups of 32 (N/32 iterations).
        ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.x1)
        for n_blk in range(TILE_N // 32):
            col = n_blk * 32 + lane_id
            lane = lane_base + lane_id
            addr = tmem_addr | (lane << 16) | col
            val = ld_op(addr)
            if n_blk == TILE_N // 32 - 1:
                arch.fence_view_async_tmem_load()
            # Accumulate into output (ADD because we call this 9x for BF16x9).
            gC[lane, col] = gC[lane, col] + val

    arch.sync_threads()
    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(tmem_addr, TILE_N, arch="sm_100")


@cute.jit
def run_mma_tile(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, clear_acc: cute.Int32):
    """Run one 128x256x64 BF16->FP32 MMA tile."""
    _bf16_mma_kernel(gA, gB, gC, clear_acc).launch(
        grid=(1, 1, 1),
        block=(NWARPS * 32, 1, 1),
    )
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    """Smoke: tiny 128x256x64 BF16->FP32 GEMM. Returns in seconds if correct."""
    import os
    import sys
    import torch
    import traceback
    import importlib.util

    print("=" * 72)
    print("STAGE 1 SMOKE — CuTe-DSL tcgen05 128x256x64 BF16->FP32")
    print("=" * 72)

    # Write kernel source to a real file (DSL requires inspect.getsource).
    kpath = "/root/_stage1_smoke_kernel.py"
    with open(kpath, "w") as f:
        f.write(_TILED_MMA_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_smoke_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_smoke_kernel"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  KERNEL LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    M, N, K = 128, 256, 64

    torch.manual_seed(42)
    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_bf16 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C_out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    gA = from_dlpack(A_bf16)
    gB = from_dlpack(B_bf16)
    gC = from_dlpack(C_out)

    try:
        print("  Compiling...")
        compiled = cute.compile(kmod.run_mma_tile, gA, gB, gC, cute.Int32(1))
        print("  cute.compile OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    try:
        print("  Running...")
        compiled(gA, gB, gC, cute.Int32(1))
        torch.cuda.synchronize()
        print("  Kernel returned (no hang).")
    except Exception as e:
        print(f"  KERNEL EXEC FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Check correctness vs BF16 matmul.
    ref = A_bf16.float() @ B_bf16.float()
    err = (C_out - ref).abs().max().item()
    denom = ref.abs().max().item()
    rel_err = err / denom if denom > 0 else err
    print(f"  max abs err={err:.5f}  rel={rel_err:.2e}  (expect ~1e-2 for single BF16 MMA)")
    print(f"  C_out[0,:4] = {C_out[0,:4].tolist()}")
    print(f"  ref  [0,:4] = {ref[0,:4].tolist()}")
    ok = rel_err < 5e-2
    print(f"  >>> SMOKE {'PASS' if ok else 'FAIL'}")
    return ok


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def bf16x9_test():
    """BF16x9: 9 MMA passes for FP32-exact result. Verify rel_err ~3e-7 vs FP64."""
    import os
    import sys
    import torch
    import traceback
    import importlib.util

    print("=" * 72)
    print("STAGE 1 BF16x9 — 9-pass Ozaki FP32-exact verification")
    print("=" * 72)

    kpath = "/root/_stage1_kernel.py"
    with open(kpath, "w") as f:
        f.write(_TILED_MMA_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_kernel"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  KERNEL LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    M, N, K = 128, 256, 64

    torch.manual_seed(123)
    # FP32 operands — we'll split them to BF16 hi/mid/lo.
    A_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_fp32 = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # BF16x9 Ozaki split: A -> (a0, a1, a2), B -> (b0, b1, b2) in BF16.
    def split_bf16(x):
        x0 = x.bfloat16()
        r1 = x - x0.float()
        x1 = r1.bfloat16()
        r2 = r1 - x1.float()
        x2 = r2.bfloat16()
        return x0, x1, x2

    a0, a1, a2 = split_bf16(A_fp32)
    b0, b1, b2 = split_bf16(B_fp32)
    # 9 pairs in descending magnitude order: (0,0), (0,1), (1,0), (0,2), (2,0), (1,1), (1,2), (2,1), (2,2)
    pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]

    C_out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    print("  Compiling BF16x9 kernel (first pair)...")
    gA0 = from_dlpack(pairs[0][0].contiguous())
    gB0 = from_dlpack(pairs[0][1].contiguous())
    gC = from_dlpack(C_out)
    try:
        compiled = cute.compile(kmod.run_mma_tile, gA0, gB0, gC, cute.Int32(1))
        print("  cute.compile OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print("  Running 9 MMA passes...")
    for i, (ai, bi) in enumerate(pairs):
        ai_c = ai.contiguous()
        bi_c = bi.contiguous()
        gAi = from_dlpack(ai_c)
        gBi = from_dlpack(bi_c)
        clear = cute.Int32(1 if i == 0 else 0)
        try:
            compiled(gAi, gBi, gC, clear)
        except Exception as e:
            print(f"  PASS {i} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False
    torch.cuda.synchronize()
    print("  All 9 passes done.")

    # Reference: FP32 matmul (torch) and FP64.
    ref_f32 = A_fp32 @ B_fp32
    ref_f64 = A_fp32.double() @ B_fp32.double()

    err_vs_f32 = (C_out - ref_f32).abs().max().item()
    err_vs_f64 = (C_out.double() - ref_f64).abs().max().item()
    denom_f64 = ref_f64.abs().max().item()
    rel_f64 = err_vs_f64 / denom_f64 if denom_f64 > 0 else err_vs_f64

    print(f"  err vs torch FP32 = {err_vs_f32:.3e}")
    print(f"  err vs FP64       = {err_vs_f64:.3e}  rel = {rel_f64:.3e}  (target ~3e-7)")
    print(f"  C_out[0,:4] = {C_out[0,:4].tolist()}")
    print(f"  ref32[0,:4] = {ref_f32[0,:4].tolist()}")

    ok = rel_f64 < 1e-5  # generous for now; target is ~3e-7
    print(f"  >>> BF16x9 accuracy {'PASS' if ok else 'FAIL (check split order)'}")
    return ok


@app.local_entrypoint()
def main(stage: str = "smoke"):
    print(f">>> Running stage: {stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
        print(f">>> BF16x9: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown stage: {stage}")
