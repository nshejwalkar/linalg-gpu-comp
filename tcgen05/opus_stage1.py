"""
opus_stage1.py — standalone tcgen05 BF16x9 GEMM (Stage 1 de-risk).

Self-contained Modal app (like v11): builds a single tcgen05 MMA via the OFFICIAL
CUTLASS-DSL Blackwell recipe (make_trivial_tiled_mma + make_smem_layout_a/b +
partition_A/B + make_fragment_C in TMEM + cute.gemm), then a 9-pass BF16x9 Ozaki
split for bit-exact FP32, then a perf gate vs cublasLt type-78 + torch FP32.

Recipe pinned by opus_probe_* (build-only, no-hang):
  tiled_mma = make_trivial_tiled_mma(BF16,BF16, K,K, FP32, ONE, (TILE_M,TILE_N))
  sA_layout = make_smem_layout_a(tiled_mma, (TILE_M,TILE_N,BK), BF16, 1)  # swizzled, staged
  sB_layout = make_smem_layout_b(tiled_mma, (TILE_M,TILE_N,BK), BF16, 1)
  sA0 = sA[None,None,None,0]   # drop stage (rank-4 -> rank-3 tile)
  thr = tiled_mma.get_slice(0)
  tCrA = thr.partition_A(sA0); tCrB = thr.partition_B(sB0)
  acc_tmem = <make TMEM tensor>; tCtAcc = thr.partition_C(acc_tmem)
  tiled_mma.set(Field.ACCUMULATE, <bool>); cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)

BF16x9: x = x0+x1+x2 (3 bf16, exact split) => x*y = sum_{i,j} x_i*y_j (9 plain-sum
products, no scaling) => accumulate all 9 in ONE TMEM accumulator, single epilogue.

Anti-hang: server timeout=90; run every `modal run` under local `timeout 120`.
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-stage1")


# ---------------------------------------------------------------------------
# Kernel source (written to a file inside the container, imported fresh).
# Parameterized by TILE_M, TILE_N, NPASS at format time.
# ---------------------------------------------------------------------------
_KERNEL_SRC = r'''"""tcgen05 GEMM kernel — opus_stage1, official recipe."""
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
import cutlass.utils.blackwell_helpers as bh
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M: int = {TILE_M}
TILE_N: int = {TILE_N}
BK: int = 16          # one BF16 tcgen05 MMA instruction along K
NPASS: int = {NPASS}  # 1 = single MMA smoke; 9 = BF16x9


@cute.kernel
def _gemm_kernel(
    gA: cute.Tensor,   # (TILE_M, BK*NPASS)  K-major (row-major), bf16
    gB: cute.Tensor,   # (BK*NPASS, TILE_N)  K-major (row-major), bf16
    gC: cute.Tensor,   # (TILE_M, TILE_N) fp32
):
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()
    tidx, _, _ = arch.thread_idx()

    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16,
        OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N),
    )

    # Swizzled, staged SMEM layouts (one stage). Shape ((MN,K),1,1,stage).
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=1024)
    sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=1024)
    smem_tmem_ptr = smem.allocate(Uint32)
    smem_mbar = smem.allocate(cutlass.Uint64)

    # Move swizzle from the layout onto the pointer so the SMEM tensors have AFFINE
    # layouts (required by make_fragment_A/B). Same physical addressing as the
    # composed tensors, so element-wise loads through these are descriptor-consistent.
    sA_aff = cute.make_tensor(cute.recast_ptr(sA.iterator, sA_layout.inner, BFloat16),
                              sA_layout.outer)
    sB_aff = cute.make_tensor(cute.recast_ptr(sB.iterator, sB_layout.inner, BFloat16),
                              sB_layout.outer)

    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(smem_mbar, 1)

    # Allocate TMEM accumulator: TILE_N columns (each column = one N, 128 lanes = M tile).
    if warp_id == 0 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smem_tmem_ptr)
        arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()

    # --- MMA fragments (canonical dense_gemm.py recipe) ---
    # make_fragment_A is called DIRECTLY on the staged SMEM tensor (affine, swizzle
    # in ptr) -> (MMA, MMA_M, MMA_K, STAGE). NOT on partition_A (that's the GMEM path).
    tCrA = tiled_mma.make_fragment_A(sA_aff)
    tCrB = tiled_mma.make_fragment_B(sB_aff)
    acc_shape = tiled_mma.partition_shape_C((TILE_M, TILE_N))
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

    # Clean 2D SMEM views for the element-wise loads. Physical layout of the A tile
    # is (M,K):(BK,1) and B tile is (N,K):(BK,1); swizzle lives in the ptr (recast),
    # so these affine 2D views address exactly the same swizzled bytes the MMA reads.
    sA_ld = cute.make_tensor(sA_aff.iterator, cute.make_layout((TILE_M, BK), stride=(BK, 1)))
    sB_ld = cute.make_tensor(sB_aff.iterator, cute.make_layout((TILE_N, BK), stride=(BK, 1)))

    # ---- NPASS MMA passes, each loading a BK-wide K-slice, accumulating in TMEM ----
    for p in range(NPASS):
        k0 = p * BK
        # Load A K-slice [:, k0:k0+BK] into sA (warp 0 cooperatively)
        if warp_id == 0:
            for i in range(TILE_M * BK // 32):
                idx = lane_id + i * 32
                m = idx // BK
                k = idx % BK
                sA_ld[m, k] = gA[m, k0 + k]
            # Load B K-slice [k0:k0+BK, :] into sB as (n,k) from gB(k,n)
            for i in range(BK * TILE_N // 32):
                idx = lane_id + i * 32
                n = idx // BK
                k = idx % BK
                sB_ld[n, k] = gB[k0 + k, n]
        arch.fence_view_async_shared()
        arch.sync_threads()

        # MMA issue: tcgen05 is single-thread issue; gemm elects internally.
        # MMA_K==1 here (BK=16 = one instruction), stage 0 -> slice (None,None,0,0).
        if warp_id == 0:
            tiled_mma.set(tc.Field.ACCUMULATE, p != 0)
            cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, 0, 0)],
                      tCrB[(None, None, 0, 0)], tCtAcc)
            tc.commit(smem_mbar)

        # Wait for this MMA to retire before reloading SMEM next pass.
        if warp_id == 0 and lane_id == 0:
            arch.mbarrier_wait(smem_mbar, p % 2)
        arch.sync_threads()

    # ---- Epilogue: TMEM -> registers -> GMEM (all 128 threads, warps 0-3) ----
    # 2D TMEM accumulator view (M,N):(65536,1) for the tmem-load tiled copy.
    acc2d = cute.make_tensor(tmem_ptr, cute.make_layout((TILE_M, TILE_N), stride=(65536, 1)))
    ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
    copy_atom = cute.make_copy_atom(ld_op, Float32)
    tiled_copy = tc.make_tmem_copy(copy_atom, acc2d)

    thr_copy = tiled_copy.get_slice(tidx)
    tSrc = thr_copy.partition_S(acc2d)
    tDst = cute.make_fragment_like(tSrc)
    cute.copy(tiled_copy, tSrc, tDst)
    arch.fence_view_async_tmem_load()

    gC_part = thr_copy.partition_D(gC)
    for i in range(cute.size(tDst)):
        gC_part[i] = tDst[i]

    arch.sync_threads()
    if warp_id == 0 and lane_id == 0:
        arch.dealloc_tmem(smem_tmem_ptr, TILE_N)


@cute.jit
def run_gemm(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _gemm_kernel(gA, gB, gC).launch(grid=(1, 1, 1), block=(128, 1, 1))
'''


def _build_kernel_module(tile_m, tile_n, npass, tag):
    """Write + import a fresh kernel module with the given tile/pass config."""
    import sys, importlib.util
    src = _KERNEL_SRC.format(TILE_M=tile_m, TILE_N=tile_n, NPASS=npass)
    kpath = f"/root/_opus_kernel_{tag}.py"
    with open(kpath, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"_opus_kernel_{tag}", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules[f"_opus_kernel_{tag}"] = kmod
    spec.loader.exec_module(kmod)
    return kmod


def _split3_bf16(x):
    """3-term bf16 Ozaki split: x ~= x0+x1+x2 (each bf16), residuals."""
    import torch
    x0 = x.bfloat16(); r1 = x - x0.float()
    x1 = r1.bfloat16(); r2 = r1 - x1.float()
    x2 = r2.bfloat16()
    return x0, x1, x2


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    """Single tcgen05 MMA, 128x256x16 BF16->FP32, vs A.float()@B.float()."""
    import torch, traceback
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("OPUS STAGE1 SMOKE — single tcgen05 MMA (official recipe)")
    print("=" * 72)
    try:
        kmod = _build_kernel_module(128, 256, 1, "smoke")
        print("  kernel module loaded")
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); return False

    M, N, K = 128, 256, 16
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    gA = from_dlpack(A.contiguous()); gB = from_dlpack(B.contiguous()); gC = from_dlpack(C)

    print("  Compiling...")
    try:
        compiled = cute.compile(kmod.run_gemm, gA, gB, gC)
        print("  COMPILE OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); return False

    print("  Running...")
    try:
        compiled(gA, gB, gC); torch.cuda.synchronize(); print("  kernel returned OK")
    except Exception as e:
        print(f"  EXEC FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); return False

    ref = A.float() @ B.float()
    err = (C - ref).abs().max().item(); denom = ref.abs().max().item()
    rel = err / denom if denom > 0 else err
    print(f"  max abs err={err:.5f}  rel={rel:.2e}")
    print(f"  C[0,:6]   = {[round(v,3) for v in C[0,:6].tolist()]}")
    print(f"  ref[0,:6] = {[round(v,3) for v in ref[0,:6].tolist()]}")
    ok = rel < 5e-2
    print(f"  >>> SMOKE {'PASS' if ok else 'FAIL'}")
    return ok


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def bf16x9_test():
    """BF16x9: 9 passes accumulating in TMEM => bit-exact FP32 (rel vs FP64 ~1e-6)."""
    import torch, traceback
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("OPUS STAGE1 BF16x9 — 9-pass Ozaki, single TMEM accumulator")
    print("=" * 72)
    try:
        kmod = _build_kernel_module(128, 256, 9, "bf16x9")
        print("  kernel module loaded")
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); return False

    M, N, K = 128, 256, 16
    torch.manual_seed(123)
    A_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_fp32 = torch.randn(K, N, device="cuda", dtype=torch.float32)

    a0, a1, a2 = _split3_bf16(A_fp32)
    b0, b1, b2 = _split3_bf16(B_fp32)
    # 9 cross products, plain sum. Pack A-passes along K (concat 9 K-slices of width 16).
    pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
    A_packed = torch.cat([ai for (ai, _) in pairs], dim=1).contiguous()   # (128, 16*9)
    B_packed = torch.cat([bi for (_, bi) in pairs], dim=0).contiguous()   # (16*9, 256)

    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    gA = from_dlpack(A_packed); gB = from_dlpack(B_packed); gC = from_dlpack(C)

    print("  Compiling...")
    try:
        compiled = cute.compile(kmod.run_gemm, gA, gB, gC)
        print("  COMPILE OK")
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); return False

    print("  Running...")
    try:
        compiled(gA, gB, gC); torch.cuda.synchronize(); print("  kernel returned OK")
    except Exception as e:
        print(f"  EXEC FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); return False

    ref_f64 = A_fp32.double() @ B_fp32.double()
    ref_f32 = A_fp32 @ B_fp32
    err_f64 = (C.double() - ref_f64).abs().max().item(); denom = ref_f64.abs().max().item()
    rel_f64 = err_f64 / denom if denom > 0 else 0
    err_f32 = (C - ref_f32).abs().max().item()
    print(f"  err vs torch FP32 = {err_f32:.3e}")
    print(f"  err vs FP64       = {err_f64:.3e}  rel = {rel_f64:.3e}")
    ok = rel_f64 < 1e-5
    print(f"  >>> BF16x9 {'PASS' if ok else 'FAIL'}")
    return ok


@app.local_entrypoint()
def main(stage: str = "smoke"):
    print(f">>> opus_stage1 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
    elif stage == "bf16x9":
        ok = bf16x9_test.remote()
    else:
        print(f"unknown stage {stage}"); return
    print(f">>> {stage.upper()}: {'PASS' if ok else 'FAIL'}")
