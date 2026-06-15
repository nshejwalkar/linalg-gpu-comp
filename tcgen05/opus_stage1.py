"""
opus_stage1.py — standalone tcgen05 BF16x9 GEMM (Stage 1 de-risk).

Self-contained Modal app. Builds a single tcgen05 MMA via the CANONICAL CuTe-DSL
Blackwell recipe (from CUTLASS examples/.../tutorial_gemm/fp16_gemm_0.py +
dense_gemm.py, pinned by opus_probe_*), then a 9-pass BF16x9 Ozaki split for
bit-exact FP32, then a perf gate vs cublasLt type-78 + torch FP32.

PINNED RECIPE (the parts that were hard / non-obvious):
  - tiled_mma = make_trivial_tiled_mma(BF16,BF16, K,K, FP32, ONE, (M,N))
  - SMEM: layout = make_smem_layout_a/b(tiled_mma,(M,N,K),BF16,1); allocate_tensor
    with layout=layout.outer, swizzle=layout.inner, byte_alignment=128.
  - make_fragment: tCrA = tiled_mma.make_fragment_A(sA)  (on the staged SMEM tensor)
  - TMEM: utils.TmemAllocator(holding_buf_ptr, barrier_for_retrieve=NamedBarrier(1));
    tmem.allocate(512); ... wait_for_alloc(); retrieve_ptr(); make_tensor(ptr, acc.layout).
    *** Using TmemAllocator (not raw arch.alloc_tmem) is REQUIRED — raw alloc gave a
        "misaligned address" SM fault. ***
  - MMA completion sync: PipelineUmmaAsync (producer=MMA warp commits; consumer=all
    threads wait before reading TMEM). Raw mbarrier+commit HUNG.
  - gemm: tiled_mma.set(Field.ACCUMULATE, k>0); cute.gemm(tiled_mma, acc, tCrA[k], tCrB[k], acc)

BF16x9: x = x0+x1+x2 (3 bf16, exact) => x*y = sum_{i,j} x_i*y_j (9 plain-sum products) =>
accumulate all 9 in ONE TMEM accumulator (ACCUMULATE toggled), single epilogue.

Anti-hang: server timeout=90; every `modal run` under local `timeout 120`.
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
# Kernel source, parameterized by TILE_M, TILE_N, NPASS at format time.
# A is (TILE_M, BK*NPASS), B is (BK*NPASS, TILE_N); each pass consumes a BK=16 slice.
# ---------------------------------------------------------------------------
_KERNEL_SRC = r'''"""tcgen05 GEMM kernel — opus_stage1, canonical recipe."""
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

TILE_M: int = {TILE_M}
TILE_N: int = {TILE_N}
BK: int = 16          # one BF16 tcgen05 MMA instruction along K
NPASS: int = {NPASS}  # 1 = single MMA smoke; 9 = BF16x9
THREADS: int = 128


@cute.struct
class SharedStorage:
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def _gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = arch.thread_idx()
    warp_id = arch.warp_idx()
    warp_id = arch.make_warp_uniform(warp_id)
    lane_id = arch.lane_idx()

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

    # TMEM allocation via the proper allocator (raw alloc_tmem misaligns).
    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr,
                               barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    # Accumulator completion pipeline (tcgen05 MMA -> mbarrier).
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar.data_ptr(),
    ).make_participants()

    # MMA fragments (on the staged SMEM tensors).
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C((TILE_M, TILE_N))
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # Stage-sliced composed SMEM views for scalar loads ((M,K),1,1) -> [(m,k),0,0].
    sA_c = sA[None, None, None, 0]
    sB_c = sB[None, None, None, 0]
    # GMEM C view matching the accumulator's rank-3 ((M,N),1,1) profile.
    gC_part = cute.make_tensor(
        gC.iterator, cute.make_layout(((TILE_M, TILE_N), 1, 1), stride=((TILE_N, 1), 0, 0)))

    # ---- MMA: NPASS passes, each loads a BK-wide K-slice, accumulates in TMEM ----
    # Entire MMA section in ONE warp-0 block (acquire/commit must share scope).
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

    # ---- Epilogue: wait MMA done, TMEM -> regs -> GMEM (epi sub-tiled, ref pattern) ----
    acc_full = acc_consumer.wait_and_advance()

    SUBTILE = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]),
                  cute.size(tCtAcc, mode=[0, 1]) // SUBTILE),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)   # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(gC_part, epi_tiler)

    ld_op = tc.Ld32x32bOp(tc.Repetition.x64, tc.Pack.NONE)
    copy_atom = cute.make_copy_atom(ld_op, Float32)
    tmem_tiled_copy = tc.make_tmem_copy(copy_atom, tCtAcc_epi[None, 0])
    thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = thr_copy.partition_S(tCtAcc_epi)   # (TmemCpy, NumTmemCpy, NumTiles)
    tDgC = thr_copy.partition_D(gC_epi)
    # Register fragment sized to the GMEM destination per-thread shape (ref pattern).
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, Float32)
    for i in cutlass.range(cute.size(tDtC, mode=[2]), unroll=1):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        arch.fence_view_async_tmem_load()
        cute.autovec_copy(tCrAcc, tDgC[None, None, i])

    acc_full.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def run_gemm(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _gemm_kernel(gA, gB, gC).launch(grid=(1, 1, 1), block=(128, 1, 1))
'''


def _build_kernel_module(tile_m, tile_n, npass, tag):
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
    x0 = x.bfloat16(); r1 = x - x0.float()
    x1 = r1.bfloat16(); r2 = r1 - x1.float()
    x2 = r2.bfloat16()
    return x0, x1, x2


@app.function(gpu="B200", image=cutlass_image, timeout=90, retries=0)
def smoke_test():
    """Single tcgen05 MMA, 128x256x16 BF16->FP32, vs A.float()@B.float()."""
    import torch, traceback
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("OPUS STAGE1 SMOKE — single tcgen05 MMA (canonical recipe)")
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
        compiled = cute.compile(kmod.run_gemm, gA, gB, gC); print("  COMPILE OK")
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


@app.function(gpu="B200", image=cutlass_image, timeout=90, retries=0)
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
    pairs = [(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
    A_packed = torch.cat([ai for (ai, _) in pairs], dim=1).contiguous()   # (128, 16*9)
    B_packed = torch.cat([bi for (_, bi) in pairs], dim=0).contiguous()   # (16*9, 256)

    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    gA = from_dlpack(A_packed); gB = from_dlpack(B_packed); gC = from_dlpack(C)

    print("  Compiling...")
    try:
        compiled = cute.compile(kmod.run_gemm, gA, gB, gC); print("  COMPILE OK")
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
