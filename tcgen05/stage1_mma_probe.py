"""
stage1_mma_probe.py — probe how to call cute.gemm with tcgen05/TMEM accumulator.

Specifically probes:
  1. make_trivial_tiled_mma signature + result type
  2. cute.gemm signature
  3. TiledMma methods (partition_A, partition_B, partition_C, get_slice)
  4. What shape TMEM accumulator looks like from tiled_mma

Anti-hang: Modal timeout=90s; local `timeout 120 modal run ...`
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-mma-probe")

_PROBE_SRC = r'''"""
Probe: make_trivial_tiled_mma + cute.gemm call pattern.
Minimal 128x256x64 BF16->FP32 MMA using the higher-level TiledMma API.
"""
import inspect
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.utils import SmemAllocator
from cutlass.utils.blackwell_helpers import make_trivial_tiled_mma
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M: int = 128
TILE_N: int = 256
TILE_K: int = 64
BK: int = 16

# Build TiledMma object at Python-level (before @cute.kernel).
# make_trivial_tiled_mma(op, (M,N,K), cta_group, a_src, a_major, b_major)
tiled_mma = make_trivial_tiled_mma(
    tc.MmaF16BF16Op(
        BFloat16, Float32,
        (TILE_M, TILE_N, BK),
        tc.CtaGroup.ONE,
        tc.OperandSource.SMEM,
        tc.OperandMajorMode.K,
        tc.OperandMajorMode.K,
    )
)


@cute.kernel
def _probe_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    """Probe: uses make_trivial_tiled_mma + cute.gemm."""
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

    # MMA warp using TiledMma + cute.gemm
    if warp_id == 2:
        with arch.elect_one():
            # Build SMEM descriptors
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

            # Build TMEM accumulator tensor via retrieve_tmem_ptr.
            tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)
            tCtAcc = cute.make_tensor(
                tmem_ptr,
                cute.make_layout((TILE_M, TILE_N), stride=(65536, 1))
            )

            # Call cute.gemm with the tiled_mma, smem descriptors, TMEM accum.
            # K-loop: N_K = TILE_K // BK = 4
            # Pattern from tcgen05_tmem.md: accumulate_ toggled after first k_block
            for k_it in range(TILE_K // BK):
                k_off = k_it * BK
                # Sub-tensor of sA for this k-block
                sA_k_ptr = arch.retrieve_tmem_ptr  # placeholder - need smem ptr arithmetic
                # Actually use descriptors with k-offset encoded differently.
                # For now, just call once with full K and see the error.
                if k_it == 0:
                    cute.gemm(tiled_mma, desc_a, desc_b, tCtAcc)
                else:
                    cute.gemm(tiled_mma, desc_a, desc_b, tCtAcc)

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
            tGOut[i] = tDst[i]

    arch.sync_threads()

    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(smem_tmem_ptr, TILE_N)


@cute.jit
def run_probe(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _probe_kernel(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    """Probe cute.gemm + TiledMma API."""
    import sys
    import torch
    import traceback
    import inspect
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("=" * 72)
    print("MMA API PROBE — cute.gemm + make_trivial_tiled_mma + TMEM")
    print("=" * 72)

    # Probe cute.gemm signature
    gemm_fn = getattr(cute, "gemm", None)
    if gemm_fn:
        try:
            print(f"  cute.gemm{inspect.signature(gemm_fn)}")
        except Exception as e:
            print(f"  cute.gemm: sig unavail ({e})")
    else:
        print("  cute.gemm: NOT FOUND")

    # Probe TiledMma methods
    from cutlass.utils.blackwell_helpers import make_trivial_tiled_mma
    from cutlass.cutlass_dsl import BFloat16, Float32
    import cutlass.cute.nvgpu.tcgen05 as tc

    mma_op = tc.MmaF16BF16Op(
        BFloat16, Float32, (128, 256, 16),
        tc.CtaGroup.ONE, tc.OperandSource.SMEM,
        tc.OperandMajorMode.K, tc.OperandMajorMode.K,
    )
    print(f"  MmaF16BF16Op instance: {mma_op}")
    print(f"  MmaF16BF16Op type: {type(mma_op)}")
    print(f"  MmaF16BF16Op attrs: {[a for a in dir(mma_op) if not a.startswith('_')]}")

    # Check if MmaF16BF16Op is a MmaOp or has a different base
    import cutlass.cute.atom as atom_mod
    print(f"  MmaOp type: {getattr(atom_mod, 'MmaOp', 'NOT FOUND')}")

    try:
        tiled_mma = make_trivial_tiled_mma(mma_op)
        print(f"  make_trivial_tiled_mma(op) -> {type(tiled_mma)}")
        print(f"  TiledMma methods: {[a for a in dir(tiled_mma) if not a.startswith('_')]}")
    except Exception as e:
        print(f"  make_trivial_tiled_mma FAILED: {e}")
        traceback.print_exc()

    # Try compiling the probe kernel
    kpath = "/root/_mma_probe_kernel.py"
    with open(kpath, "w") as f:
        f.write(_PROBE_SRC)

    spec = importlib.util.spec_from_file_location("_mma_probe_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_mma_probe_kernel"] = kmod
    try:
        spec.loader.exec_module(kmod)
        print("  Module load OK")
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        traceback.print_exc()
        return

    M, N, K = 128, 256, 64
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    gA = from_dlpack(A.contiguous())
    gB = from_dlpack(B.contiguous())
    gC = from_dlpack(C)

    print("  Compiling probe kernel...")
    try:
        compiled = cute.compile(kmod.run_probe, gA, gB, gC)
        print("  cute.compile OK")
        compiled(gA, gB, gC)
        torch.cuda.synchronize()
        ref = A.float() @ B.float()
        err = (C - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-9)
        print(f"  rel_err={rel:.3e}")
        print(f"  PROBE PASS (no hang)")
    except Exception as e:
        print(f"  COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
