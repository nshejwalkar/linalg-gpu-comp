"""
stage1_v9.py — tcgen05 BF16x9 GEMM, v9.

KEY FIX attempt 2:
- cute.gemm calls _cute_ir.gemm(atom_val, d.value, a_vals, b_vals, c.value)
- _cute_ir.gemm accepts raw ir.Values in a_vals/b_vals (it's the MLIR op, no isinstance check)
- So bypass cute.gemm Python wrapper entirely and call _cute_ir.gemm directly

Import path:
  from cutlass._mlir.dialects.cute import gemm as _cute_gemm

But we also need atom._unpack() to get atom_val.
atom._unpack() is defined in atom.py - what does it return?

Actually, the simplest: patch the call to use mma_atom_call directly from atom.py,
but also bypass _normalize_variadic_tensor_operand.

OR: call _cute_ir.gemm directly with:
  - atom._unpack(...) for the atom value
  - d.value, c.value for d/c
  - [desc_a_val] for a_vals (raw ir.Value list)
  - [desc_b_val] for b_vals (raw ir.Value list)

Let me also look at what atom._unpack() does.
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tcgen05-stage1-v9")


_KERNEL_SRC = r'''"""
tcgen05 BF16->FP32 v9.
Key: call _cute_ir.gemm directly, bypassing Python-level isinstance checks.
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
BK: int = 16   # one instruction


@cute.kernel
def _mma_kernel_clear(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    """Single tcgen05 MMA, K=16, overwrite TMEM."""
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

            # Build SMEM descriptors
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

            # Bypass cute.gemm Python wrapper; call _cute_ir.gemm directly.
            # cute.gemm implementation (algorithm.py line 153-156):
            #   value = atom._unpack(loc=loc, ip=ip, **kwargs)
            #   a_vals = [t.value for t in a_list]
            #   b_vals = [t.value for t in b_list]
            #   return _cute_ir.gemm(value, d.value, a_vals, b_vals, c.value)
            #
            # Here: desc_a_val and desc_b_val ARE ir.Values from make_umma_smem_desc
            # tCtAcc.value is the TMEM tensor ir.Value
            atom_val = mma_atom._unpack()
            _cute_ir.gemm(
                atom_val,
                tCtAcc.value,   # d
                [desc_a_val],   # a_vals (list of ir.Value)
                [desc_b_val],   # b_vals
                tCtAcc.value,   # c
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
def run_mma_clear(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _mma_kernel_clear(gA, gB, gC).launch(grid=(1, 1, 1), block=(256, 1, 1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def smoke_test():
    """Smoke: 128x256x16 BF16->FP32 one MMA tile (direct _cute_ir.gemm)."""
    import sys
    import torch
    import traceback
    import importlib.util
    import inspect
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.atom import MmaAtom

    print("=" * 72)
    print("STAGE 1 V9 SMOKE — direct _cute_ir.gemm bypass")
    print("=" * 72)

    # First probe atom._unpack
    print("\n=== Probing MmaAtom._unpack ===")
    for name in ['_unpack', 'unpack', '_get_value', 'value']:
        val = getattr(MmaAtom, name, None)
        if val:
            print(f"  MmaAtom.{name}: {val}")
            try:
                print(f"    sig: {inspect.signature(val)}")
            except:
                pass
    # Also check _cute_ir.gemm signature
    import cutlass._mlir.dialects.cute as _cute_ir
    g = getattr(_cute_ir, 'gemm', None)
    if g:
        try:
            print(f"  _cute_ir.gemm sig: {inspect.signature(g)}")
        except Exception as e:
            print(f"  _cute_ir.gemm: {e}")

    kpath = "/root/_stage1_v9_kernel.py"
    with open(kpath, "w") as f:
        f.write(_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_stage1_v9_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_v9_kernel"] = kmod
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

    print("\n  Compiling run_mma_clear...")
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


@app.local_entrypoint()
def main(stage: str = "smoke"):
    print(f">>> stage1_v9 stage={stage}")
    if stage == "smoke":
        ok = smoke_test.remote()
        print(f">>> SMOKE: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"Unknown stage: {stage}")
