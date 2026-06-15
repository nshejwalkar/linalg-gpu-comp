"""Probe TMEM address handling and Ld32x32bOp call signature."""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-tmem-probe")

_PROBE_KERNEL_SRC = '''
"""Probe: use retrieve_tmem_ptr to get a TMEM Pointer and call Ld32x32bOp on it."""
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.utils import SmemAllocator
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32, Uint64

TILE_M: int = 128
TILE_N: int = 32   # smallest valid alloc
N_COLS: int = 32


@cute.kernel
def _probe_kernel(
    gOut: cute.Tensor,  # (128, 32) FP32 output
):
    """Probe: alloc TMEM, init with zeros, read back."""
    warp_id = arch.warp_idx()
    lane_id = arch.lane_idx()

    smem = SmemAllocator()
    smem_tmem_ptr = smem.allocate(Uint32)
    smem_mbar = smem.allocate(Uint64)

    if warp_id == 0 and lane_id == 0:
        arch.mbarrier_init(smem_mbar, 1)

    if warp_id == 1 and lane_id == 0:
        arch.alloc_tmem(N_COLS, smem_tmem_ptr)
        arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()

    # Warp 2: "MMA" that does nothing, just commit the barrier.
    # (In real use, MMA fills TMEM. Here TMEM is zero-initialized.)
    if warp_id == 2 and lane_id == 0:
        tc.commit(smem_mbar)

    arch.sync_threads()

    # Warps 4-7: epilogue - read from TMEM using retrieve_tmem_ptr.
    if warp_id >= 4:
        warp_epi = warp_id - 4
        lane_base = warp_epi * 32

        if lane_id == 0:
            arch.mbarrier_wait(smem_mbar, 0)
        arch.sync_warp()

        # Get a typed TMEM pointer via retrieve_tmem_ptr.
        # This takes the uint32 smem slot and returns a typed TMEM Pointer.
        tmem_fp32_ptr = arch.retrieve_tmem_ptr(Float32, 128, smem_tmem_ptr)

        # Make a TMEM tensor from the pointer.
        # Layout: (128 lanes, N_COLS columns), stride = (1<<16, 1) because
        # TMEM addr = (lane << 16) | col.
        tAcc = cute.make_tensor(
            tmem_fp32_ptr,
            cute.make_layout((TILE_M, N_COLS), stride=(65536, 1))
        )

        # Use make_tmem_copy to create a tiled copy for TMEM->reg.
        ld_op = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
        tiled_copy_t2r = tc.make_tmem_copy(ld_op, tAcc)

        # thr_idx for partitioning
        tidx, _, _ = arch.thread_idx()

        # Partition the TMEM tensor and destination registers.
        thr_copy = tiled_copy_t2r.get_slice(tidx)
        tSrc = thr_copy.partition_S(tAcc)  # source: TMEM partition
        tDst = cute.make_fragment_like(tSrc)  # destination: registers

        # Copy.
        cute.copy(tiled_copy_t2r, tSrc, tDst)
        arch.fence_view_async_tmem_load()

        # Write to output for verification.
        tGOut = thr_copy.partition_D(
            cute.make_tensor(
                gOut.iterator,
                cute.make_layout((TILE_M, N_COLS), stride=(N_COLS, 1))
            )
        )
        # Store from registers to gmem.
        for i in range(cute.size(tDst)):
            tGOut[i] = tDst[i]

    arch.sync_threads()

    if warp_id == 1 and lane_id == 0:
        arch.dealloc_tmem(smem_tmem_ptr, N_COLS)


@cute.jit
def run_probe(gOut: cute.Tensor):
    _probe_kernel(gOut).launch(grid=(1,1,1), block=(256,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe_tmem():
    """Probe TMEM allocation + retrieve_tmem_ptr + make_tmem_copy."""
    import sys
    import torch
    import traceback
    import importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import inspect

    print("=" * 72)
    print("TMEM PROBE — retrieve_tmem_ptr + make_tmem_copy + Ld32x32bOp")
    print("=" * 72)

    # First inspect retrieve_tmem_ptr + make_tmem_copy signatures.
    import cutlass.cute.arch as arch
    import cutlass.cute.nvgpu.tcgen05 as tc

    obj = getattr(arch, "retrieve_tmem_ptr", None)
    if obj:
        try:
            print(f"  retrieve_tmem_ptr{inspect.signature(obj)}")
        except:
            print(f"  retrieve_tmem_ptr: {obj}")

    obj = getattr(tc, "make_tmem_copy", None)
    if obj:
        try:
            print(f"  make_tmem_copy{inspect.signature(obj)}")
        except:
            print(f"  make_tmem_copy: {obj}")

    # Also dump Ld32x32bOp instance methods
    ld = tc.Ld32x32bOp(tc.Repetition.x1, tc.Pack.NONE)
    print(f"  Ld32x32bOp instance: {ld}")
    print(f"  Ld32x32bOp attrs: {[a for a in dir(ld) if not a.startswith('_')]}")
    try:
        print(f"  Ld32x32bOp.__call__{inspect.signature(ld.__call__)}")
    except:
        print("  Ld32x32bOp.__call__: no sig")

    # Try to compile the probe kernel.
    kpath = "/root/_tmem_probe_kernel.py"
    with open(kpath, "w") as f:
        f.write(_PROBE_KERNEL_SRC)

    spec = importlib.util.spec_from_file_location("_tmem_probe_kernel", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_tmem_probe_kernel"] = kmod
    try:
        spec.loader.exec_module(kmod)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    import torch
    Out = torch.full((128, 32), -1.0, device="cuda", dtype=torch.float32)
    gOut = from_dlpack(Out)

    print("  Compiling probe kernel...")
    try:
        compiled = cute.compile(kmod.run_probe, gOut)
        print("  cute.compile OK")
        compiled(gOut)
        torch.cuda.synchronize()
        print(f"  Out[0,:4] = {Out[0,:4].tolist()} (expect zeros from uninit TMEM)")
        print("  >>> PROBE PASS (no hang)")
    except Exception as e:
        print(f"  COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    probe_tmem.remote()
