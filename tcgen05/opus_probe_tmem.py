"""
opus_probe_tmem.py — figure out the canonical TMEM accumulator wiring for a
tcgen05 tiled_mma:
  - partition_shape_C / make_fragment_C body
  - how to build a TMEM tensor (cute.make_tensor with tmem ptr) that partition_C accepts
  - arch.alloc_tmem signature + retrieve_tmem_ptr + make_tmem_copy
This compiles a tiny kernel that ONLY builds the tiled_mma + smem layouts + TMEM
fragment and PRINTS shapes/layouts (no MMA issue => cannot hang). timeout=90.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-tmem")

_PROBE = r'''
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
import cutlass.utils.blackwell_helpers as bh
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M = 128
TILE_N = 256
BK = 16

@cute.kernel
def _probe_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16,
        OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N),
    )
    cute.printf("tiled_mma built\n")

    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    print("sA_layout:", sA_layout)
    print("sB_layout:", sB_layout)

    # partition_shape_C
    psC = tiled_mma.partition_shape_C((TILE_M, TILE_N))
    print("partition_shape_C:", psC)
    psA = tiled_mma.partition_shape_A((TILE_M, BK))
    psB = tiled_mma.partition_shape_B((TILE_N, BK))
    print("partition_shape_A:", psA)
    print("partition_shape_B:", psB)

    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=1024)
    sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=1024)
    print("sA tensor:", sA)
    print("sB tensor:", sB)

    # layout is ((M,K),1,1,stage) -> rank 4. Drop the stage (last mode).
    sA0 = sA[None, None, None, 0]
    sB0 = sB[None, None, None, 0]
    print("sA0:", sA0)
    print("sB0:", sB0)

    thr_mma = tiled_mma.get_slice(0)
    tCrA = thr_mma.partition_A(sA0)
    tCrB = thr_mma.partition_B(sB0)
    print("tCrA:", tCrA)
    print("tCrB:", tCrB)

    # TMEM accumulator fragment
    tCtAcc_frag = thr_mma.make_fragment_C(psC)
    print("tCtAcc_frag (make_fragment_C):", tCtAcc_frag)

@cute.jit
def run_probe(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _probe_kernel(gA, gB, gC).launch(grid=(1,1,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import sys, torch, traceback, importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import inspect
    import cutlass.cute.arch as arch

    print("alloc_tmem sig:", inspect.signature(arch.alloc_tmem))
    print("retrieve_tmem_ptr sig:", inspect.signature(arch.retrieve_tmem_ptr))
    try:
        print("make_tmem_copy:", inspect.signature(__import__('cutlass.cute.nvgpu.tcgen05', fromlist=['make_tmem_copy']).make_tmem_copy))
    except Exception as e:
        print("make_tmem_copy sig err:", e)

    kpath = "/root/_opus_probe_tmem.py"
    with open(kpath, "w") as f:
        f.write(_PROBE)
    spec = importlib.util.spec_from_file_location("_opus_probe_tmem", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_opus_probe_tmem"] = kmod
    try:
        spec.loader.exec_module(kmod)
        print("module loaded")
    except Exception as e:
        print("LOAD FAILED:", e); traceback.print_exc(); return

    M, N, K = 128, 256, 16
    A = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.zeros(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    gA = from_dlpack(A.contiguous()); gB = from_dlpack(B.contiguous()); gC = from_dlpack(C)

    print("Compiling probe (build-only, no MMA issue)...")
    try:
        compiled = cute.compile(kmod.run_probe, gA, gB, gC)
        print("COMPILE OK")
    except Exception as e:
        print("COMPILE FAILED:", type(e).__name__, e); traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
