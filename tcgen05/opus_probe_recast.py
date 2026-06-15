"""
opus_probe_recast.py — figure out the swizzle-on-pointer pattern:
  recast_ptr(ptr, swizzle, dtype) so the SMEM tensor's LAYOUT is affine and
  make_fragment_A/B accepts it. Build-only (no MMA issue) => no hang.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-recast")

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
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N),
    )
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    print("sA_layout:", sA_layout)
    print("type(sA_layout):", type(sA_layout))
    # Composed layout: .inner = swizzle, .outer = affine layout
    print("sA_layout.inner (swizzle):", sA_layout.inner)
    print("sA_layout.outer (affine):", sA_layout.outer)

    smem = SmemAllocator()
    # Allocate with the FULL composed layout (gives swizzled tensor as before)
    sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=1024)
    sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=1024)
    print("sA (composed tensor):", sA)
    print("sA.iterator:", sA.iterator)

    # Move swizzle to the pointer, then view with affine layout
    sw_a = sA_layout.inner
    affine_a = sA_layout.outer
    pA = cute.recast_ptr(sA.iterator, sw_a, BFloat16)
    print("pA (recast):", pA)
    sA_affine = cute.make_tensor(pA, affine_a)
    print("sA_affine:", sA_affine)

    thr_mma = tiled_mma.get_slice(0)
    sA0 = sA_affine[None, None, None, 0]
    print("sA0:", sA0)
    tCsA = thr_mma.partition_A(sA0)
    print("tCsA (partition_A on affine):", tCsA)
    tCrA = thr_mma.make_fragment_A(tCsA)
    print("tCrA (make_fragment_A) OK:", tCrA)

@cute.jit
def run_probe(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _probe_kernel(gA, gB, gC).launch(grid=(1,1,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import sys, torch, traceback, importlib.util, inspect
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    print("recast_ptr sig:", inspect.signature(cute.recast_ptr))

    kpath = "/root/_opus_probe_recast.py"
    with open(kpath, "w") as f:
        f.write(_PROBE)
    spec = importlib.util.spec_from_file_location("_opus_probe_recast", kpath)
    kmod = importlib.util.module_from_spec(spec)
    sys.modules["_opus_probe_recast"] = kmod
    try:
        spec.loader.exec_module(kmod); print("module loaded")
    except Exception as e:
        print("LOAD FAILED:", e); traceback.print_exc(); return

    M, N, K = 128, 256, 16
    A = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.zeros(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    gA = from_dlpack(A.contiguous()); gB = from_dlpack(B.contiguous()); gC = from_dlpack(C)
    print("Compiling (build-only)...")
    try:
        cute.compile(kmod.run_probe, gA, gB, gC); print("COMPILE OK")
    except Exception as e:
        print("COMPILE FAILED:", type(e).__name__, e); traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
