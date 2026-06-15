"""
opus_probe_load.py — isolate the misaligned-address bug. Two LAUNCHED kernels:
  K_plain:  GMEM->SMEM (PLAIN affine, no swizzle) ->GMEM round-trip. Verify load logic.
  K_swz:    GMEM->SMEM (swizzled via make_smem_layout_a + recast) ->GMEM round-trip.
No MMA, no TMEM => if K_swz misaligns, the scalar swizzled store is the culprit.
Anti-hang: server timeout=90, local timeout 120.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-load")

_K = r'''
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
import cutlass.utils.blackwell_helpers as bh
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M = 128; BK = 16

@cute.kernel
def _k_plain(gA: cute.Tensor, gOut: cute.Tensor):
    warp_id = arch.warp_idx(); lane_id = arch.lane_idx()
    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, cute.make_layout((TILE_M, BK), stride=(BK,1)), byte_alignment=1024)
    if warp_id == 0:
        for i in range(TILE_M*BK//32):
            idx = lane_id + i*32; m = idx//BK; k = idx%BK
            sA[m,k] = gA[m,k]
    arch.fence_view_async_shared(); arch.sync_threads()
    if warp_id == 0:
        for i in range(TILE_M*BK//32):
            idx = lane_id + i*32; m = idx//BK; k = idx%BK
            gOut[m,k] = sA[m,k].to(Float32)

@cute.kernel
def _k_swz(gA: cute.Tensor, gOut: cute.Tensor):
    warp_id = arch.warp_idx(); lane_id = arch.lane_idx()
    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, 256))
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, 256, BK), BFloat16, 1)
    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=1024)
    sA_aff = cute.make_tensor(cute.recast_ptr(sA.iterator, sA_layout.inner, BFloat16), sA_layout.outer)
    sA_ld = cute.make_tensor(sA_aff.iterator, cute.make_layout((TILE_M, BK), stride=(BK,1)))
    if warp_id == 0:
        for i in range(TILE_M*BK//32):
            idx = lane_id + i*32; m = idx//BK; k = idx%BK
            sA_ld[m,k] = gA[m,k]
    arch.fence_view_async_shared(); arch.sync_threads()
    if warp_id == 0:
        for i in range(TILE_M*BK//32):
            idx = lane_id + i*32; m = idx//BK; k = idx%BK
            gOut[m,k] = sA_ld[m,k].to(Float32)

@cute.jit
def run_plain(gA: cute.Tensor, gOut: cute.Tensor):
    _k_plain(gA, gOut).launch(grid=(1,1,1), block=(128,1,1))

@cute.jit
def run_swz(gA: cute.Tensor, gOut: cute.Tensor):
    _k_swz(gA, gOut).launch(grid=(1,1,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import sys, torch, traceback, importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    with open("/root/_opus_probe_load.py","w") as f:
        f.write(_K)
    spec = importlib.util.spec_from_file_location("_opus_probe_load","/root/_opus_probe_load.py")
    kmod = importlib.util.module_from_spec(spec); sys.modules["_opus_probe_load"]=kmod
    spec.loader.exec_module(kmod)
    M,K = 128,16
    torch.manual_seed(0)
    A = torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
    for name, fn in [("PLAIN", kmod.run_plain), ("SWZ", kmod.run_swz)]:
        Out = torch.zeros(M,K,device="cuda",dtype=torch.float32)
        gA = from_dlpack(A.contiguous()); gOut = from_dlpack(Out)
        print(f"\n=== {name} ===")
        try:
            c = cute.compile(fn, gA, gOut); print("  compiled")
            c(gA, gOut); torch.cuda.synchronize()
            err = (Out - A.float()).abs().max().item()
            print(f"  roundtrip max err = {err:.3e}  -> {'OK' if err < 1e-2 else 'MISMATCH'}")
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:200]}")


@app.local_entrypoint()
def main():
    probe.remote()
