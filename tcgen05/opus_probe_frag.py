"""
opus_probe_frag.py — build-only: test the make_fragment_A/B (smem_desc_view) + D
candidates into cute.gemm. No launch => no hang.

From errors so far:
 - gemm wants A/B as smem_desc_view (=> make_fragment_A/B output), affine swizzle-in-ptr.
 - make_fragment_A(partition_A(sA0)) -> smem_desc_view shape (1,2,1,1).
 - D candidates: partition_C(acc) -> ((M,N),256,1,1) rank4 ; make_fragment_C(psC) -> ((M,N),1,1) rank3.
Test combos:
  C1: A=mkfragA(partA), B=mkfragB(partB), D=partition_C(acc)        [rank4 D]
  C2: A=mkfragA(partA), B=mkfragB(partB), D=make_fragment_C@tmem    [rank3 D]
Each in its own try/except; print which VERIFIES.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-frag")

# We compile two separate kernels (one per combo) so a failure in C1 doesn't block C2.
_K = r'''
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
import cutlass.utils.blackwell_helpers as bh
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M = 128; TILE_N = 256; BK = 16

def _setup():
    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N))
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=1024)
    sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=1024)
    smp = smem.allocate(Uint32)
    return tiled_mma, sA_layout, sB_layout, sA, sB, smp

@cute.kernel
def _kC1(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    warp_id = arch.warp_idx(); lane_id = arch.lane_idx()
    tiled_mma, sA_layout, sB_layout, sA, sB, smp = _setup()
    if warp_id == 0 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smp); arch.relinquish_tmem_alloc_permit()
    arch.sync_threads()
    tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smp)
    sA_aff = cute.make_tensor(cute.recast_ptr(sA.iterator, sA_layout.inner, BFloat16), sA_layout.outer)
    sB_aff = cute.make_tensor(cute.recast_ptr(sB.iterator, sB_layout.inner, BFloat16), sB_layout.outer)
    thr = tiled_mma.get_slice(0)
    tCrA = thr.make_fragment_A(thr.partition_A(sA_aff[None,None,None,0]))
    tCrB = thr.make_fragment_B(thr.partition_B(sB_aff[None,None,None,0]))
    acc = cute.make_tensor(tmem_ptr, cute.make_layout(((TILE_M,TILE_N),1,1), stride=((65536,1),0,0)))
    tCtAcc = thr.partition_C(acc)
    print("C1 tCrA:", tCrA); print("C1 tCtAcc:", tCtAcc)
    tiled_mma.set(tc.Field.ACCUMULATE, False)
    cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
    print("C1 VERIFIED")

@cute.kernel
def _kC2(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    warp_id = arch.warp_idx(); lane_id = arch.lane_idx()
    tiled_mma, sA_layout, sB_layout, sA, sB, smp = _setup()
    if warp_id == 0 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smp); arch.relinquish_tmem_alloc_permit()
    arch.sync_threads()
    tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smp)
    sA_aff = cute.make_tensor(cute.recast_ptr(sA.iterator, sA_layout.inner, BFloat16), sA_layout.outer)
    sB_aff = cute.make_tensor(cute.recast_ptr(sB.iterator, sB_layout.inner, BFloat16), sB_layout.outer)
    thr = tiled_mma.get_slice(0)
    tCrA = thr.make_fragment_A(thr.partition_A(sA_aff[None,None,None,0]))
    tCrB = thr.make_fragment_B(thr.partition_B(sB_aff[None,None,None,0]))
    psC = tiled_mma.partition_shape_C((TILE_M, TILE_N))
    accF = thr.make_fragment_C(psC)
    acc = cute.make_tensor(tmem_ptr, accF.layout)
    print("C2 tCrA:", tCrA); print("C2 acc:", acc)
    tiled_mma.set(tc.Field.ACCUMULATE, False)
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)
    print("C2 VERIFIED")

@cute.jit
def runC1(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _kC1(gA, gB, gC).launch(grid=(1,1,1), block=(128,1,1))

@cute.jit
def runC2(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _kC2(gA, gB, gC).launch(grid=(1,1,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import sys, torch, traceback, importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    with open("/root/_opus_probe_frag.py", "w") as f:
        f.write(_K)
    spec = importlib.util.spec_from_file_location("_opus_probe_frag", "/root/_opus_probe_frag.py")
    kmod = importlib.util.module_from_spec(spec); sys.modules["_opus_probe_frag"] = kmod
    spec.loader.exec_module(kmod)
    M, N, K = 128, 256, 16
    A = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.zeros(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    gA = from_dlpack(A.contiguous()); gB = from_dlpack(B.contiguous()); gC = from_dlpack(C)
    for name, fn in [("C1 (D=partition_C rank4)", kmod.runC1), ("C2 (D=make_fragment_C rank3)", kmod.runC2)]:
        print("\n" + "=" * 60)
        print("TRY", name)
        try:
            cute.compile(fn, gA, gB, gC); print(f">>> {name} COMPILE OK")
        except Exception as e:
            print(f">>> {name} FAILED: {type(e).__name__}")
            msg = str(e)
            # print just the verifier error line
            for line in msg.split("\n"):
                if "cute.gemm' op" in line or "error:" in line:
                    print("   ", line.strip()[:300])


@app.local_entrypoint()
def main():
    probe.remote()
