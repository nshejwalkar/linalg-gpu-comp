"""
opus_probe_swz.py — is the SMEM SWIZZLE the cause of the MMA misalign?
Test two SMEM layouts for the same single MMA (load+gemm+dummy write, no tmem read):
  K_INTER (no swizzle) vs K_SW32 (heuristic). retries=0 so a GPU fault does NOT retry 8x.
If K_INTER launches clean and K_SW32 faults => swizzle/descriptor mismatch is the bug.

make_smem_layout_atom(kind, dtype) + tile_to_mma_shape gives a layout for a chosen kind.
We bypass make_smem_layout_a's heuristic and build A/B layouts with an explicit kind.
Anti-hang: server timeout=90 + retries=0; local timeout 120.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-swz")

_K = r'''
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.cute.nvgpu.tcgen05 as tc
import cutlass.utils as utils
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.utils import SmemAllocator
import cutlass.utils.blackwell_helpers as bh
from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

TILE_M = 128; TILE_N = 256; BK = 16

def _build(mode):
    @cute.kernel
    def _k(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
        warp_id = arch.warp_idx(); lane_id = arch.lane_idx()
        tiled_mma = bh.make_trivial_tiled_mma(
            BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
            Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N))
        if cutlass.const_expr(mode == "plain"):
            # NON-swizzled affine SMEM, K-major contiguous (M,K):(BK,1)
            sA_layout = cute.make_layout((TILE_M, BK), stride=(BK, 1))
            sB_layout = cute.make_layout((TILE_N, BK), stride=(BK, 1))
        else:
            sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
            sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
        warp_id = arch.make_warp_uniform(warp_id)
        smem = SmemAllocator()
        # REFERENCE allocation: affine OUTER layout + swizzle passed SEPARATELY.
        sA = smem.allocate_tensor(element_type=BFloat16, layout=sA_layout.outer,
                                  byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(element_type=BFloat16, layout=sB_layout.outer,
                                  byte_alignment=128, swizzle=sB_layout.inner)
        smp = smem.allocate(Uint32); mbar = smem.allocate(cutlass.Uint64)
        if warp_id==0 and lane_id==0: arch.mbarrier_init(mbar,1)
        # LOAD through the stage-sliced sA/sB (swizzle baked into the tensor by the
        # allocator => descriptor-consistent). ((M,K),1,1) -> index [(m,k),0,0].
        sA_c = sA[None, None, None, 0]
        sB_c = sB[None, None, None, 0]
        if warp_id==0:
            for i in range(TILE_M*BK//32):
                idx=lane_id+i*32; m=idx//BK; k=idx%BK; sA_c[(m,k),0,0]=gA[m,k]
            for i in range(BK*TILE_N//32):
                idx=lane_id+i*32; n=idx//BK; k=idx%BK; sB_c[(n,k),0,0]=gB[k,n]
        sA_aff = sA; sB_aff = sB
        arch.fence_view_async_shared()
        if warp_id==0 and lane_id==0: arch.alloc_tmem(256, smp)
        arch.sync_threads()
        tCrA = tiled_mma.make_fragment_A(sA_aff)
        tCrB = tiled_mma.make_fragment_B(sB_aff)
        acc_shape = tiled_mma.partition_shape_C((TILE_M,TILE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tmem_ptr = arch.retrieve_tmem_ptr(Float32,128,smp)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        if warp_id==0:
            tiled_mma.set(tc.Field.ACCUMULATE, False)
            cute.gemm(tiled_mma, tCtAcc, tCrA[(None,None,0,0)], tCrB[(None,None,0,0)], tCtAcc)
            tc.commit(mbar)
        if warp_id==0 and lane_id==0:
            arch.mbarrier_wait(mbar,0); arch.relinquish_tmem_alloc_permit()
        arch.sync_threads()
        if warp_id==0:
            for i in range(TILE_M*TILE_N//32):
                idx=lane_id+i*32; m=idx//TILE_N; n=idx%TILE_N; gC[m,n]=Float32(5.0)
        arch.sync_threads()
        if warp_id==0 and lane_id==0: arch.dealloc_tmem(tmem_ptr, 256)
    return _k

_kPLAIN = _build("plain")
_kSW32  = _build("sw32")

@cute.jit
def run_inter(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _kPLAIN(gA,gB,gC).launch(grid=(1,1,1), block=(128,1,1))
@cute.jit
def run_sw32(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _kSW32(gA,gB,gC).launch(grid=(1,1,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90, retries=0)
def probe(which: str):
    import sys, torch, importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    with open("/root/_opus_probe_swz.py","w") as f:
        f.write(_K)
    spec = importlib.util.spec_from_file_location("_opus_probe_swz","/root/_opus_probe_swz.py")
    kmod = importlib.util.module_from_spec(spec); sys.modules["_opus_probe_swz"]=kmod
    spec.loader.exec_module(kmod)
    M,N,K = 128,256,16
    torch.manual_seed(0)
    A = torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
    B = torch.randn(K,N,device="cuda",dtype=torch.bfloat16)
    fn = kmod.run_inter if which=="inter" else kmod.run_sw32
    C = torch.zeros(M,N,device="cuda",dtype=torch.float32)
    gA=from_dlpack(A.contiguous()); gB=from_dlpack(B.contiguous()); gC=from_dlpack(C)
    print(f"=== {which} ===")
    try:
        c = cute.compile(fn, gA, gB, gC); print("  compiled")
        c(gA, gB, gC); torch.cuda.synchronize()
        print(f"  LAUNCH OK gC[0,0]={C[0,0].item()}")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:140]}")
        return False


@app.local_entrypoint()
def main(which: str = "inter"):
    probe.remote(which)
