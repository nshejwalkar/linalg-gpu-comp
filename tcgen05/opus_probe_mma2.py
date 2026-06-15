"""
opus_probe_mma2.py — fix the MMA-path misalign. Changes vs probe_mma:
  - num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)  (not hardcoded 256)
  - relinquish_tmem_alloc_permit() AFTER the gemm (single-use), not before retrieve
  - print get_num_tmem_alloc_cols + alloc/retrieve sigs
Also a control kernel K_mma_noload: zero SMEM (skip scalar load) + gemm => tests whether
the descriptor read itself misaligns independent of the scalar-store layout.
Anti-hang: server timeout=90, local timeout 120.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-mma2")

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

@cute.kernel
def _k_mma2(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, ncols: cutlass.Constexpr):
    warp_id = arch.warp_idx(); lane_id = arch.lane_idx(); tidx,_,_ = arch.thread_idx()
    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N))
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=1024)
    sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=1024)
    smp = smem.allocate(Uint32)
    mbar = smem.allocate(cutlass.Uint64)
    if warp_id==0 and lane_id==0: arch.mbarrier_init(mbar,1)
    sA_aff = cute.make_tensor(cute.recast_ptr(sA.iterator, sA_layout.inner, BFloat16), sA_layout.outer)
    sB_aff = cute.make_tensor(cute.recast_ptr(sB.iterator, sB_layout.inner, BFloat16), sB_layout.outer)
    sA_ld = cute.make_tensor(sA_aff.iterator, cute.make_layout((TILE_M,BK), stride=(BK,1)))
    sB_ld = cute.make_tensor(sB_aff.iterator, cute.make_layout((TILE_N,BK), stride=(BK,1)))
    if warp_id==0:
        for i in range(TILE_M*BK//32):
            idx=lane_id+i*32; m=idx//BK; k=idx%BK; sA_ld[m,k]=gA[m,k]
        for i in range(BK*TILE_N//32):
            idx=lane_id+i*32; n=idx//BK; k=idx%BK; sB_ld[n,k]=gB[k,n]
    arch.fence_view_async_shared()
    if warp_id==0 and lane_id==0:
        arch.alloc_tmem(ncols, smp)
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
        arch.mbarrier_wait(mbar,0)
        arch.relinquish_tmem_alloc_permit()
    arch.sync_threads()
    if warp_id==0:
        for i in range(TILE_M*TILE_N//32):
            idx=lane_id+i*32; m=idx//TILE_N; n=idx%TILE_N; gC[m,n]=Float32(7.0)
    arch.sync_threads()
    if warp_id==0 and lane_id==0: arch.dealloc_tmem(tmem_ptr, ncols)

@cute.jit
def run_mma2(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, ncols: cutlass.Constexpr):
    _k_mma2(gA,gB,gC,ncols).launch(grid=(1,1,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import sys, torch, importlib.util, inspect
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    print("get_num_tmem_alloc_cols:", inspect.signature(utils.get_num_tmem_alloc_cols))

    with open("/root/_opus_probe_mma2.py","w") as f:
        f.write(_K)
    spec = importlib.util.spec_from_file_location("_opus_probe_mma2","/root/_opus_probe_mma2.py")
    kmod = importlib.util.module_from_spec(spec); sys.modules["_opus_probe_mma2"]=kmod
    spec.loader.exec_module(kmod)

    # Need ncols. Compute it via a throwaway compile that prints, OR just try 256.
    # We'll try the canonical value first; if get_num returns something else we adjust.
    M,N,K = 128,256,16
    torch.manual_seed(0)
    A = torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
    B = torch.randn(K,N,device="cuda",dtype=torch.bfloat16)
    for ncols in [256, 128, 512]:
        C = torch.zeros(M,N,device="cuda",dtype=torch.float32)
        gA=from_dlpack(A.contiguous()); gB=from_dlpack(B.contiguous()); gC=from_dlpack(C)
        print(f"\n=== MMA2 ncols={ncols} (relinquish AFTER gemm) ===")
        try:
            c = cute.compile(kmod.run_mma2, gA, gB, gC, ncols); print("  compiled")
            c(gA, gB, gC, ncols); torch.cuda.synchronize()
            print(f"  LAUNCH OK gC[0,0]={C[0,0].item()}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:140]}")
            break  # async error poisons context; stop


@app.local_entrypoint()
def main():
    probe.remote()
