"""
opus_probe_tmemalloc.py — use the PROPER utils.TmemAllocator (NamedBarrier +
wait_for_alloc + retrieve_ptr) + 512 cols, with SCALAR SMEM loads + MMA + dummy
gC write (no tmem read). If this launches clean -> raw alloc_tmem was the bug and
scalar SMEM loads are fine. If still misaligns -> need TMA for SMEM.
retries=0 to avoid the 8x storm. Anti-hang: server timeout=90, local timeout 120.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-tmemalloc")

_K = r'''
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

TILE_M = 128; TILE_N = 256; BK = 16
THREADS = 128

@cute.struct
class SharedStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32

@cute.kernel
def _k(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx,_,_ = arch.thread_idx()
    warp_id = arch.warp_idx(); warp_id = arch.make_warp_uniform(warp_id)
    lane_id = arch.lane_idx()
    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N))
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    smem = SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(element_type=BFloat16, layout=sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
    sB = smem.allocate_tensor(element_type=BFloat16, layout=sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)
    mbar = storage.mbar.data_ptr()
    if warp_id==0 and lane_id==0: arch.mbarrier_init(mbar,1)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    # scalar SMEM loads via stage-sliced composed tensor
    sA_c = sA[None, None, None, 0]; sB_c = sB[None, None, None, 0]
    if warp_id==0:
        for i in range(TILE_M*BK//32):
            idx=lane_id+i*32; m=idx//BK; k=idx%BK; sA_c[(m,k),0,0]=gA[m,k]
        for i in range(BK*TILE_N//32):
            idx=lane_id+i*32; n=idx//BK; k=idx%BK; sB_c[(n,k),0,0]=gB[k,n]
    arch.fence_view_async_shared()

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C((TILE_M,TILE_N))
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    if warp_id==0:
        tiled_mma.set(tc.Field.ACCUMULATE, False)
        cute.gemm(tiled_mma, tCtAcc, tCrA[(None,None,0,0)], tCrB[(None,None,0,0)], tCtAcc)
        tc.commit(mbar)
    if warp_id==0 and lane_id==0: arch.mbarrier_wait(mbar,0)
    tmem.relinquish_alloc_permit()
    pipeline.sync(barrier_id=1)

    if warp_id==0:
        for i in range(TILE_M*TILE_N//32):
            idx=lane_id+i*32; m=idx//TILE_N; n=idx%TILE_N; gC[m,n]=Float32(9.0)
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)

@cute.jit
def run(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _k(gA,gB,gC).launch(grid=(1,1,1), block=(THREADS,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90, retries=0)
def probe():
    import sys, torch, importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    with open("/root/_opus_probe_tmemalloc.py","w") as f:
        f.write(_K)
    spec = importlib.util.spec_from_file_location("_opus_probe_tmemalloc","/root/_opus_probe_tmemalloc.py")
    kmod = importlib.util.module_from_spec(spec); sys.modules["_opus_probe_tmemalloc"]=kmod
    spec.loader.exec_module(kmod)
    M,N,K = 128,256,16
    torch.manual_seed(0)
    A = torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
    B = torch.randn(K,N,device="cuda",dtype=torch.bfloat16)
    C = torch.zeros(M,N,device="cuda",dtype=torch.float32)
    gA=from_dlpack(A.contiguous()); gB=from_dlpack(B.contiguous()); gC=from_dlpack(C)
    print("=== TmemAllocator + scalar loads + MMA ===")
    try:
        c = cute.compile(kmod.run, gA, gB, gC); print("  compiled")
        c(gA, gB, gC); torch.cuda.synchronize()
        print(f"  LAUNCH OK gC[0,0]={C[0,0].item()}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:160]}")


@app.local_entrypoint()
def main():
    probe.remote()
