"""
opus_probe_tile.py — isolate the multi-block crash. Single shape M=512,N=256,K=64,
grid=(4,1), NPASS=36. Use cute.local_tile for clean per-block GMEM tiling (instead
of manual iterator+offset). Correctness check vs FP64. retries=0, timeout=90, local 120.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-tile")

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

TILE_M=128; TILE_N=256; BK=16; NPASS=36; THREADS=128

@cute.struct
class SharedStorage:
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32

@cute.kernel
def _k(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tidx,_,_ = arch.thread_idx()
    bidx,bidy,_ = arch.block_idx()
    warp_id = arch.warp_idx(); warp_id = arch.make_warp_uniform(warp_id)
    lane_id = arch.lane_idx()

    # Per-block GMEM tiles via local_tile.
    # mA=(M, 9K): tile (TILE_M, 9K) at (bidx, 0) -> this block's A rows.
    gA = cute.local_tile(mA, (TILE_M, NPASS*BK), (bidx, 0))         # (128, 9K)
    gB = cute.local_tile(mB, (NPASS*BK, TILE_N), (0, bidy))         # (9K, 256)
    gC = cute.local_tile(mC, (TILE_M, TILE_N), (bidx, bidy))        # (128, 256)

    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N))
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    smem = SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(element_type=BFloat16, layout=sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
    sB = smem.allocate_tensor(element_type=BFloat16, layout=sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    accP, accC = pipeline.PipelineUmmaAsync.create(
        num_stages=1, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar.data_ptr()).make_participants()
    tCrA = tiled_mma.make_fragment_A(sA); tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C((TILE_M,TILE_N))
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tmem.wait_for_alloc(); tmem_ptr = tmem.retrieve_ptr(Float32)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)
    sA_c = sA[None,None,None,0]; sB_c = sB[None,None,None,0]
    gC_part = cute.make_tensor(gC.iterator, cute.make_layout(((TILE_M,TILE_N),1,1), stride=((cute.size(mC,mode=[1]),1),0,0)))

    if warp_id==0:
        acc_empty = accP.acquire_and_advance()
        for p in cutlass.range(NPASS, unroll=1):
            k0 = p*BK
            for i in cutlass.range(TILE_M*BK//32, unroll=1):
                idx=lane_id+i*32; m=idx//BK; k=idx%BK; sA_c[(m,k),0,0]=gA[m,k0+k]
            for i in cutlass.range(BK*TILE_N//32, unroll=1):
                idx=lane_id+i*32; n=idx//BK; k=idx%BK; sB_c[(n,k),0,0]=gB[k0+k,n]
            arch.fence_view_async_shared(); arch.sync_warp()
            tiled_mma.set(tc.Field.ACCUMULATE, p!=0)
            cute.gemm(tiled_mma, tCtAcc, tCrA[(None,None,0,0)], tCrB[(None,None,0,0)], tCtAcc)
        acc_empty.commit()
    tmem.relinquish_alloc_permit()
    acc_full = accC.wait_and_advance()
    SUB=4
    epi=((cute.size(tCtAcc,mode=[0,0]), cute.size(tCtAcc,mode=[0,1])//SUB),)
    tEpi=cute.zipped_divide(tCtAcc, epi); gEpi=cute.zipped_divide(gC_part, epi)
    cp=cute.make_copy_atom(tc.Ld32x32bOp(tc.Repetition.x64, tc.Pack.NONE), Float32)
    tcp=tc.make_tmem_copy(cp, tEpi[None,0]); thr=tcp.get_slice(tidx)
    tDtC=thr.partition_S(tEpi); tDgC=thr.partition_D(gEpi)
    rAcc=cute.make_rmem_tensor(tDgC[None,None,0].shape, Float32)
    for i in cutlass.range(cute.size(tDtC,mode=[2]), unroll=1):
        cute.copy(tcp, tDtC[None,None,i], rAcc); arch.fence_view_async_tmem_load()
        cute.autovec_copy(rAcc, tDgC[None,None,i])
    acc_full.release(); pipeline.sync(barrier_id=1); tmem.free(tmem_ptr)

@cute.jit
def run(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    _k(mA,mB,mC).launch(grid=(4,2,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90, retries=0)
def probe():
    import sys, torch, traceback, importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    with open("/root/_opus_probe_tile.py","w") as f: f.write(_K)
    spec = importlib.util.spec_from_file_location("_opus_probe_tile","/root/_opus_probe_tile.py")
    km = importlib.util.module_from_spec(spec); sys.modules["_opus_probe_tile"]=km
    spec.loader.exec_module(km)
    M,N,K = 512,512,64
    torch.manual_seed(7)
    A = torch.randn(M,K,device="cuda",dtype=torch.float32)
    B = torch.randn(K,N,device="cuda",dtype=torch.float32)
    def split(x):
        x0=x.bfloat16(); r1=x-x0.float(); x1=r1.bfloat16(); r2=r1-x1.float(); x2=r2.bfloat16()
        return x0,x1,x2
    a0,a1,a2=split(A); b0,b1,b2=split(B)
    pairs=[(a0,b0),(a0,b1),(a1,b0),(a0,b2),(a2,b0),(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
    Ap=torch.cat([p[0] for p in pairs],dim=1).contiguous()
    Bp=torch.cat([p[1] for p in pairs],dim=0).contiguous()
    C=torch.zeros(M,N,device="cuda",dtype=torch.float32)
    gA=from_dlpack(Ap); gB=from_dlpack(Bp); gC=from_dlpack(C)
    print("=== tiled M=512 N=256 K=64 grid(4,1) ===")
    try:
        c=cute.compile(km.run, gA, gB, gC); print("  compiled")
        c(gA, gB, gC); torch.cuda.synchronize()
        ref=(A.double()@B.double())
        rel=((C.double()-ref).abs().max()/ref.abs().max()).item()
        print(f"  LAUNCH OK rel-vs-FP64={rel:.3e} {'PASS' if rel<1e-5 else 'FAIL'}")
        print(f"  C[200,:4]={[round(v,3) for v in C[200,:4].tolist()]}")
        print(f"  ref[200,:4]={[round(v,3) for v in ref[200,:4].tolist()]}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}"); traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
