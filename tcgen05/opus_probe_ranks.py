"""
opus_probe_ranks.py — build-only: try several A/B/D shapings into cute.gemm and see
which one VERIFIES (compiles). No launch => no hang. We never run, just cute.compile.

Candidates for A (and symmetric B), D:
  Recall make_smem_layout_a -> ((M,K),1,1,stage). slice stage -> sA0=((M,K),1,1) rank3.
    partition_A(sA0) -> ((M,K),16,1,1) rank4.
  make_fragment_C(partition_shape_C) -> ((M,N),1,1) rank3.
    partition_C(acc) -> ((M,N),256,1,1) rank4.

Hypotheses:
  H1: gemm wants the *partition_A* (rank4) for A and *partition_C* (rank4) for D. (already failed: invalid layout)
  H2: gemm wants sA0 (rank3 stage-sliced) for A and make_fragment_C (rank3) for D.
  H3: gemm wants partition_A (rank4) A but make_fragment_C(rank3) D — mismatched, likely no.
We test H2 here (most likely correct: matches dispatch [5] (V,M,K)x(V,N,K)=>(V,M,N) rank3).
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-ranks")

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
def _k(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    warp_id = arch.warp_idx(); lane_id = arch.lane_idx()
    tiled_mma = bh.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, tc.CtaGroup.ONE, (TILE_M, TILE_N))
    sA_layout = bh.make_smem_layout_a(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    sB_layout = bh.make_smem_layout_b(tiled_mma, (TILE_M, TILE_N, BK), BFloat16, 1)
    smem = SmemAllocator()
    sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=1024)
    sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=1024)
    smp = smem.allocate(Uint32)
    if warp_id == 0 and lane_id == 0:
        arch.alloc_tmem(TILE_N, smp); arch.relinquish_tmem_alloc_permit()
    arch.sync_threads()
    tmem_ptr = arch.retrieve_tmem_ptr(Float32, 128, smp)

    sA_aff = cute.make_tensor(cute.recast_ptr(sA.iterator, sA_layout.inner, BFloat16), sA_layout.outer)
    sB_aff = cute.make_tensor(cute.recast_ptr(sB.iterator, sB_layout.inner, BFloat16), sB_layout.outer)
    sA0 = sA_aff[None, None, None, 0]
    sB0 = sB_aff[None, None, None, 0]
    print("sA0 rank/shape:", cute.rank(sA0), sA0.shape)
    print("sB0 rank/shape:", cute.rank(sB0), sB0.shape)

    thr = tiled_mma.get_slice(0)
    # D candidate: make_fragment_C on partition_shape_C, then point at TMEM
    psC = tiled_mma.partition_shape_C((TILE_M, TILE_N))
    accF = thr.make_fragment_C(psC)   # ((M,N),1,1) at tmem placeholder ptr
    print("accF rank/shape:", cute.rank(accF), accF.shape)
    # rebuild accF at the real tmem_ptr with the same layout
    acc = cute.make_tensor(tmem_ptr, accF.layout)
    print("acc rank/shape:", cute.rank(acc), acc.shape)

    # H2: pass sA0 (rank3), sB0 (rank3), acc (rank3) directly
    tiled_mma.set(tc.Field.ACCUMULATE, False)
    cute.gemm(tiled_mma, acc, sA0, sB0, acc)
    print("H2 cute.gemm VERIFIED")

@cute.jit
def run(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    _k(gA, gB, gC).launch(grid=(1,1,1), block=(128,1,1))
'''


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import sys, torch, traceback, importlib.util
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    kpath = "/root/_opus_probe_ranks.py"
    with open(kpath, "w") as f:
        f.write(_PROBE)
    spec = importlib.util.spec_from_file_location("_opus_probe_ranks", kpath)
    kmod = importlib.util.module_from_spec(spec); sys.modules["_opus_probe_ranks"] = kmod
    try:
        spec.loader.exec_module(kmod); print("module loaded")
    except Exception as e:
        print("LOAD FAILED:", e); traceback.print_exc(); return
    M, N, K = 128, 256, 16
    A = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.zeros(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    gA = from_dlpack(A.contiguous()); gB = from_dlpack(B.contiguous()); gC = from_dlpack(C)
    print("Compiling (build-only, NO launch)...")
    try:
        cute.compile(kmod.run, gA, gB, gC); print(">>> COMPILE OK — H2 layout is valid")
    except Exception as e:
        print("COMPILE FAILED:", type(e).__name__, e); traceback.print_exc()


@app.local_entrypoint()
def main():
    probe.remote()
