"""
stage1_smoke.py — minimal Gluon tcgen05 MMA that COMPILES + RUNS on B200, using the
EXACT idiom mined from triton's tl_dot_scaled_blackwell. One 128x256x64 BF16->FP32
tile, plain gl.load -> SMEM -> tcgen05_mma -> TMEM -> load back -> store.

This is the foundation; once green we add the BF16x9 split + K-loop + batching.
"""
import modal

clean_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("qr-tcgen05-stage1-smoke")


@app.function(gpu="B200", image=clean_image, timeout=900)
def smoke():
    import torch
    import triton
    import triton.language as tl
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma, tcgen05_commit,
        mbarrier, fence_async_shared, get_tmem_reg_layout,
    )

    print("=" * 78)
    print("STAGE 1 SMOKE — Gluon tcgen05 128x256x64 BF16->FP32 (single tile)")
    print("=" * 78)

    M: int = 128
    N: int = 256
    K: int = 64

    @gluon.jit
    def mma_kernel(a_ptr, b_ptr, c_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        # ---- shared-memory operand layouts (NVMMA 128B swizzle, bf16=16-bit) ----
        a_sh: tl.constexpr = gl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2)
        b_sh: tl.constexpr = gl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2)
        smem_a = gl.allocate_shared_memory(gl.bfloat16, [M, K], a_sh)
        smem_b = gl.allocate_shared_memory(gl.bfloat16, [K, N], b_sh)

        # ---- load A (MxK) and B (KxN) from gmem with a blocked reg layout ----
        blk_a: tl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0])
        am = gl.arange(0, M, layout=gl.SliceLayout(1, blk_a))[:, None]
        ak = gl.arange(0, K, layout=gl.SliceLayout(0, blk_a))[None, :]
        a = gl.load(a_ptr + am * K + ak)
        smem_a.store(a)

        blk_b: tl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0])
        bk = gl.arange(0, K, layout=gl.SliceLayout(1, blk_b))[:, None]
        bn = gl.arange(0, N, layout=gl.SliceLayout(0, blk_b))[None, :]
        b = gl.load(b_ptr + bk * N + bn)
        smem_b.store(b)
        fence_async_shared()

        # ---- TMEM accumulator ----
        col_stride: tl.constexpr = 32 // gl.float32.primitive_bitwidth  # =1
        acc_layout: tl.constexpr = TensorMemoryLayout((M, N), col_stride=col_stride)
        reg_layout: tl.constexpr = get_tmem_reg_layout(
            gl.float32, (M, N), acc_layout, gl.num_warps())
        acc0 = gl.zeros([M, N], gl.float32, layout=reg_layout)
        acc_tmem = allocate_tensor_memory(gl.float32, [M, N], acc_layout, acc0)
        fence_async_shared()

        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)

        # use_acc=True -> add into acc0 (which is zero); commit signals the barrier
        tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=True, mbarriers=[bar])
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

        out = acc_tmem.load(reg_layout)
        ret: tl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0])
        out = gl.convert_layout(out, ret)
        cm = gl.arange(0, M, layout=gl.SliceLayout(1, ret))[:, None]
        cn = gl.arange(0, N, layout=gl.SliceLayout(0, ret))[None, :]
        gl.store(c_ptr + cm * N + cn, out)

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    try:
        mma_kernel[(1,)](a, b, c, M, N, K, num_warps=4)
        torch.cuda.synchronize()
        ref = a.float() @ b.float()
        err = (c - ref).abs().max().item()
        denom = ref.abs().max().item()
        print(f"  COMPILED + RAN. max abs err={err:.5f} rel={err/denom:.2e}")
        print(f"  c[0,:4]   = {[round(x,3) for x in c[0,:4].tolist()]}")
        print(f"  ref[0,:4] = {[round(x,3) for x in ref[0,:4].tolist()]}")
        ok = err / denom < 1e-2   # bf16 single-MMA: ~1e-2 expected (8-bit mantissa)
        print(f"  >>> {'GLUON tcgen05 LIVE' if ok else 'MISMATCH'} (no cubin embed)")
    except Exception as e:
        import traceback
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    smoke.remote()
