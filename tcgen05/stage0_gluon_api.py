"""
stage0_gluon_api.py — deep dive into the Gluon (Triton) tcgen05/TMEM API so we can
write the Stage-1 BF16x9 GEMM. Grader-mirror image (torch cu130 + triton, no cutlass).

Dumps: the blackwell tma/mbarrier/async_copy submodule signatures, TensorMemoryLayout,
allocate_tensor_memory, tcgen05_mma, the gluon jit entry, and whether a Gluon kernel
actually COMPILES + RUNS on B200 (a tiny smoke MMA). This is the make-or-break for the
"ship a Gluon kernel with no cubin-embed" thesis.

Usage:
  modal run tcgen05/stage0_gluon_api.py --stage api
  modal run tcgen05/stage0_gluon_api.py --stage smoke
"""

import modal

clean_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)

app = modal.App("qr-tcgen05-gluon-api")


def _sig(obj, name):
    import inspect
    try:
        return f"{name}{inspect.signature(obj)}"
    except Exception as e:
        return f"{name}(<no sig: {type(e).__name__}: {e}>)"


@app.function(gpu="B200", image=clean_image, timeout=900)
def api():
    import importlib
    print("=" * 78)
    print("GLUON BLACKWELL API — deep signatures")
    print("=" * 78)
    import triton
    print("triton", triton.__version__)

    # tma / mbarrier / async_copy submodules
    for modname in [
        "triton.experimental.gluon.language.nvidia.blackwell.tma",
        "triton.experimental.gluon.language.nvidia.blackwell.mbarrier",
        "triton.experimental.gluon.language.nvidia.blackwell.async_copy",
    ]:
        try:
            m = importlib.import_module(modname)
            print(f"\n-- {modname} --")
            for nm in [a for a in dir(m) if not a.startswith("_")]:
                obj = getattr(m, nm)
                if callable(obj):
                    print("  ", _sig(obj, nm))
                else:
                    print(f"   {nm} = {obj!r}"[:120])
        except Exception as e:
            print(f"  -- {modname}: {type(e).__name__}: {e}")

    # The blackwell __init__ key functions
    import triton.experimental.gluon.language.nvidia.blackwell as bw
    print("\n-- blackwell core fns --")
    for nm in ["tcgen05_mma", "tcgen05_commit", "tcgen05_copy",
               "allocate_tensor_memory", "TensorMemoryLayout",
               "fence_async_shared", "get_tmem_reg_layout",
               "tcgen05_mma_scaled", "tensor_memory_descriptor"]:
        obj = getattr(bw, nm, None)
        if obj is not None:
            print("  ", _sig(obj, nm))

    # Gluon top-level: jit, warp_specialize
    import triton.experimental.gluon as gluon
    print("\n-- gluon top-level --")
    for nm in ["jit", "must_use_result", "constexpr_function"]:
        obj = getattr(gluon, nm, None)
        if obj is not None:
            print("  ", _sig(obj, nm))
    import triton.experimental.gluon.language as gl
    print("\n-- gl key fns --")
    for nm in ["allocate_shared_memory", "shared_memory_descriptor", "warp_specialize",
               "NVMMASharedLayout", "BlockedLayout", "convert_layout",
               "set_auto_layout", "thread_idx" ]:
        obj = getattr(gl, nm, None)
        if obj is not None:
            print("  ", _sig(obj, nm) if callable(obj) else f"{nm}={obj}")

    # NVMMASharedLayout / TensorMemoryLayout constructors
    print("\n-- layout constructors --")
    for nm, mod in [("NVMMASharedLayout", gl), ("TensorMemoryLayout", bw)]:
        obj = getattr(mod, nm, None)
        if obj is not None:
            print("  ", _sig(obj, nm))
            for meth in ["__init__"]:
                mm = getattr(obj, meth, None)
                if mm:
                    print("    ", _sig(mm, f"{nm}.{meth}"))


@app.function(gpu="B200", image=clean_image, timeout=900)
def smoke():
    """Smoke-test: does a minimal Gluon tcgen05 MMA kernel COMPILE + RUN on B200?
    Single tile BF16 x BF16 -> FP32 accumulator -> store. Validates the whole
    no-cubin-embed thesis end to end."""
    import torch
    import triton
    import triton.language as tl
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia import blackwell as bw
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma, tcgen05_commit,
        mbarrier, tma, fence_async_shared,
    )

    print("=" * 78)
    print("GLUON tcgen05 SMOKE — compile+run a 128x128x64 BF16->FP32 MMA on B200")
    print("=" * 78)

    M, N, K = 128, 128, 64

    @gluon.jit
    def mma_smoke(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        # shared layouts for A (MxK) and B (KxN), bf16, 128B swizzle.
        a_sh_layout: tl.constexpr = gl.NVMMASharedLayout.get_default_for(
            [M, K], gl.bfloat16)
        b_sh_layout: tl.constexpr = gl.NVMMASharedLayout.get_default_for(
            [K, N], gl.bfloat16)
        smem_a = gl.allocate_shared_memory(gl.bfloat16, [M, K], a_sh_layout)
        smem_b = gl.allocate_shared_memory(gl.bfloat16, [K, N], b_sh_layout)

        # blocked layout to load A,B from gmem
        blk: tl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        offs_am = gl.arange(0, M, layout=gl.SliceLayout(1, blk))[:, None]
        offs_ak = gl.arange(0, K, layout=gl.SliceLayout(0, blk))[None, :]
        a = gl.load(a_ptr + offs_am * K + offs_ak)
        smem_a.store(a)
        offs_bk = gl.arange(0, K, layout=gl.SliceLayout(1, blk))[:, None]
        offs_bn = gl.arange(0, N, layout=gl.SliceLayout(0, blk))[None, :]
        b = gl.load(b_ptr + offs_bk * N + offs_bn)
        smem_b.store(b)
        fence_async_shared()

        # TMEM accumulator
        tmem_layout: tl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [M, N], tmem_layout)

        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)

        tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, mbarriers=[bar])
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)

        # read TMEM -> registers -> store
        res = acc_tmem.load(gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0]))
        offs_cm = gl.arange(0, M, layout=gl.SliceLayout(1, blk))[:, None]
        offs_cn = gl.arange(0, N, layout=gl.SliceLayout(0, blk))[None, :]
        gl.store(c_ptr + offs_cm * N + offs_cn, res)

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    try:
        mma_smoke[(1,)](a, b, c, M, N, K)
        torch.cuda.synchronize()
        ref = (a.float() @ b.float())
        err = (c - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"  COMPILED + RAN. max abs err={err:.4f} rel={rel:.2e}")
        print(f"  c[0,:4]={c[0,:4].tolist()}  ref[0,:4]={ref[0,:4].tolist()}")
        print("  >>> GLUON tcgen05 PATH IS LIVE (no cubin embed)")
    except Exception as e:
        import traceback
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


@app.local_entrypoint()
def main(stage: str = "api"):
    if stage in ("api", "all"):
        api.remote()
    if stage in ("smoke", "all"):
        smoke.remote()
