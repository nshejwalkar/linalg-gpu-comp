"""Quick probe: find smem allocation + MMA atom API in the installed CuTe DSL."""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-stage1-api-probe")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.arch as arch
    import cutlass.cute.nvgpu.tcgen05 as tc
    import cutlass.utils as utils
    import inspect

    def _sig(obj, name):
        try:
            return f"{name}{inspect.signature(obj)}"
        except Exception as e:
            return f"{name}: <no sig: {e}>"

    print("=" * 72)
    print("API PROBE — smem alloc + MMA atom surface")
    print("=" * 72)

    # cute module attrs — looking for smem/alloc related things
    cute_attrs = [a for a in dir(cute) if not a.startswith("_")]
    print("\ncute attrs:", cute_attrs)

    # Look for smem-related
    smem_related = [a for a in cute_attrs if "smem" in a.lower() or "alloc" in a.lower() or "shared" in a.lower() or "make_" in a.lower()]
    print("\nsmem/alloc/make_ attrs in cute:", smem_related)
    for nm in smem_related:
        obj = getattr(cute, nm, None)
        if obj is not None and callable(obj):
            print(f"  {_sig(obj, nm)}")
        elif obj is not None:
            print(f"  {nm} = {repr(obj)[:80]}")

    # SmemAllocator
    print("\n-- SmemAllocator --")
    for path in ["cutlass.utils.SmemAllocator", "cutlass.cute.SmemAllocator",
                 "cutlass.utils.TmemAllocator", "cutlass.cute.arch.SmemAllocator"]:
        mod_name, _, attr = path.rpartition(".")
        try:
            import importlib
            m = importlib.import_module(mod_name)
            obj = getattr(m, attr, None)
            if obj is not None:
                print(f"  {path}: {obj}")
                if hasattr(obj, "__init__"):
                    print(f"    __init__{inspect.signature(obj.__init__)}")
                methods = [x for x in dir(obj) if not x.startswith("_")]
                print(f"    methods: {methods}")
        except Exception as e:
            print(f"  {path}: {type(e).__name__}: {e}")

    # arch module
    print("\n-- arch module attrs --")
    arch_attrs = [a for a in dir(arch) if not a.startswith("_")]
    print("  ", arch_attrs)
    for nm in ["alloc_tmem", "dealloc_tmem", "relinquish_tmem_alloc_permit",
               "elect_one", "mbarrier_init", "mbarrier_arrive", "mbarrier_wait",
               "mbarrier_arrive_and_expect_tx", "mbarrier_try_wait",
               "fence_view_async_tmem_load", "fence_view_async_tmem_store",
               "fence_view_async_shared", "fence_proxy", "sync_threads",
               "sync_warp", "thread_idx", "block_idx", "warp_idx", "lane_idx"]:
        obj = getattr(arch, nm, None)
        if obj is not None:
            print(f"  {_sig(obj, nm) if callable(obj) else nm + '=' + repr(obj)}")

    # tc module - MmaF16BF16Op details
    print("\n-- tcgen05 MmaF16BF16Op --")
    op = getattr(tc, "MmaF16BF16Op", None)
    if op:
        print(f"  {_sig(op, 'MmaF16BF16Op')}")
        if hasattr(op, "__call__"):
            try:
                print(f"  __call__{inspect.signature(op.__call__)}")
            except Exception as e:
                print(f"  __call__ sig unavail: {e}")
    else:
        print("  MmaF16BF16Op: NOT FOUND")

    # make_umma_smem_desc
    obj = getattr(tc, "make_umma_smem_desc", None)
    if obj:
        print(f"\n  {_sig(obj, 'make_umma_smem_desc')}")

    # commit
    obj = getattr(tc, "commit", None)
    if obj:
        print(f"  {_sig(obj, 'commit')}")

    # Ld32x32bOp
    obj = getattr(tc, "Ld32x32bOp", None)
    if obj:
        print(f"  {_sig(obj, 'Ld32x32bOp')}")

    # Field enum members
    field_e = getattr(tc, "Field", None)
    if field_e:
        print(f"\n  Field members: {[x for x in dir(field_e) if not x.startswith('_')]}")

    # Repetition enum members
    rep_e = getattr(tc, "Repetition", None)
    if rep_e:
        print(f"  Repetition members: {[x for x in dir(rep_e) if not x.startswith('_')]}")

    # Pack enum members
    pack_e = getattr(tc, "Pack", None)
    if pack_e:
        print(f"  Pack members: {[x for x in dir(pack_e) if not x.startswith('_')]}")

    # utils helpers
    print("\n-- cutlass.utils make_trivial_tiled_mma --")
    obj = getattr(utils, "make_trivial_tiled_mma", None)
    if obj:
        print(f"  {_sig(obj, 'make_trivial_tiled_mma')}")
    obj = getattr(utils, "SmemAllocator", None)
    if obj:
        print(f"  SmemAllocator: {obj}")
        print(f"    methods: {[x for x in dir(obj) if not x.startswith('_')]}")
        try:
            print(f"    __init__{inspect.signature(obj)}")
        except: pass

    # cute.make_tensor + cute.make_layout + cute.make_smem_ptr
    print("\n-- cute.make_tensor / make_layout / make_smem_ptr --")
    for nm in ["make_tensor", "make_layout", "make_smem_ptr", "make_gmem_ptr",
               "smem_ptr", "Tensor", "Layout", "Int32", "bfloat16", "float32",
               "uint32", "uint64", "size", "rank", "depth"]:
        obj = getattr(cute, nm, None)
        if obj is not None:
            info = _sig(obj, nm) if callable(obj) else f"{nm} = {repr(obj)[:60]}"
            print(f"  {info}")

    # Look for example files shipped with the wheel
    print("\n-- Example files in installed wheel --")
    import os, glob
    cutlass_dir = os.path.dirname(cutlass.__file__)
    print(f"  cutlass dir: {cutlass_dir}")
    for pattern in ["**/*blackwell*", "**/*gemm*", "**/*tmem*", "**/*tcgen05*"]:
        hits = glob.glob(os.path.join(cutlass_dir, pattern), recursive=True)
        for h in hits[:5]:
            print(f"  {h}")

    # Check if there's a simple_gemm or dense_gemm example we can look at.
    for root in [cutlass_dir, os.path.dirname(cutlass_dir)]:
        for pattern in ["**/dense_gemm*.py", "**/simple_gemm*.py", "**/blackwell/*.py"]:
            hits = glob.glob(os.path.join(root, pattern), recursive=True)
            for h in hits[:3]:
                print(f"  FOUND: {h}")


@app.local_entrypoint()
def main():
    probe.remote()
