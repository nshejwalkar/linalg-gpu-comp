"""
stage0_probe.py — Stage-0 toolchain pin for the tcgen05/TMEM de-risk.

Independent Modal driver (does NOT touch modal_qr.py / modal_cute.py). Three jobs:
  1. probe_cute   : introspect the INSTALLED nvidia-cutlass-dsl tcgen05/TMEM/utils
                    API surface + signatures (confirm the names tcgen05_tmem.md §9.5
                    flagged as unverified).
  2. dump_examples: pull the Blackwell CuTeDSL GEMM example bodies from the installed
                    wheel (dense_blockscaled_gemm_persistent.py, grouped_gemm.py,
                    blackwell_helpers.py). Save first/last lines + key signatures.
  3. probe_gluon  : check whether Triton's Gluon sublanguage (tcgen05/TMEM) is present
                    in the grader-matching Triton, and dump its tcgen05 API surface.

Usage (conda modal env, inside qr_competition/):
  modal run tcgen05/stage0_probe.py --stage cute
  modal run tcgen05/stage0_probe.py --stage examples
  modal run tcgen05/stage0_probe.py --stage gluon
  modal run tcgen05/stage0_probe.py --stage all
"""

import modal

# Image with CuTe DSL (matches modal_cute.py cutlass_image) for the CuTe probes.
cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

# Clean image = grader mirror (torch cu130 + triton, NO cutlass). Triton ships with
# torch; this is where we check Gluon (the grader has exactly this).
clean_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)

app = modal.App("qr-tcgen05-stage0")


def _sig(obj, name):
    import inspect
    try:
        return f"{name}{inspect.signature(obj)}"
    except Exception as e:
        return f"{name}(<no sig: {type(e).__name__}>)"


@app.function(gpu="B200", image=cutlass_image, timeout=900)
def probe_cute():
    import importlib
    import inspect

    print("=" * 78)
    print("STAGE 0 — CuTe-DSL tcgen05 / TMEM / utils API PROBE (installed wheel)")
    print("=" * 78)

    import cutlass
    import cutlass.cute as cute
    print(f"cutlass.__version__ = {getattr(cutlass, '__version__', '?')}")
    print(f"cutlass file: {getattr(cutlass, '__file__', '?')}")

    # ---- 1. tcgen05 module ----
    print("\n" + "-" * 60)
    print("cutlass.cute.nvgpu.tcgen05 — contents")
    print("-" * 60)
    try:
        import cutlass.cute.nvgpu.tcgen05 as tg
        names = [a for a in dir(tg) if not a.startswith("_")]
        print("  attrs:", names)
        # Signatures for the load-bearing names.
        for nm in ["MmaF16BF16Op", "MmaTF32Op", "MmaFP8Op", "Ld32x32bOp",
                   "Ld16x256bOp", "Ld16x128bOp", "St32x32bOp", "make_tmem_copy",
                   "make_s2t_copy", "make_umma_smem_desc", "make_smem_layout_atom",
                   "commit", "CtaGroup", "OperandSource", "OperandMajorMode",
                   "Repetition", "Field", "SmemLayoutAtomKind",
                   "get_tmem_copy_properties", "find_tmem_tensor_col_offset",
                   "fence", "fence_view_async_tmem_load", "relinquish_alloc_permit"]:
            obj = getattr(tg, nm, None)
            if obj is not None:
                print("  ", _sig(obj, nm) if callable(obj) else f"{nm} = {obj!r}")
        # Enums: list their members.
        for enm in ["CtaGroup", "OperandSource", "OperandMajorMode", "Field",
                    "SmemLayoutAtomKind", "Repetition"]:
            obj = getattr(tg, enm, None)
            if obj is not None and isinstance(obj, type):
                try:
                    mems = [m for m in dir(obj) if not m.startswith("_")]
                    print(f"  enum {enm}: {mems}")
                except Exception:
                    pass
    except Exception as e:
        print("  FAILED importing tcgen05:", type(e).__name__, e)

    # ---- 2. cpasync (TMA) ----
    print("\n" + "-" * 60)
    print("cutlass.cute.nvgpu.cpasync — contents")
    print("-" * 60)
    try:
        import cutlass.cute.nvgpu.cpasync as cp
        print("  attrs:", [a for a in dir(cp) if not a.startswith("_")])
        for nm in ["CopyBulkTensorTileG2SOp", "CopyBulkTensorTileS2GOp",
                   "make_tiled_tma_atom", "tma_partition", "prefetch_descriptor"]:
            obj = getattr(cp, nm, None)
            if obj is not None:
                print("  ", _sig(obj, nm) if callable(obj) else f"{nm}={obj!r}")
    except Exception as e:
        print("  FAILED importing cpasync:", type(e).__name__, e)

    # ---- 3. cute.arch ----
    print("\n" + "-" * 60)
    print("cutlass.cute.arch — warp-spec / tmem primitives")
    print("-" * 60)
    try:
        import cutlass.cute.arch as arch
        names = [a for a in dir(arch) if not a.startswith("_")]
        print("  attrs:", names)
        # The TMEM alloc/dealloc + mbarrier + elect names specifically.
        for nm in ["alloc_tmem", "dealloc_tmem", "relinquish_tmem_alloc_permit",
                   "elect_one", "mbarrier_init", "mbarrier_arrive",
                   "mbarrier_arrive_and_expect_tx", "mbarrier_wait",
                   "mbarrier_try_wait", "warp_idx", "lane_idx",
                   "fence_view_async_tmem_load", "fence_view_async_tmem_store",
                   "tmem_alloc", "tmem_dealloc"]:
            obj = getattr(arch, nm, None)
            if obj is not None:
                print("  ", _sig(obj, nm) if callable(obj) else f"{nm}={obj!r}")
    except Exception as e:
        print("  FAILED importing arch:", type(e).__name__, e)

    # ---- 4. cutlass.utils (SM100 helpers) ----
    print("\n" + "-" * 60)
    print("cutlass.utils — SM100 helpers")
    print("-" * 60)
    try:
        import cutlass.utils as utils
        names = [a for a in dir(utils) if not a.startswith("_")]
        print("  attrs:", names)
        for nm in ["make_trivial_tiled_mma", "make_smem_layout_a",
                   "make_smem_layout_b", "make_smem_layout_epi",
                   "get_num_tmem_alloc_cols", "get_tmem_load_op",
                   "compute_epilogue_tile_shape", "get_smem_store_op",
                   "cluster_shape_to_tma_atom_A", "cluster_shape_to_tma_atom_B"]:
            obj = getattr(utils, nm, None)
            if obj is not None:
                print("  ", _sig(obj, nm))
    except Exception as e:
        print("  FAILED importing utils:", type(e).__name__, e)

    # ---- 4b. blackwell_helpers specifically ----
    print("\n" + "-" * 60)
    print("cutlass.utils.blackwell_helpers — signatures")
    print("-" * 60)
    for modname in ["cutlass.utils.blackwell_helpers",
                    "cutlass.utils.blackwell"]:
        try:
            bh = importlib.import_module(modname)
            print(f"  OK {modname}: {[a for a in dir(bh) if not a.startswith('_')]}")
            for nm in ["make_trivial_tiled_mma", "make_smem_layout_a",
                       "make_smem_layout_b", "get_num_tmem_alloc_cols",
                       "get_tmem_load_op", "make_smem_layout_epi"]:
                obj = getattr(bh, nm, None)
                if obj is not None:
                    print("  ", _sig(obj, nm))
        except Exception as e:
            print(f"  -- {modname}: {type(e).__name__}: {str(e)[:80]}")

    # ---- 5. pipeline classes ----
    print("\n" + "-" * 60)
    print("cutlass.pipeline — class names")
    print("-" * 60)
    try:
        from cutlass import pipeline
        names = [a for a in dir(pipeline) if not a.startswith("_")]
        print("  attrs:", names)
        for nm in ["PipelineTmaUmma", "PipelineUmmaAsync", "PipelineAsync",
                   "PipelineState", "PipelineTmaMultiConsumerAsync",
                   "make_pipeline_state"]:
            obj = getattr(pipeline, nm, None)
            if obj is not None:
                print(f"  has {nm}: {obj}")
                if isinstance(obj, type):
                    methods = [m for m in dir(obj) if not m.startswith("_")][:25]
                    print(f"     methods: {methods}")
    except Exception as e:
        print("  FAILED importing pipeline:", type(e).__name__, e)

    # ---- 6. TMEM allocator (cute.TMEM / utils) ----
    print("\n" + "-" * 60)
    print("TMEM allocator surfaces")
    print("-" * 60)
    for path in ["cutlass.cute.TMEM", "cutlass.utils.TmemAllocator",
                 "cutlass.cute.arch"]:
        try:
            mod, _, attr = path.rpartition(".")
            m = importlib.import_module(mod)
            obj = getattr(m, attr, None)
            if obj is not None:
                print(f"  {path}: {obj}")
                if isinstance(obj, type):
                    print(f"     methods: {[x for x in dir(obj) if not x.startswith('_')]}")
        except Exception as e:
            print(f"  -- {path}: {type(e).__name__}")
    # Allocator1Sm specifically (mentioned in spec §2.2)
    try:
        import cutlass.cute as cute
        for nm in ["TMEM"]:
            obj = getattr(cute, nm, None)
            print(f"  cute.{nm}: {obj}")
            if obj is not None:
                subs = [x for x in dir(obj) if not x.startswith("_")]
                print(f"     {subs}")
                alloc = getattr(obj, "Allocator1Sm", None) or getattr(obj, "Allocator", None)
                if alloc is not None:
                    print(f"     allocator: {alloc}; methods: {[x for x in dir(alloc) if not x.startswith('_')]}")
    except Exception as e:
        print("  cute.TMEM probe failed:", e)


@app.function(gpu="B200", image=cutlass_image, timeout=900)
def dump_examples():
    """Locate the installed Blackwell CuTeDSL GEMM examples + helper lib and print
    their key structure (imports, the kernel/epilogue function signatures, MMA-atom
    construction lines) so we can copy the real wiring."""
    import os
    import glob
    import cutlass

    print("=" * 78)
    print("STAGE 0 — Blackwell CuTeDSL example bodies (from installed wheel)")
    print("=" * 78)

    cutlass_dir = os.path.dirname(cutlass.__file__)
    print(f"cutlass package dir: {cutlass_dir}")
    # The examples may live under .../cutlass/.. or a sibling; search broadly.
    roots = [cutlass_dir, os.path.dirname(cutlass_dir),
             "/usr/local/lib/python3.11/site-packages"]
    targets = ["dense_blockscaled_gemm_persistent.py", "grouped_gemm.py",
               "dense_gemm_persistent.py", "blackwell_helpers.py",
               "dense_gemm.py"]
    found = {}
    for root in roots:
        for t in targets:
            for hit in glob.glob(os.path.join(root, "**", t), recursive=True):
                found.setdefault(t, hit)
    # Also just list every *.py under any 'blackwell' dir.
    print("\n-- searching for blackwell example dirs --")
    for root in roots:
        for d in glob.glob(os.path.join(root, "**", "blackwell"), recursive=True):
            if os.path.isdir(d):
                pys = [os.path.basename(p) for p in glob.glob(os.path.join(d, "*.py"))]
                print(f"  {d}: {pys}")

    print(f"\n-- found target files: {list(found.keys())} --")
    for t, path in found.items():
        print("\n" + "=" * 70)
        print(f"FILE: {t}  @ {path}")
        print("=" * 70)
        with open(path) as f:
            lines = f.readlines()
        print(f"  ({len(lines)} lines)")
        # Print imports + any line that touches the tcgen05/tma/tmem machinery.
        keys = ["import", "tcgen05", "MmaF16BF16Op", "make_trivial_tiled_mma",
                "make_tiled_tma_atom", "make_smem_layout", "TmemAllocator",
                "alloc_tmem", "get_num_tmem_alloc_cols", "PipelineTmaUmma",
                "PipelineUmmaAsync", "tcgen05.mma", "make_tmem_copy",
                "Ld32x32b", "commit(", "umma", "ScaleOut", "cta_group",
                "def ", "@cute.kernel", "@cute.jit", "elect_one",
                "relinquish", ".launch(", "warp_idx", "OperandMajorMode",
                "OperandSource", "acc_dtype", "ab_dtype", "a_dtype", "b_dtype"]
        for i, ln in enumerate(lines):
            s = ln.rstrip("\n")
            if any(k in s for k in keys):
                print(f"  {i+1:4}: {s}")


@app.function(gpu="B200", image=clean_image, timeout=900)
def probe_gluon():
    """Check whether Triton's Gluon sublanguage with tcgen05/TMEM is present in the
    grader-matching Triton (clean image = torch cu130 + triton, no cutlass)."""
    import importlib
    import inspect

    print("=" * 78)
    print("STAGE 0 — Triton Gluon tcgen05/TMEM probe (grader-mirror image)")
    print("=" * 78)
    import triton
    print(f"triton.__version__ = {triton.__version__}")
    print(f"triton file: {triton.__file__}")
    try:
        import torch
        print(f"torch {torch.__version__}; cuda cap {torch.cuda.get_device_capability()}")
    except Exception as e:
        print("torch probe:", e)

    # confirm cutlass is NOT here (true mirror)
    try:
        import cutlass  # noqa
        print("  WARNING: cutlass importable (not a clean mirror)")
    except Exception:
        print("  cutlass NOT importable (true grader mirror) — good")

    # ---- The Gluon module tree ----
    candidates = [
        "triton.experimental.gluon",
        "triton.experimental.gluon.language",
        "triton.experimental.gluon.language.nvidia",
        "triton.experimental.gluon.language.nvidia.blackwell",
        "triton.experimental.gluon.language.nvidia.hopper",
        "triton._gluon",
        "triton.gluon",
    ]
    for c in candidates:
        try:
            m = importlib.import_module(c)
            attrs = [a for a in dir(m) if not a.startswith("_")]
            print(f"\n  OK  {c}")
            print(f"      attrs: {attrs}")
        except Exception as e:
            print(f"  --  {c}: {type(e).__name__}: {str(e)[:70]}")

    # ---- tcgen05-specific symbols in blackwell gluon ----
    print("\n" + "-" * 60)
    print("Blackwell Gluon tcgen05/TMEM symbols")
    print("-" * 60)
    for modname in ["triton.experimental.gluon.language.nvidia.blackwell",
                    "triton.experimental.gluon.language.nvidia.blackwell.tma"]:
        try:
            m = importlib.import_module(modname)
            for nm in ["tcgen05_mma", "tcgen05_commit", "TensorMemoryLayout",
                       "allocate_tensor_memory", "tensor_memory_descriptor",
                       "mbarrier", "tma", "async_copy", "fence_async_shared",
                       "tcgen05_copy", "TensorMemoryScalesLayout"]:
                obj = getattr(m, nm, None)
                if obj is not None:
                    try:
                        print(f"  {modname}.{nm}: {_sig(obj, nm)}")
                    except Exception:
                        print(f"  {modname}.{nm}: present ({type(obj)})")
        except Exception as e:
            print(f"  -- {modname}: {type(e).__name__}: {str(e)[:70]}")

    # ---- Is the 06-tcgen05.py tutorial shipped? ----
    print("\n" + "-" * 60)
    print("Gluon tutorial files shipped with the wheel?")
    print("-" * 60)
    import os, glob
    troot = os.path.dirname(triton.__file__)
    for d in [troot, os.path.dirname(troot)]:
        for hit in glob.glob(os.path.join(d, "**", "*gluon*"), recursive=True)[:20]:
            print("  ", hit)
        for hit in glob.glob(os.path.join(d, "**", "06-tcgen05*"), recursive=True):
            print("  TUTORIAL:", hit)


@app.local_entrypoint()
def main(stage: str = "all"):
    if stage in ("cute", "all"):
        probe_cute.remote()
    if stage in ("examples", "all"):
        dump_examples.remote()
    if stage in ("gluon", "all"):
        probe_gluon.remote()
