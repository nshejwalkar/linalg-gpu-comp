"""Find any Gluon tcgen05 usage shipped inside the installed triton wheel (tests,
tutorials, internal kernels) so we can copy the exact idiom. Grader-mirror image."""
import modal

clean_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("qr-tcgen05-find-gluon")


@app.function(gpu="B200", image=clean_image, timeout=900)
def find():
    import os, glob, triton
    troot = os.path.dirname(triton.__file__)
    print("triton root:", troot)
    # 1. Any python file mentioning tcgen05_mma or allocate_tensor_memory
    print("\n-- files using tcgen05_mma / allocate_tensor_memory --")
    hits = []
    for py in glob.glob(os.path.join(troot, "**", "*.py"), recursive=True):
        try:
            with open(py) as f:
                txt = f.read()
        except Exception:
            continue
        if "tcgen05_mma" in txt or "allocate_tensor_memory" in txt:
            hits.append(py)
    for h in hits:
        print("  ", h)

    # 2. Show how the lowering/test helper uses TensorMemoryLayout + NVMMASharedLayout.
    #    Print any .load(...) usage on a tmem descriptor and the mma call site.
    for h in hits:
        base = os.path.basename(h)
        if base in ("blackwell.py", "_layouts.py", "__init__.py"):
            continue
        print("\n" + "=" * 70)
        print("USAGE FILE:", h)
        print("=" * 70)
        with open(h) as f:
            lines = f.readlines()
        keys = ["tcgen05_mma", "tcgen05_commit", "allocate_tensor_memory",
                "TensorMemoryLayout", "NVMMASharedLayout", "BlockedLayout",
                ".load(", ".store(", "mbarrier", "make_tensor_descriptor",
                "async_copy_global", "fence_async_shared", "warp_specialize",
                "def ", "@gluon.jit", "@gl.", "allocate_shared", "get_tmem_reg_layout",
                "tma."]
        for i, ln in enumerate(lines):
            s = ln.rstrip("\n")
            if any(k in s for k in keys):
                print(f"  {i+1:4}: {s}")

    # 3. The blackwell __init__ source for allocate_tensor_memory + the descriptor
    #    .load/.store methods (so we know the reg-layout arg).
    import triton.experimental.gluon.language.nvidia.blackwell as bw
    print("\n" + "=" * 70)
    print("blackwell module source (key defs)")
    print("=" * 70)
    bwfile = bw.__file__
    with open(bwfile) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        s = ln.rstrip("\n")
        if any(k in s for k in ["def load", "def store", "def slice", "class tensor_memory",
                                "def index", "reg_layout", "def alloc", "def _reinterpret",
                                "class TensorMemoryLayout"]):
            print(f"  {i+1:4}: {s}")


@app.local_entrypoint()
def main():
    find.remote()
