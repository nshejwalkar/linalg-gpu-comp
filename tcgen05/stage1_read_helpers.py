"""Read key helper files from the installed wheel to understand API patterns."""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-stage1-read-helpers")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def read_helpers():
    import os
    import glob

    files_to_read = [
        "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/blackwell_helpers.py",
        "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/arch/tmem.py",
        "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/smem_allocator.py",
        "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/tmem_allocator.py",
    ]

    # Also find any GEMM example
    gemm_dir = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/gemm"
    if os.path.isdir(gemm_dir):
        for f in sorted(os.listdir(gemm_dir)):
            print(f"  gemm dir: {f}")
        # Read first gemm file
        for f in sorted(os.listdir(gemm_dir)):
            if f.endswith(".py") and not f.startswith("_"):
                files_to_read.insert(0, os.path.join(gemm_dir, f))
                break

    for path in files_to_read:
        if not os.path.exists(path):
            print(f"\n!!! NOT FOUND: {path}")
            continue
        print(f"\n{'='*70}")
        print(f"FILE: {path}")
        print(f"{'='*70}")
        with open(path) as f:
            lines = f.readlines()
        print(f"  ({len(lines)} lines)")
        # Print all lines (if short) or key lines
        if len(lines) <= 200:
            for i, ln in enumerate(lines):
                print(f"  {i+1:4}: {ln}", end="")
        else:
            # Print first 80 + lines with key keywords
            for i, ln in enumerate(lines[:80]):
                print(f"  {i+1:4}: {ln}", end="")
            print("  ...")
            keywords = ["alloc_tmem", "SmemAllocator", "TmemAllocator", "alloc_smem",
                        "alloc(", "allocate(", "allocate_tensor", "make_tensor",
                        "make_ptr", "AddressSpace", "smem", "mbarrier", "elect_one",
                        "make_umma", "MmaF16BF16", "commit(", "Field", "CtaGroup",
                        "Ld32x32b", "fence_view", "tcgen05", "TiledMma", "mma_atom",
                        "def make_trivial", "def get_num_tmem", "warp_idx", "lane_idx"]
            for i, ln in enumerate(lines[80:], start=80):
                if any(k in ln for k in keywords):
                    print(f"  {i+1:4}: {ln}", end="")

    # Also look for actual GEMM example code
    print("\n\n=== SEARCHING FOR GEMM EXAMPLE FILES ===")
    root = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl"
    for pattern in ["**/*gemm*.py", "**/*blackwell*.py", "**/*example*.py"]:
        for hit in glob.glob(os.path.join(root, pattern), recursive=True)[:5]:
            if "__pycache__" not in hit:
                print(f"  {hit}")


@app.local_entrypoint()
def main():
    read_helpers.remote()
