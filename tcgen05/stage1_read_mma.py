"""Read tcgen05/mma.py and algorithm.py to understand cute.gemm and MMA call pattern."""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-read-mma")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def read_mma():
    import os, glob

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    # Read tcgen05/mma.py - the key file
    mma_path = os.path.join(base, "cute/nvgpu/tcgen05/mma.py")
    print(f"\n=== {mma_path} ===")
    if os.path.exists(mma_path):
        with open(mma_path) as f:
            content = f.read()
        # Print all (it's important)
        print(content[:8000])  # first 8KB should cover the key parts
    else:
        print("NOT FOUND")

    # Read algorithm.py where cute.gemm lives
    algo_path = os.path.join(base, "cute/algorithm.py")
    print(f"\n=== {algo_path} ===")
    if os.path.exists(algo_path):
        with open(algo_path) as f:
            content = f.read()
        print(content[:4000])
    else:
        print("NOT FOUND")


@app.local_entrypoint()
def main():
    read_mma.remote()
