"""
stage1_type_probe4.py — read full helpers.py, find what make_umma_smem_desc returns.
Also: try calling make_umma_smem_desc with proper smem pointer INSIDE kernel trace.
The crucial test: does it return cute.Tensor?

Also: look at the FULL mma.py for make_fragment_A, partition_A for smem-ss atoms.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe4")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe4():
    import inspect
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tc
    from cutlass.cutlass_dsl import BFloat16, Float32, Uint32

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    print("=" * 72)
    print("TYPE PROBE 4 — full helpers.py + mma.py partial")
    print("=" * 72)

    # 1. Print ALL of helpers.py
    print("\n=== helpers.py COMPLETE ===")
    with open(base + "/cute/nvgpu/tcgen05/helpers.py") as f:
        helpers_src = f.read()
    print(helpers_src)

    # 2. Print first 200 lines of mma.py (where make_mma_atom / TiledMma / make_fragment_A might be)
    print("\n=== mma.py lines 1-200 ===")
    with open(base + "/cute/nvgpu/tcgen05/mma.py") as f:
        mma_src = f.read()
    mma_lines = mma_src.split('\n')
    for i, line in enumerate(mma_lines[:200]):
        print(f"  {i+1}: {line}")

    # 3. Search mma.py for 'fragment' or 'partition' or 'desc'
    print("\n=== mma.py lines with 'fragment' or 'partition' or 'desc' ===")
    for i, line in enumerate(mma_lines):
        if 'fragment' in line.lower() or 'partition' in line.lower() or 'desc' in line.lower():
            print(f"  {i+1}: {line}")


@app.local_entrypoint()
def main():
    type_probe4.remote()
