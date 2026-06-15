"""
opus_probe_example.py — dump shipped CUTLASS Blackwell dense GEMM example +
make_trivial_tiled_mma + tile_to_mma_shape source from the installed wheel.

Goal: get the EXACT known-good recipe for a tcgen05 SMEM-source MMA (SMEM layout
atom -> tiled MMA -> partition -> cute.gemm) instead of reverse-engineering the
layout verifier.

Anti-hang: this is a CPU-only file read (no GPU kernel launch), but still timeout=90
and run under local `timeout 120`.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-example")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import os
    import glob

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages"
    print("=" * 72)
    print("SEARCH FOR EXAMPLES / blackwell dense gemm")
    print("=" * 72)
    # Find example files
    for root, dirs, files in os.walk("/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl"):
        for fn in files:
            low = fn.lower()
            if "gemm" in low and fn.endswith(".py"):
                print("  GEMM file:", os.path.join(root, fn))

    print("\n=== look for 'examples' dir anywhere ===")
    for root, dirs, files in os.walk("/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl"):
        if "example" in root.lower():
            print("  example dir:", root, "->", files[:20])

    # Dump make_trivial_tiled_mma (new API) + tile_to_mma_shape + get_smem_layout_atom
    bh = base + "/cutlass/utils/blackwell_helpers.py"
    print("\n" + "=" * 72)
    print("blackwell_helpers.py — make_trivial_tiled_mma + tile_to_mma_shape + _bind_mma_args + get_smem_layout_atom_ab")
    print("=" * 72)
    with open(bh) as f:
        src = f.read()
    lines = src.split("\n")
    for marker, span in [
        ("def make_trivial_tiled_mma", 90),
        ("def _bind_mma_args", 120),
        ("def tile_to_mma_shape", 50),
        ("def get_smem_layout_atom_ab", 70),
        ("def make_smem_layout_a", 50),
        ("def make_smem_layout_b", 50),
    ]:
        print(f"\n----- {marker} -----")
        for i, line in enumerate(lines):
            if marker in line and line.strip().startswith("def"):
                for j in range(i, min(len(lines), i + span)):
                    print(f"  {j+1}: {lines[j]}")
                break
        else:
            print("  (not found)")


@app.local_entrypoint()
def main():
    probe.remote()
