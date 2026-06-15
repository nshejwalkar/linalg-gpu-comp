"""
opus_probe_atom.py — dump MmaAtom/TiledMma/ThrMma partition + make_fragment
method bodies from cute/atom.py so we know exactly how to feed cute.gemm.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-atom")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages"
    p = base + "/cutlass/cute/atom.py"
    with open(p) as f:
        lines = f.read().split("\n")

    def dump(start_line, span, label):
        print(f"\n===== {label} (around line {start_line}) =====")
        for j in range(start_line - 1, min(len(lines), start_line - 1 + span)):
            print(f"{j+1}: {lines[j]}")

    # From prior grep: make_fragment_A@389, make_fragment_B@408, make_fragment_C@424,
    # get_slice(ThrMma)@536, partition_A@687, partition_B@705
    dump(380, 70, "make_fragment_A / B / C")
    dump(525, 60, "ThrMma get_slice + partition methods")
    dump(680, 60, "MmaAtom/TiledMma partition_A / partition_B / partition_C")

    # Also find partition_C and TiledMma class + get_slice for tiled
    print("\n===== grep partition_C / class TiledMma / class ThrMma / def get_slice =====")
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (s.startswith("def partition_C") or s.startswith("class TiledMma")
                or s.startswith("class ThrMma") or s.startswith("class MmaAtom")
                or s.startswith("def partition_A") or s.startswith("def partition_B")
                or s.startswith("def get_slice") or s.startswith("def make_fragment_C")
                or "partition_shape_A" in s or "partition_shape_B" in s
                or s.startswith("def _make_tiled_mma") or s.startswith("def make_tiled_mma")):
            print(f"  {i+1}: {ln.rstrip()}")


@app.local_entrypoint()
def main():
    probe.remote()
