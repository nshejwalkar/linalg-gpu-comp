"""
opus_probe_gemmdisp.py — dump the rest of cute.gemm (rank validation + dispatch)
and the tcgen05 MmaF16BF16Trait._verify_fragment_A/B + the gemm helper that builds
fragments, so we know the EXACT shape gemm wants for A/B/D in the SS (smem) path.
Pure source dump => no hang.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-gemmdisp")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages"

    # full gemm() body
    p = base + "/cutlass/cute/algorithm.py"
    with open(p) as f:
        lines = f.read().split("\n")
    print("===== cute.gemm full body =====")
    for i, ln in enumerate(lines):
        if ln.strip().startswith("def gemm("):
            for j in range(i, min(len(lines), i + 170)):
                print(f"{j+1}: {lines[j]}")
            break

    # _verify_fragment_A / _verify_fragment_B in tcgen05 mma.py
    pm = base + "/cutlass/cute/nvgpu/tcgen05/mma.py"
    with open(pm) as f:
        ml = f.read().split("\n")
    for marker in ["def _verify_fragment_A", "def _verify_fragment_B", "def _verify_fragment_C"]:
        print(f"\n===== mma.py {marker} =====")
        for i, ln in enumerate(ml):
            if marker in ln:
                for j in range(i, min(len(ml), i + 40)):
                    print(f"{j+1}: {ml[j]}")
                break


@app.local_entrypoint()
def main():
    probe.remote()
