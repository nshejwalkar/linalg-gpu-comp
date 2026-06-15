"""
opus_probe_accum.py — how to set the ACCUMULATE field on a tcgen05 tiled_mma,
and how the multi-K-tile gemm loop toggles it (first tile accum=False, rest True).
Pure source dump + signature inspection. No GPU kernel => no hang.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-accum")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import inspect
    import cutlass.cute as cute
    import cutlass.cute.atom as atom
    import cutlass.cute.nvgpu.tcgen05 as tc

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages"

    # MmaAtom.set / set_field methods
    print("=== MmaAtom methods with 'set' or 'field' ===")
    for name in dir(atom.MmaAtom):
        if "set" in name.lower() or "field" in name.lower() or "accum" in name.lower():
            print("  ", name)

    # dump atom.py around 'def set' and 'Field'
    p = base + "/cutlass/cute/atom.py"
    with open(p) as f:
        lines = f.read().split("\n")
    print("\n=== atom.py: 'def set' / 'Field' / 'accum' lines ===")
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (s.startswith("def set") or "Field" in ln or "accum" in ln.lower()
                or "def make_fragment_C" in s):
            print(f"  {i+1}: {ln.rstrip()}")

    # Show full body of any 'def set' in MmaAtom region (300-470)
    print("\n=== atom.py lines 440-540 (TiledMma body, look for set/with) ===")
    for j in range(440, min(len(lines), 540)):
        print(f"{j+1}: {lines[j]}")

    # mma.py: how Field is used; look for .set / set_op_attr / make_mma_atom accumulate
    pm = base + "/cutlass/cute/nvgpu/tcgen05/mma.py"
    with open(pm) as f:
        ml = f.read().split("\n")
    print("\n=== mma.py: 'accum' / 'Field' / 'def set' / 'ACCUMULATE' ===")
    for i, ln in enumerate(ml):
        if ("accum" in ln.lower() or "Field" in ln or "def set" in ln or "with_" in ln):
            print(f"  {i+1}: {ln.rstrip()}")


@app.local_entrypoint()
def main():
    probe.remote()
