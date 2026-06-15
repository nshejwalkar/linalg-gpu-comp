"""
opus_probe_recipe.py — print the FULL bodies of the canonical helpers so we can
copy the known-good tcgen05 SMEM-source MMA recipe verbatim:
  make_trivial_tiled_mma, make_smem_layout_a, make_smem_layout_b,
  and how partition_A/partition_B are meant to be used (grep examples).
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus-probe-recipe")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def probe():
    import os

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages"
    bh = base + "/cutlass/utils/blackwell_helpers.py"
    with open(bh) as f:
        lines = f.read().split("\n")

    def dump(marker, span):
        print(f"\n===== {marker} =====")
        for i, line in enumerate(lines):
            if marker in line and line.strip().startswith("def"):
                for j in range(i, min(len(lines), i + span)):
                    print(f"{j+1}: {lines[j]}")
                return
        print("  (not found)")

    dump("def make_trivial_tiled_mma", 95)
    dump("def make_smem_layout_a", 60)
    dump("def make_smem_layout_b", 60)

    # Find any example/test that calls partition_A on a tiled_mma + cute.gemm
    print("\n" + "=" * 72)
    print("GREP for partition_A / partition_B / make_fragment_A usage across wheel")
    print("=" * 72)
    root = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl"
    hits = 0
    for r, dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(r, fn)
            try:
                with open(p) as f:
                    txt = f.read()
            except Exception:
                continue
            if "partition_A" in txt and ("make_fragment" in txt or "thr_mma" in txt or "get_slice" in txt):
                print(f"\n--- {p} ---")
                tl = txt.split("\n")
                for idx, ln in enumerate(tl):
                    if any(k in ln for k in ["partition_A", "partition_B", "make_fragment_A",
                                              "make_fragment_B", "make_fragment_C", "get_slice",
                                              "thr_mma", ".partition_C", "cute.gemm", "tcgen05.commit",
                                              "make_smem_layout_a", "make_smem_layout_b",
                                              "make_trivial_tiled_mma"]):
                        print(f"  {idx+1}: {ln.rstrip()}")
                hits += 1
                if hits >= 4:
                    break
        if hits >= 4:
            break


@app.local_entrypoint()
def main():
    probe.remote()
