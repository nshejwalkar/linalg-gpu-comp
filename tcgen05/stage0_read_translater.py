"""Dump the full tl_dot_scaled_blackwell + plain blackwell matmul helper + the
tensor_memory_descriptor.load/store source from the triton wheel so we have the
EXACT working Gluon idiom to copy for Stage 1."""
import modal

clean_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("qr-tcgen05-read-translater")


@app.function(gpu="B200", image=clean_image, timeout=900)
def dump():
    import os, glob, triton
    troot = os.path.dirname(triton.__file__)
    # the translater dir
    tfile = None
    for py in glob.glob(os.path.join(troot, "**", "*.py"), recursive=True):
        if "triton_to_gluon" in py and py.endswith(("ops.py", "core.py", "language.py")):
            tfile = py
    # fallback: find the file containing tl_dot_scaled_blackwell
    if tfile is None:
        for py in glob.glob(os.path.join(troot, "**", "*.py"), recursive=True):
            try:
                with open(py) as f:
                    if "tl_dot_scaled_blackwell" in f.read():
                        tfile = py
                        break
            except Exception:
                pass
    print("translater file:", tfile)
    with open(tfile) as f:
        lines = f.readlines()
    # print lines 280-420 (the dot helpers)
    print("\n===== tl_dot* helpers (lines 280-420) =====")
    for i in range(279, min(420, len(lines))):
        print(f"{i+1:4}: {lines[i].rstrip()}")

    # The blackwell descriptor load/store/index/slice source.
    import triton.experimental.gluon.language.nvidia.blackwell as bw
    with open(bw.__file__) as f:
        bl = f.readlines()
    print("\n===== blackwell tensor_memory_descriptor + TensorMemoryLayout (36-420) =====")
    for i in range(35, min(420, len(bl))):
        print(f"{i+1:4}: {bl[i].rstrip()}")


@app.local_entrypoint()
def main():
    dump.remote()
