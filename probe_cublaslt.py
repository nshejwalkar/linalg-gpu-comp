"""
Probe BF16x9 (CUBLAS_COMPUTE_32F_EMULATED_16BFX9) reachability on B200 / torch 2.12+cu130.

Tests, in order:
  (0) What does cuda.bindings expose? (cublas / cublasLt submodules, the EMULATED enum)
  (1) torch backend flags / env vars that might flip cuBLAS to the emulated compute type.
  (2) cublasLt strided-batched GEMM via cuda.bindings.cublasLt requesting the
      EMULATED_16BFX9 compute type -> validate exact-FP32 + time vs torch.bmm FP32.

Run: conda activate modal && PYTHONUTF8=1 modal run probe_cublaslt.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("cublaslt-probe", image=image)


@app.function(gpu="B200", timeout=900)
def probe():
    import torch
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))

    # ────────────────────────────────────────────────────────────────────────
    # (0) What's in cuda.bindings?
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("(0) cuda.bindings submodules")
    print("=" * 74)
    import importlib, pkgutil
    import cuda.bindings as cb
    print("  cuda.bindings:", getattr(cb, "__version__", "?"))
    subs = []
    try:
        for m in pkgutil.iter_modules(cb.__path__):
            subs.append(m.name)
    except Exception as e:
        print("  iter_modules failed:", e)
    print("  submodules:", subs)

    # Try to import cublas / cublasLt bindings.
    for modname in ["cuda.bindings.cublas", "cuda.bindings.cublasLt"]:
        try:
            m = importlib.import_module(modname)
            print(f"\n  OK import {modname}")
            # Find the EMULATED compute-type enum members.
            names = [n for n in dir(m) if "EMULAT" in n.upper() or "16BF" in n.upper()
                     or "BFX" in n.upper()]
            print(f"    EMULATED-ish top-level names: {names}")
            # cublasComputeType enum
            for ename in ["cublasComputeType_t", "ComputeType"]:
                if hasattr(m, ename):
                    enum = getattr(m, ename)
                    members = [x for x in dir(enum) if not x.startswith("_")]
                    emu = [x for x in members if "EMULAT" in x.upper() or "16BF" in x.upper()
                           or "BFX" in x.upper()]
                    print(f"    {ename}: {len(members)} members; EMULATED: {emu}")
                    for x in emu:
                        try:
                            print(f"        {x} = {int(getattr(enum, x))}")
                        except Exception:
                            print(f"        {x} = {getattr(enum, x)!r}")
        except Exception as e:
            print(f"  -- import {modname} failed: {type(e).__name__}: {e}")

    # Also check ctypes route on the shared libs (fallback if bindings lack the enum).
    print("\n  ctypes scan for libcublasLt / libcublas:")
    import ctypes.util, glob, os
    for pat in ["/usr/local/cuda*/lib64/libcublasLt.so*", "/usr/lib/x86_64-linux-gnu/libcublasLt.so*",
                "*/torch/lib/libcublasLt*.so*", "*/nvidia/cu*/lib/libcublasLt.so*",
                "/usr/local/lib/python*/site-packages/nvidia/*/lib/libcublasLt.so*"]:
        hits = glob.glob(pat)
        if hits:
            print(f"    {pat} -> {hits}")
    # torch ships its own cublasLt; find it
    import torch as _t
    tdir = os.path.dirname(_t.__file__)
    for root, _, fs in os.walk(os.path.dirname(tdir)):
        for f in fs:
            if "cublasLt" in f and f.endswith((".so", ".so.13", ".so.12")) or \
               (f.startswith("libcublasLt") and ".so" in f):
                print(f"    found: {os.path.join(root, f)}")

    # ────────────────────────────────────────────────────────────────────────
    # (1) torch backend flags / env
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("(1) torch.backends.cuda.matmul flags")
    print("=" * 74)
    mm = torch.backends.cuda.matmul
    print("  attrs:", [a for a in dir(mm) if not a.startswith("_")])
    for a in dir(mm):
        if a.startswith("_"):
            continue
        try:
            print(f"    {a} = {getattr(mm, a)}")
        except Exception:
            pass


@app.local_entrypoint()
def main():
    probe.remote()
