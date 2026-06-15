"""
Probe 2: find the CUBLAS_COMPUTE_32F_EMULATED_16BFX9 enum value + confirm we can
drive cublasLt via ctypes on the nvidia-cu13 libcublasLt.so.13, AND check what
libcublasLt torch itself loads (so we use the same one that's already initialised).

Run: conda activate modal && PYTHONUTF8=1 modal run probe_cublaslt2.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("cublaslt-probe2", image=image)


@app.function(gpu="B200", timeout=900)
def probe():
    import os, glob, subprocess, torch, ctypes

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))

    # ── (A) Which libcublasLt is actually loaded into THIS process? ─────────────
    print("\n" + "=" * 74)
    print("(A) loaded shared libs containing 'cublas'")
    print("=" * 74)
    pid = os.getpid()
    try:
        with open(f"/proc/{pid}/maps") as f:
            libs = sorted({ln.split()[-1] for ln in f
                           if "cublas" in ln.lower() and ".so" in ln})
        for L in libs:
            print("   ", L)
    except Exception as e:
        print("  maps read failed:", e)

    # torch needs a cuda context first
    torch.cuda.init()
    _ = torch.ones(4, device="cuda") @ torch.ones(4, device="cuda")
    torch.cuda.synchronize()
    print("\n  after a cuBLAS call:")
    with open(f"/proc/{pid}/maps") as f:
        libs = sorted({ln.split()[-1] for ln in f
                       if "cublas" in ln.lower() and ".so" in ln})
    for L in libs:
        print("   ", L)

    # ── (B) Grep cuBLAS headers (if present) for the EMULATED enum value ───────
    print("\n" + "=" * 74)
    print("(B) search for cublas_api.h / the EMULATED_16BFX9 enum")
    print("=" * 74)
    hdrs = []
    for pat in ["/usr/local/cuda*/include/cublas_api.h",
                "/usr/local/lib/python*/site-packages/nvidia/*/include/cublas_api.h",
                "/usr/include/cublas_api.h"]:
        hdrs += glob.glob(pat)
    print("  headers found:", hdrs)
    for h in hdrs:
        try:
            with open(h) as f:
                txt = f.read()
            i = txt.find("cublasComputeType_t")
            if i >= 0:
                # print the enum block
                j = txt.find("}", i)
                block = txt[txt.rfind("enum", 0, i):j + 1]
                print(f"\n  --- {h} (cublasComputeType_t enum) ---")
                for ln in block.splitlines():
                    if "EMULAT" in ln or "16BF" in ln or "COMPUTE_32F" in ln or "BFX" in ln:
                        print("   ", ln.strip())
        except Exception as e:
            print("  read failed", h, e)

    # Look for any symbol/string in the lib mentioning the emulated type
    print("\n  strings in libcublasLt for 'EMULAT'/'16BF':")
    libpath = "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13"
    try:
        out = subprocess.run(["strings", libpath], capture_output=True, text=True, timeout=60)
        hits = [l for l in out.stdout.splitlines()
                if ("EMULAT" in l.upper() or "16BF" in l.upper() or "BFX" in l.upper())][:30]
        for l in hits:
            print("   ", l)
        if not hits:
            print("    (no matching strings)")
    except Exception as e:
        print("  strings failed:", e)

    # ── (C) Can we dlopen libcublasLt and resolve the core matmul symbols? ─────
    print("\n" + "=" * 74)
    print("(C) ctypes dlopen libcublasLt + resolve symbols")
    print("=" * 74)
    try:
        lt = ctypes.CDLL(libpath)
        for sym in ["cublasLtCreate", "cublasLtMatmul", "cublasLtMatmulDescCreate",
                    "cublasLtMatmulDescSetAttribute", "cublasLtMatrixLayoutCreate",
                    "cublasLtMatmulPreferenceCreate", "cublasLtMatmulAlgoGetHeuristic",
                    "cublasLtMatmulDescDestroy", "cublasLtMatrixLayoutDestroy",
                    "cublasLtDestroy", "cublasLtGetVersion"]:
            ok = hasattr(lt, sym)
            print(f"    {sym:42} {'OK' if ok else 'MISSING'}")
        try:
            v = lt.cublasLtGetVersion()
            print(f"    cublasLtGetVersion() = {v}")
        except Exception as e:
            print("    version call failed:", e)
    except Exception as e:
        print("  dlopen failed:", e)


@app.local_entrypoint()
def main():
    probe.remote()
