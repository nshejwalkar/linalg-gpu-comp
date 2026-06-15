"""
Probe the grader-mirror image (torch 2.12+cu130 + triton, same stack the grader runs)
for which kernel backends are actually usable in a submission. Run:
    conda activate modal && modal run probe_env.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("qr-probe", image=image)


@app.function(gpu="B200", timeout=600)
def probe():
    import importlib, shutil

    print("=" * 70)
    print("PYTHON IMPORTS (must be importable ON the grader to use directly)")
    print("=" * 70)
    for name in ["torch", "triton", "numpy",
                 "cuda", "cuda.bindings", "cuda.bindings.nvrtc", "cuda.bindings.driver",
                 "cupy", "cutlass", "cutlass_library", "nvidia.cutlass", "cute", "numba"]:
        try:
            m = importlib.import_module(name)
            print(f"  OK   {name:24} {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"  --   {name:24} ({type(e).__name__})")

    print("\n" + "=" * 70)
    print("SYSTEM TOOLS on PATH (nvcc/ninja => load_inline; ptxas => offline compile)")
    print("=" * 70)
    for t in ["nvcc", "ptxas", "ninja", "cicc", "gcc", "g++", "cc"]:
        print(f"  {t:8} -> {shutil.which(t)}")

    import torch
    print(f"\n  torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"cap {torch.cuda.get_device_capability(0)}")

    print("\n" + "=" * 70)
    print("TRITON AOT: can we extract PTX/cubin from a compiled kernel? (embed route)")
    print("=" * 70)
    try:
        import triton, triton.language as tl

        @triton.jit
        def _dbl(x_ptr, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) * 2.0, mask=i < n)

        x = torch.ones(64, device="cuda")
        c = _dbl.warmup(x, 64, BLOCK=64, grid=(1,))
        c._init_handles()
        asm = getattr(c, "asm", {})
        print(f"  compiled.asm keys: {list(asm.keys())}")
        for k in ("ptx", "cubin"):
            if k in asm:
                print(f"    {k}: {len(asm[k])} bytes")
        print(f"  n_regs={getattr(c,'n_regs',None)} spills={getattr(c,'n_spills',None)} "
              f"smem={getattr(getattr(c,'metadata',None),'shared',None)}")
    except Exception as e:
        import traceback; print(f"  FAILED: {type(e).__name__}: {e}"); traceback.print_exc()

    print("\n" + "=" * 70)
    print("NVRTC compile + DRIVER load (the embedded-PTX path, no nvcc needed)")
    print("=" * 70)
    try:
        from cuda.bindings import nvrtc, driver

        src = b'extern "C" __global__ void f(float* x){ int i=threadIdx.x; x[i]=x[i]*2.0f; }'
        e, prog = nvrtc.nvrtcCreateProgram(src, b"f.cu", 0, [], [])
        (e2,) = nvrtc.nvrtcCompileProgram(prog, 0, [])
        e3, sz = nvrtc.nvrtcGetPTXSize(prog)
        ptx = b" " * sz
        (e4,) = nvrtc.nvrtcGetPTX(prog, ptx)
        print(f"  nvrtc compile -> PTX {sz} bytes  (rc {e},{e2},{e3},{e4})")
        # load the PTX via the driver API (uses torch's existing context)
        torch.cuda.init()
        (r0,) = driver.cuInit(0)
        e5, mod = driver.cuModuleLoadData(ptx)
        e6, fn = driver.cuModuleGetFunction(mod, b"f")
        print(f"  driver cuModuleLoadData + GetFunction OK  (rc {r0},{e5},{e6})  -> embedded-PTX path WORKS")
    except Exception as e:
        import traceback; print(f"  FAILED: {type(e).__name__}: {e}"); traceback.print_exc()

    print("\n" + "=" * 70)
    print("torch load_inline (raw CUDA via nvcc+ninja)")
    print("=" * 70)
    try:
        from torch.utils.cpp_extension import load_inline
        cuda_src = (
            'torch::Tensor dbl(torch::Tensor x){ return x*2; }'
        )
        m = load_inline(name="probe_inline", cpp_sources=[cuda_src],
                        functions=["dbl"], verbose=False)
        print(f"  load_inline OK: dbl(3)={m.dbl(torch.tensor([3.0]))}")
    except Exception as e:
        print(f"  FAILED (expected if no nvcc/ninja): {type(e).__name__}: {str(e)[:160]}")


@app.local_entrypoint()
def main():
    probe.remote()
