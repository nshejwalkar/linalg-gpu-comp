"""
opus2_ltprobe.py — verify cublasLt precision on the batched-K32 trailing shape.

In opus2_bench the STRIDED-BATCHED cublasLt-78 call returned ~5µs but rel-vs-FP64 ≈ 2e-3
(≈ plain bf16, NOT the ~3e-7 of true BF16x9 / compute-type-78). Before claiming "cuBLAS is a
free FP32 trailing win" we must know what precision actually ran. This probe compares, on the
SAME batched-K32 data, FOUR cublasLt configurations + torch references:

  refbmm32   torch.bmm(A.fp32, B.fp32)                       — true FP32 (rel vs FP64 baseline)
  refbmmbf16 torch.bmm(A.bf16, B.bf16).fp32                  — plain bf16 (the ~2e-3 floor)
  LT78-batch strided-batched, COMPUTE_78                     — what opus2_bench used
  LT32-batch strided-batched, COMPUTE_32F (plain fp32 emul) — control
  LT78-loop  per-matrix COMPUTE_78 in a Python loop          — the PROVEN B6/B9 exact path (rel 3e-7)

Reports rel-vs-FP64 + µs for each, so we can tell whether type-78 really engages in the
batched path (and whether a cuBLAS route is FP32-exact enough for the B4 trailing gate).

Anti-hang: server timeout=90; local timeout 120, FOREGROUND.
"""

import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-opus2-ltprobe")


@app.function(gpu="B200", image=cutlass_image, timeout=90, retries=0)
def probe():
    import torch, time, ctypes, traceback

    print("=" * 92)
    print("OPUS2 LT PRECISION PROBE — what does cublasLt actually run on batched-K32?")
    print("=" * 92)

    lt = None
    for name in ("libcublasLt.so.13",
                 "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13",
                 "libcublasLt.so"):
        try:
            lt = ctypes.CDLL(name); break
        except OSError:
            continue
    assert lt is not None, "no cublasLt"
    c_void_p, c_int, c_size_t, c_int64, byref = (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_int64, ctypes.byref)
    for fn in ["cublasLtCreate","cublasLtMatmul","cublasLtMatmulDescCreate",
               "cublasLtMatmulDescSetAttribute","cublasLtMatrixLayoutCreate",
               "cublasLtMatrixLayoutSetAttribute",
               "cublasLtMatmulPreferenceCreate","cublasLtMatmulPreferenceSetAttribute",
               "cublasLtMatmulAlgoGetHeuristic"]:
        getattr(lt, fn).restype = c_int
    handle = c_void_p(); assert lt.cublasLtCreate(byref(handle)) == 0

    CUDA_R_32F = 0; CUDA_R_16BF = 14
    COMPUTE_32F = 0      # CUBLAS_COMPUTE_32F
    COMPUTE_78 = 78      # CUBLAS_COMPUTE_32F_EMULATED_16BFX9
    LAYOUT_BATCH_COUNT = 4; LAYOUT_BATCH_STRIDE = 5

    class Heur(ctypes.Structure):
        _fields_ = [("algo", ctypes.c_byte*72), ("workspaceSize", c_size_t),
                    ("state", c_int), ("wavesCount", ctypes.c_float), ("reserved", c_int*4)]

    def make_layouts(M, N, K, batch=None):
        lB = c_void_p(); lA = c_void_p(); lC = c_void_p()
        lt.cublasLtMatrixLayoutCreate(byref(lB), CUDA_R_16BF, N, K, N)
        lt.cublasLtMatrixLayoutCreate(byref(lA), CUDA_R_16BF, K, M, K)
        lt.cublasLtMatrixLayoutCreate(byref(lC), CUDA_R_32F, N, M, N)
        if batch is not None:
            for lay, stride in ((lB, N*K), (lA, M*K), (lC, M*N)):
                bc = c_int(batch)
                lt.cublasLtMatrixLayoutSetAttribute(lay, LAYOUT_BATCH_COUNT, byref(bc), ctypes.sizeof(bc))
                bs = c_int64(stride)
                lt.cublasLtMatrixLayoutSetAttribute(lay, LAYOUT_BATCH_STRIDE, byref(bs), ctypes.sizeof(bs))
        return lB, lA, lC

    def heuristic(desc, lB, lA, lC, wsbytes):
        pref = c_void_p(); lt.cublasLtMatmulPreferenceCreate(byref(pref))
        ws = c_size_t(wsbytes)
        lt.cublasLtMatmulPreferenceSetAttribute(pref, 0, byref(ws), ctypes.sizeof(ws))
        res = (Heur*1)(); cnt = c_int()
        rc = lt.cublasLtMatmulAlgoGetHeuristic(handle, desc, lB, lA, lC, lC, pref, 1, byref(res), byref(cnt))
        return rc, cnt.value, res, ws

    def bench(fn, iters=50, warmup=10):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6

    # batched-K32 representative: n512 first block.
    BATCH, M, N, K = 640, 512, 480, 32
    torch.manual_seed(7)
    A = torch.randn(BATCH, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(BATCH, K, N, device="cuda", dtype=torch.float32)
    Abf = A.bfloat16().contiguous(); Bbf = B.bfloat16().contiguous()
    ref64 = A.double() @ B.double()
    denom = ref64.abs().max()

    def relof(C):
        return ((C.double() - ref64).abs().max() / denom).item()

    print(f"  shape: BATCH={BATCH} M={M} N={N} K={K}\n")

    # ---- torch references ----
    Cbmm32 = torch.bmm(A, B); print(f"  torch.bmm FP32     rel={relof(Cbmm32):.2e}  t={bench(lambda: torch.bmm(A,B)):.1f}us")
    Cbmmbf = torch.bmm(Abf, Bbf).float(); print(f"  torch.bmm BF16     rel={relof(Cbmmbf):.2e}  (plain-bf16 floor)")

    alpha = ctypes.c_float(1.0); beta = ctypes.c_float(0.0)

    # ---- strided-batched, COMPUTE_78 ----
    for tag, ctype in (("LT78-batch", COMPUTE_78), ("LT32-batch", COMPUTE_32F)):
        try:
            C = torch.empty(BATCH, M, N, device="cuda", dtype=torch.float32)
            desc = c_void_p(); lt.cublasLtMatmulDescCreate(byref(desc), ctype, CUDA_R_32F)
            lB, lA, lC = make_layouts(M, N, K, batch=BATCH)
            rc, cnt, res, ws = heuristic(desc, lB, lA, lC, 64*1024*1024)
            if rc != 0 or cnt == 0:
                print(f"  {tag:>11}        heuristic FAIL rc={rc} cnt={cnt}"); continue
            wsbuf = torch.empty(ws.value, device="cuda", dtype=torch.uint8)
            Bp=c_void_p(Bbf.data_ptr()); Ap=c_void_p(Abf.data_ptr()); Cp=c_void_p(C.data_ptr()); wsp=c_void_p(wsbuf.data_ptr())
            def run():
                rc2 = lt.cublasLtMatmul(handle, desc, byref(alpha), Bp, lB, Ap, lA, byref(beta),
                                        Cp, lC, Cp, lC, byref(res[0].algo), wsp, ws, c_void_p(0))
                if rc2 != 0: raise RuntimeError(f"matmul rc={rc2}")
            run(); torch.cuda.synchronize()
            print(f"  {tag:>11}        rel={relof(C):.2e}  t={bench(run):.1f}us  (wavesCount={res[0].wavesCount:.2f})")
        except Exception as e:
            print(f"  {tag:>11}        FAILED: {str(e)[:80]}"); traceback.print_exc()

    # ---- per-matrix COMPUTE_78 in a loop (proven exact path) ----
    try:
        C = torch.empty(BATCH, M, N, device="cuda", dtype=torch.float32)
        desc = c_void_p(); lt.cublasLtMatmulDescCreate(byref(desc), COMPUTE_78, CUDA_R_32F)
        lB, lA, lC = make_layouts(M, N, K, batch=None)   # single-GEMM layouts
        rc, cnt, res, ws = heuristic(desc, lB, lA, lC, 32*1024*1024)
        if rc != 0 or cnt == 0:
            print(f"  LT78-loop          heuristic FAIL rc={rc} cnt={cnt}")
        else:
            wsbuf = torch.empty(ws.value, device="cuda", dtype=torch.uint8)
            wsp = c_void_p(wsbuf.data_ptr())
            Bptrs = [c_void_p(Bbf[i].data_ptr()) for i in range(BATCH)]
            Aptrs = [c_void_p(Abf[i].data_ptr()) for i in range(BATCH)]
            Cptrs = [c_void_p(C[i].data_ptr()) for i in range(BATCH)]
            algo = res[0].algo
            def run():
                for i in range(BATCH):
                    rc2 = lt.cublasLtMatmul(handle, desc, byref(alpha), Bptrs[i], lB, Aptrs[i], lA,
                                            byref(beta), Cptrs[i], lC, Cptrs[i], lC, byref(algo), wsp, ws, c_void_p(0))
                    if rc2 != 0: raise RuntimeError(f"matmul[{i}] rc={rc2}")
            run(); torch.cuda.synchronize()
            print(f"  LT78-loop          rel={relof(C):.2e}  t={bench(run, iters=20, warmup=5):.1f}us  (per-matrix x{BATCH}; CPU-loop bound)")
    except Exception as e:
        print(f"  LT78-loop          FAILED: {str(e)[:80]}"); traceback.print_exc()

    print("\n  INTERPRETATION:")
    print("   - If LT78-batch rel ~2e-3 == BF16 floor -> strided-batch did NOT engage type-78 (fell back to bf16).")
    print("   - If LT78-loop rel ~3e-7 -> type-78 IS exact per-matrix, but the batched API can't deliver it.")
    print("   - torch.bmm FP32 rel ~0 is the only confirmed-exact fast-ish option for the v19 trailing.")
    return True


@app.local_entrypoint()
def main():
    probe.remote()
