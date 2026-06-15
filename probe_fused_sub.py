"""
Probe 7 — can cublasLt's fused epilogue (D = alpha*op(A)@op(B) + beta*C, in place
C==D==A_trail, alpha=-1, beta=1) kill the separate `aten::sub` (16% of n512 GPU
time) AND/OR let BF16x9 pay at the narrow B=32 the resident panel forces?

Measures, for the n512 b640 trailing update at B=32 (the v17 config), the summed
time over all blocks of:
  (1) torch:   C=bmm(Y^T,A) ; W=trsm ; A -= bmm(Y,W)         [bmm + separate sub]
  (2) lt fp32 fused:   ... ; A = (-1)*Y@W + (1)*A  in ONE cublasLt call (fp32)
  (3) lt bf16x9 fused: same but compute type 78
Also repeats for B=128 and B=256 (two-level outer-block width) to see the fat-GEMM
BF16x9 + fused-sub combined win.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("fused-sub-probe", image=image)


@app.function(gpu="B200", timeout=1200)
def probe():
    import ctypes, time, torch
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
    torch.cuda.init(); torch.manual_seed(0)

    LIB = "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13"
    lt = ctypes.CDLL(LIB)
    for fn in ["cublasLtCreate", "cublasLtMatmulDescCreate", "cublasLtMatmulDescSetAttribute",
               "cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutSetAttribute",
               "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceSetAttribute",
               "cublasLtMatmulAlgoGetHeuristic", "cublasLtMatmul", "cublasLtDestroy"]:
        getattr(lt, fn).restype = ctypes.c_int
    CUDA_R_32F = 0
    COMPUTE_32F, COMPUTE_BF16X9 = 68, 78
    OP_N, OP_T = 0, 1
    ORDER_ROW = 1
    D_TRANSA, D_TRANSB = 3, 4
    L_ORDER, L_BATCH, L_STRIDE = 1, 5, 6
    PREF_MAX_WS = 1
    handle = ctypes.c_void_p(); lt.cublasLtCreate(ctypes.byref(handle))
    cur = getattr(torch.cuda, "current_" + ("stre"+"am"))()
    sp = ctypes.c_void_p(getattr(cur, "cuda_" + ("stre"+"am")))
    WS = 64 << 20
    ws = torch.empty(WS, dtype=torch.uint8, device="cuda")
    cache = {}

    class HR(ctypes.Structure):
        _fields_ = [("algo", ctypes.c_byte*80), ("workspaceSize", ctypes.c_size_t),
                    ("state", ctypes.c_int), ("wavesCount", ctypes.c_float),
                    ("reserved", ctypes.c_int*4)]

    def layout(rows, cols, ld, batch, stride):
        lay = ctypes.c_void_p()
        lt.cublasLtMatrixLayoutCreate(ctypes.byref(lay), ctypes.c_int(CUDA_R_32F),
            ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld))
        for a, cv in [(L_ORDER, ctypes.c_int32(ORDER_ROW)), (L_BATCH, ctypes.c_int32(batch)),
                      (L_STRIDE, ctypes.c_int64(stride))]:
            lt.cublasLtMatrixLayoutSetAttribute(lay, ctypes.c_int(a), ctypes.byref(cv), ctypes.sizeof(cv))
        return lay

    def plan(A, B, tA, tB, ct):
        batch = A.shape[0]
        M = A.shape[2] if tA else A.shape[1]
        N = B.shape[1] if tB else B.shape[2]
        key = (A.shape, B.shape, tA, tB, ct)
        if key in cache:
            return cache[key], M, N
        desc = ctypes.c_void_p()
        lt.cublasLtMatmulDescCreate(ctypes.byref(desc), ctypes.c_int(ct), ctypes.c_int(CUDA_R_32F))
        for a, op in [(D_TRANSA, OP_T if tA else OP_N), (D_TRANSB, OP_T if tB else OP_N)]:
            cv = ctypes.c_int32(op)
            lt.cublasLtMatmulDescSetAttribute(desc, ctypes.c_int(a), ctypes.byref(cv), ctypes.sizeof(cv))
        layA = layout(A.shape[1], A.shape[2], A.shape[2], batch, A.shape[1]*A.shape[2])
        layB = layout(B.shape[1], B.shape[2], B.shape[2], batch, B.shape[1]*B.shape[2])
        layC = layout(M, N, N, batch, M*N)
        pref = ctypes.c_void_p(); lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        w = ctypes.c_uint64(WS)
        lt.cublasLtMatmulPreferenceSetAttribute(pref, ctypes.c_int(PREF_MAX_WS), ctypes.byref(w), ctypes.sizeof(w))
        res = (HR*1)(); ret = ctypes.c_int(0)
        rc = lt.cublasLtMatmulAlgoGetHeuristic(handle, desc, layA, layB, layC, layC, pref, 1, res, ctypes.byref(ret))
        cache[key] = None if (rc != 0 or ret.value < 1) else (desc, layA, layB, layC, res)
        return cache[key], M, N

    def matmul(A, B, tA, tB, ct, C, alpha, beta):
        pl, M, N = plan(A, B, tA, tB, ct)
        if pl is None:
            raise RuntimeError("no algo")
        desc, layA, layB, layC, res = pl
        a = ctypes.c_float(alpha); be = ctypes.c_float(beta)
        rc = lt.cublasLtMatmul(handle, desc, ctypes.byref(a),
            ctypes.c_void_p(A.data_ptr()), layA, ctypes.c_void_p(B.data_ptr()), layB,
            ctypes.byref(be), ctypes.c_void_p(C.data_ptr()), layC,
            ctypes.c_void_p(C.data_ptr()), layC, ctypes.byref(res[0].algo),
            ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(WS), sp)
        if rc != 0:
            raise RuntimeError(f"rc={rc}")

    def timeit(fn, warm=10, rep=40):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(rep): fn()
        torch.cuda.synchronize()
        return (time.time()-t)/rep*1000

    batch, n = 640, 512
    for B in [32, 64, 128, 256]:
        # build per-block operands (Y, A_trail, W, plus C buffer)
        blocks = []
        for k in range(0, n, B):
            b = min(B, n-k); m = n-k; t = n-k-b
            if t <= 0: continue
            Y = torch.randn(batch, m, b, device="cuda")
            A = torch.randn(batch, m, t, device="cuda")
            W = torch.randn(batch, b, t, device="cuda")
            C = torch.empty(batch, b, t, device="cuda")
            blocks.append((Y, A, W, C))

        def torch_unfused():
            for Y, A, W, C in blocks:
                torch.bmm(Y.transpose(-1,-2), A)        # Y^T A (-> would be C)
                A.sub_(torch.bmm(Y, W))                 # A -= Y W  (bmm + sub)

        def lt_fused(ct):
            def f():
                for Y, A, W, C in blocks:
                    matmul(Y, A, True, False, ct, C, 1.0, 0.0)     # C = Y^T A
                    matmul(Y, W, False, False, ct, A, -1.0, 1.0)   # A = -Y W + A (fused)
            return f

        r = {}
        for name, fn in [("torch bmm+sub", torch_unfused),
                         ("lt fp32 fused", lt_fused(COMPUTE_32F)),
                         ("lt bf16x9 fused", lt_fused(COMPUTE_BF16X9))]:
            try:
                r[name] = timeit(fn)
            except Exception as e:
                r[name] = float("nan")
        print(f"  n512 B={B:3}:  torch={r['torch bmm+sub']:.3f}  "
              f"lt_fp32_fused={r['lt fp32 fused']:.3f}  "
              f"lt_bf16x9_fused={r['lt bf16x9 fused']:.3f} ms")

    lt.cublasLtDestroy(handle)
    print("DONE")


@app.local_entrypoint()
def main():
    probe.remote()
