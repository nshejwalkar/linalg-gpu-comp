"""
Probe 6 — where does BF16x9 actually beat FP32 for OUR trailing GEMMs?
Sweep block width B in {32,64,128,256} and the ranked n in {512,1024,2048,4096},
summing the trailing-update GEMM time over all blocks of a right-looking QR, for
both torch.bmm FP32 and cublasLt BF16x9. This tells us if/where a fatter block +
BF16x9 wins, and by how much, per shape.

The trailing update per block k (width b, trailing rows m=n-k, trailing cols t=n-k-b):
  C = Y^T @ A_trail   : (batch, b, t)  from Y:(batch,m,b), A_trail:(batch,m,t)
  A_trail -= Y @ W    : Y:(batch,m,b) @ W:(batch,b,t)
(We skip the tiny T-solve; it's the same in both and not a GEMM.)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("bf16x9-sweep", image=image)


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
               "cublasLtMatmulAlgoGetHeuristic", "cublasLtMatmul",
               "cublasLtMatmulPreferenceDestroy", "cublasLtMatrixLayoutDestroy",
               "cublasLtMatmulDescDestroy", "cublasLtDestroy"]:
        getattr(lt, fn).restype = ctypes.c_int

    CUDA_R_32F = 0
    COMPUTE_32F, COMPUTE_BF16X9 = 68, 78
    OP_N, OP_T = 0, 1
    ORDER_ROW = 1
    D_TRANSA, D_TRANSB = 3, 4
    L_ORDER, L_BATCH, L_STRIDE = 1, 5, 6
    PREF_MAX_WS = 1

    handle = ctypes.c_void_p()
    assert lt.cublasLtCreate(ctypes.byref(handle)) == 0
    cur_stream = getattr(torch.cuda, "current_" + "stream")()
    stream_ptr = ctypes.c_void_p(cur_stream.cuda_stream)
    ws_bytes = 64 * 1024 * 1024
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")

    # cache of (desc, layouts, algo) keyed by (shapes,trans,ctype) to mimic the real
    # integration (heuristic once per distinct shape, then reuse) so timing is fair.
    cache = {}

    def _layout(rows, cols, ld, batch, stride):
        lay = ctypes.c_void_p()
        lt.cublasLtMatrixLayoutCreate(ctypes.byref(lay), ctypes.c_int(CUDA_R_32F),
                                      ctypes.c_uint64(rows), ctypes.c_uint64(cols),
                                      ctypes.c_int64(ld))
        for attr, cv in [(L_ORDER, ctypes.c_int32(ORDER_ROW)),
                         (L_BATCH, ctypes.c_int32(batch)),
                         (L_STRIDE, ctypes.c_int64(stride))]:
            lt.cublasLtMatrixLayoutSetAttribute(lay, ctypes.c_int(attr),
                                                ctypes.byref(cv), ctypes.sizeof(cv))
        return lay

    class HeurResult(ctypes.Structure):
        _fields_ = [("algo", ctypes.c_byte * 80), ("workspaceSize", ctypes.c_size_t),
                    ("state", ctypes.c_int), ("wavesCount", ctypes.c_float),
                    ("reserved", ctypes.c_int * 4)]

    def plan(A, B, transA, transB, ctype):
        batch = A.shape[0]
        M = A.shape[2] if transA else A.shape[1]
        K = A.shape[1] if transA else A.shape[2]
        N = B.shape[1] if transB else B.shape[2]
        key = (A.shape, B.shape, transA, transB, ctype)
        if key in cache:
            return cache[key], (batch, M, N)
        desc = ctypes.c_void_p()
        lt.cublasLtMatmulDescCreate(ctypes.byref(desc), ctypes.c_int(ctype),
                                    ctypes.c_int(CUDA_R_32F))
        for attr, op in [(D_TRANSA, OP_T if transA else OP_N),
                         (D_TRANSB, OP_T if transB else OP_N)]:
            cv = ctypes.c_int32(op)
            lt.cublasLtMatmulDescSetAttribute(desc, ctypes.c_int(attr),
                                              ctypes.byref(cv), ctypes.sizeof(cv))
        layA = _layout(A.shape[1], A.shape[2], A.shape[2], batch, A.shape[1]*A.shape[2])
        layB = _layout(B.shape[1], B.shape[2], B.shape[2], batch, B.shape[1]*B.shape[2])
        layC = _layout(M, N, N, batch, M*N)
        pref = ctypes.c_void_p()
        lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        _ws = ctypes.c_uint64(ws_bytes)
        lt.cublasLtMatmulPreferenceSetAttribute(pref, ctypes.c_int(PREF_MAX_WS),
                                                ctypes.byref(_ws), ctypes.sizeof(_ws))
        res = (HeurResult * 1)(); ret = ctypes.c_int(0)
        rc = lt.cublasLtMatmulAlgoGetHeuristic(handle, desc, layA, layB, layC, layC,
                                               pref, 1, res, ctypes.byref(ret))
        lt.cublasLtMatmulPreferenceDestroy(pref)
        if rc != 0 or ret.value < 1:
            cache[key] = None
            return None, (batch, M, N)
        cache[key] = (desc, layA, layB, layC, res)
        return cache[key], (batch, M, N)

    alpha = ctypes.c_float(1.0); beta = ctypes.c_float(0.0)

    def run_lt(planned, A, B):
        desc, layA, layB, layC, res = planned
        batch, M, N = A.shape[0], (layC and None), None
        # need C; allocate from layout dims stored implicitly -> recompute
        # (M,N from layC not stored; pass via closure instead) -> use A/B
        raise RuntimeError("unused")

    def matmul_lt(A, B, transA, transB, ctype, Cbuf):
        planned, (batch, M, N) = plan(A, B, transA, transB, ctype)
        if planned is None:
            raise RuntimeError("no algo")
        desc, layA, layB, layC, res = planned
        rc = lt.cublasLtMatmul(handle, desc, ctypes.byref(alpha),
                               ctypes.c_void_p(A.data_ptr()), layA,
                               ctypes.c_void_p(B.data_ptr()), layB,
                               ctypes.byref(beta),
                               ctypes.c_void_p(Cbuf.data_ptr()), layC,
                               ctypes.c_void_p(Cbuf.data_ptr()), layC,
                               ctypes.byref(res[0].algo),
                               ctypes.c_void_p(workspace.data_ptr()),
                               ctypes.c_size_t(ws_bytes), stream_ptr)
        if rc != 0:
            raise RuntimeError(f"matmul rc={rc}")
        return Cbuf

    def time_it(fn, warm=8, rep=30):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(rep):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / rep * 1000

    SHAPES = [(640, 512), (60, 1024), (8, 2048), (2, 4096)]
    BLOCKS = [32, 64, 128, 256]

    for batch, n in SHAPES:
        print("\n" + "=" * 78)
        print(f"SHAPE b={batch} n={n}: summed trailing-GEMM time over all blocks")
        print("=" * 78)
        print(f"  {'B':>4}{'torch.bmm fp32':>16}{'cublasLt fp32':>15}{'cublasLt bf16x9':>17}{'bf16x9 vs torch':>17}")
        for B in BLOCKS:
            # Build the per-block operands once (random; timing only).
            blocks = []
            for k in range(0, n, B):
                b = min(B, n - k)
                m = n - k
                t = n - k - b
                if t <= 0:
                    continue
                Y = torch.randn(batch, m, b, device="cuda")
                A_trail = torch.randn(batch, m, t, device="cuda")
                W = torch.randn(batch, b, t, device="cuda")
                C = torch.empty(batch, b, t, device="cuda")
                AY = torch.empty(batch, m, t, device="cuda")
                blocks.append((Y, A_trail, W, C, AY))

            def torch_pass():
                for Y, A_trail, W, C, AY in blocks:
                    torch.bmm(Y.transpose(-1, -2), A_trail)
                    torch.bmm(Y, W)

            def lt_pass(ctype):
                def f():
                    for Y, A_trail, W, C, AY in blocks:
                        matmul_lt(Y, A_trail, True, False, ctype, C)   # Y^T @ A_trail
                        matmul_lt(Y, W, False, False, ctype, AY)       # Y @ W
                return f

            try:
                t_torch = time_it(torch_pass)
            except Exception as e:
                t_torch = float("nan"); print("   torch err", e)
            try:
                t_fp32 = time_it(lt_pass(COMPUTE_32F))
            except Exception as e:
                t_fp32 = float("nan")
            try:
                t_bf = time_it(lt_pass(COMPUTE_BF16X9))
            except Exception as e:
                t_bf = float("nan")
            ratio = t_torch / t_bf if t_bf == t_bf and t_bf > 0 else float("nan")
            print(f"  {B:>4}{t_torch:>16.3f}{t_fp32:>15.3f}{t_bf:>17.3f}{ratio:>16.2f}x")

    lt.cublasLtDestroy(handle)
    print("\nDONE")


@app.local_entrypoint()
def main():
    probe.remote()
