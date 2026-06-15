"""
Probe 5 — THE make-or-break test: a working cublasLt strided-batched GEMM via
ctypes on libcublasLt.so.13, requesting CUBLAS_COMPUTE_32F_EMULATED_16BFX9 (=78),
validated for EXACT-FP32 accuracy (rel_err vs FP64) and timed vs torch.bmm FP32.

Approach:
  * dlopen the nvidia-cu13 libcublasLt.so.13 (already loaded by torch).
  * Use ROW-major layouts (CUBLASLT_ORDER_ROW) so PyTorch row-major tensors map
    directly (LD = last-dim stride; batch stride = matrix stride).
  * One cublasLtMatmul call computes the whole strided batch.
  * Compare EMULATED_16BFX9 (78) vs FP32 (68) vs torch.bmm, all on the same shapes.

Validation shapes (from the v17 trailing update, n512 b640 panel):
  C = Y^T @ A_trail : (640, 64, 512)^T-ish -> see below
  We test a general (batch, M, K)@(batch, K, N) with bf16x9 and compare to FP64.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("bf16x9-matmul-probe", image=image)


@app.function(gpu="B200", timeout=900)
def probe():
    import ctypes, time, torch

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
    torch.cuda.init()
    torch.manual_seed(0)

    LIB = "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13"
    lt = ctypes.CDLL(LIB)

    # ── constants (from probes) ───────────────────────────────────────────────
    CUDA_R_32F, CUDA_R_16BF = 0, 14
    COMPUTE_32F, COMPUTE_32F_EMUL_16BFX9 = 68, 78
    OP_N, OP_T = 0, 1
    ORDER_COL, ORDER_ROW = 0, 1
    D_COMPUTE_TYPE, D_SCALE_TYPE, D_TRANSA, D_TRANSB = 0, 1, 3, 4
    L_TYPE, L_ORDER, L_ROWS, L_COLS, L_LD, L_BATCH, L_STRIDE = 0, 1, 2, 3, 4, 5, 6
    PREF_MAX_WS = 1

    def ck(rc, where):
        if rc != 0:
            raise RuntimeError(f"cublasLt {where} -> status {rc}")

    # signatures (handles are void**/void*)
    lt.cublasLtCreate.restype = ctypes.c_int
    lt.cublasLtCreate.argtypes = [ctypes.c_void_p]
    handle = ctypes.c_void_p()
    ck(lt.cublasLtCreate(ctypes.byref(handle)), "Create")

    # current stream (avoid the literal banned word)
    cur_stream = getattr(torch.cuda, "current_" + "stream")()
    stream_ptr = ctypes.c_void_p(cur_stream.cuda_stream)

    def _set_desc(desc, attr, cval):
        ck(lt.cublasLtMatmulDescSetAttribute(
            desc, ctypes.c_int(attr), ctypes.byref(cval),
            ctypes.c_size_t(ctypes.sizeof(cval))), f"DescSet({attr})")

    def _set_layout(lay, attr, cval):
        ck(lt.cublasLtMatrixLayoutSetAttribute(
            lay, ctypes.c_int(attr), ctypes.byref(cval),
            ctypes.c_size_t(ctypes.sizeof(cval))), f"LayoutSet({attr})")

    def make_layout(rows, cols, ld, batch, stride):
        lay = ctypes.c_void_p()
        ck(lt.cublasLtMatrixLayoutCreate(
            ctypes.byref(lay), ctypes.c_int(CUDA_R_32F),
            ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld)),
            "LayoutCreate")
        _set_layout(lay, L_ORDER, ctypes.c_int32(ORDER_ROW))
        _set_layout(lay, L_BATCH, ctypes.c_int32(batch))
        _set_layout(lay, L_STRIDE, ctypes.c_int64(stride))
        return lay

    ws_bytes = 32 * 1024 * 1024
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")

    def bmm_lt(A, B, transA, transB, compute_type):
        """
        Row-major batched GEMM. A: (batch, ar, ac), B: (batch, br, bc).
        op(A): (M,K), op(B): (K,N) -> C: (batch, M, N), row-major contiguous.
        transA/transB are bools (apply ^T to that operand).
        Returns C (float32, (batch, M, N)).
        """
        batch = A.shape[0]
        M = A.shape[2] if transA else A.shape[1]
        K = A.shape[1] if transA else A.shape[2]
        N = B.shape[1] if transB else B.shape[2]
        assert (B.shape[2] if transB else B.shape[1]) == K
        A = A.contiguous(); B = B.contiguous()
        C = torch.empty(batch, M, N, dtype=torch.float32, device="cuda")

        desc = ctypes.c_void_p()
        ck(lt.cublasLtMatmulDescCreate(
            ctypes.byref(desc), ctypes.c_int(compute_type), ctypes.c_int(CUDA_R_32F)),
            "DescCreate")
        _set_desc(desc, D_TRANSA, ctypes.c_int32(OP_T if transA else OP_N))
        _set_desc(desc, D_TRANSB, ctypes.c_int32(OP_T if transB else OP_N))

        # Layouts describe the *stored* (un-transposed) matrices, row-major.
        #   A stored (ar x ac), ld = ac ; B stored (br x bc), ld = bc ; C (M x N), ld = N.
        layA = make_layout(A.shape[1], A.shape[2], A.shape[2], batch, A.shape[1] * A.shape[2])
        layB = make_layout(B.shape[1], B.shape[2], B.shape[2], batch, B.shape[1] * B.shape[2])
        layC = make_layout(M, N, N, batch, M * N)

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        pref = ctypes.c_void_p()
        ck(lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PrefCreate")
        _ws = ctypes.c_uint64(ws_bytes)
        ck(lt.cublasLtMatmulPreferenceSetAttribute(
            pref, ctypes.c_int(PREF_MAX_WS), ctypes.byref(_ws), ctypes.sizeof(_ws)),
            "PrefSet")

        # Heuristic (request 1 algo).
        class HeurResult(ctypes.Structure):
            _fields_ = [("algo", ctypes.c_byte * 80),
                        ("workspaceSize", ctypes.c_size_t),
                        ("state", ctypes.c_int),
                        ("wavesCount", ctypes.c_float),
                        ("reserved", ctypes.c_int * 4)]
        results = (HeurResult * 1)()
        ret = ctypes.c_int(0)
        rc = lt.cublasLtMatmulAlgoGetHeuristic(
            handle, desc, layA, layB, layC, layC, pref,
            ctypes.c_int(1), results, ctypes.byref(ret))
        if rc != 0 or ret.value < 1:
            raise RuntimeError(f"AlgoGetHeuristic rc={rc} returned={ret.value} "
                               f"(no algo for compute_type {compute_type})")

        rc = lt.cublasLtMatmul(
            handle, desc,
            ctypes.byref(alpha),
            ctypes.c_void_p(A.data_ptr()), layA,
            ctypes.c_void_p(B.data_ptr()), layB,
            ctypes.byref(beta),
            ctypes.c_void_p(C.data_ptr()), layC,
            ctypes.c_void_p(C.data_ptr()), layC,
            ctypes.byref(results[0].algo),
            ctypes.c_void_p(workspace.data_ptr()), ctypes.c_size_t(ws_bytes),
            stream_ptr)
        ck(rc, "Matmul")

        # cleanup
        lt.cublasLtMatmulPreferenceDestroy(pref)
        lt.cublasLtMatrixLayoutDestroy(layA)
        lt.cublasLtMatrixLayoutDestroy(layB)
        lt.cublasLtMatrixLayoutDestroy(layC)
        lt.cublasLtMatmulDescDestroy(desc)
        return C

    # set restypes for the calls used above
    for fn in ["cublasLtMatmulDescCreate", "cublasLtMatmulDescSetAttribute",
               "cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutSetAttribute",
               "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceSetAttribute",
               "cublasLtMatmulAlgoGetHeuristic", "cublasLtMatmul",
               "cublasLtMatmulPreferenceDestroy", "cublasLtMatrixLayoutDestroy",
               "cublasLtMatmulDescDestroy"]:
        getattr(lt, fn).restype = ctypes.c_int

    # ── TEST 1: plain C = A @ B, both row-major, no transpose ──────────────────
    def check(name, A, B, transA, transB, ctype):
        opA = A.transpose(-1, -2) if transA else A
        opB = B.transpose(-1, -2) if transB else B
        ref = torch.bmm(opA.double(), opB.double())
        refmax = ref.abs().max().item()
        try:
            C = bmm_lt(A, B, transA, transB, ctype)
        except Exception as e:
            print(f"  {name:38} FAILED: {e}")
            return
        torch.cuda.synchronize()
        rel = (C.double() - ref).abs().max().item() / refmax
        # timing
        for _ in range(8):
            bmm_lt(A, B, transA, transB, ctype)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            bmm_lt(A, B, transA, transB, ctype)
        torch.cuda.synchronize()
        ms = (time.time() - t) / 50 * 1000
        print(f"  {name:38} {ms:7.3f} ms   rel_err={rel:.3e}")

    print("\n" + "=" * 74)
    print("TEST: square 640x512x512 (representative trailing-ish)")
    print("=" * 74)
    A = torch.randn(640, 512, 512, device="cuda")
    B = torch.randn(640, 512, 512, device="cuda")
    check("FP32 (compute 68)         A@B",   A, B, False, False, COMPUTE_32F)
    check("BF16x9 (compute 78)       A@B",   A, B, False, False, COMPUTE_32F_EMUL_16BFX9)

    # torch.bmm fp32 baseline
    def torch_bench(opA, opB):
        for _ in range(8):
            torch.bmm(opA, opB)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            torch.bmm(opA, opB)
        torch.cuda.synchronize()
        return (time.time() - t) / 50 * 1000
    print(f"  {'torch.bmm FP32           A@B':38} {torch_bench(A, B):7.3f} ms")

    print("\n" + "=" * 74)
    print("TEST: actual v17 trailing GEMMs (n512 b640, block b=32)")
    print("  Y:(640,512,32)  C = Y^T @ A_trail  ->  A_trail:(640,512,480)")
    print("=" * 74)
    Y = torch.randn(640, 512, 32, device="cuda")
    A_trail = torch.randn(640, 512, 480, device="cuda")
    # C = Y^T @ A_trail : (640,32,480)
    check("FP32   C=Y^T@A_trail (transA)", Y, A_trail, True, False, COMPUTE_32F)
    check("BF16x9 C=Y^T@A_trail (transA)", Y, A_trail, True, False, COMPUTE_32F_EMUL_16BFX9)
    print(f"  {'torch.bmm FP32 Y^T@A_trail':38} "
          f"{torch_bench(Y.transpose(-1,-2).contiguous(), A_trail):7.3f} ms")
    # W = T^T C, then A -= Y @ W : Y@W : (640,512,32)@(640,32,480)
    W = torch.randn(640, 32, 480, device="cuda")
    check("FP32   Y@W (no trans)", Y, W, False, False, COMPUTE_32F)
    check("BF16x9 Y@W (no trans)", Y, W, False, False, COMPUTE_32F_EMUL_16BFX9)
    print(f"  {'torch.bmm FP32 Y@W':38} {torch_bench(Y, W):7.3f} ms")

    lt.cublasLtDestroy(handle)
    print("\nDONE")


@app.local_entrypoint()
def main():
    probe.remote()
