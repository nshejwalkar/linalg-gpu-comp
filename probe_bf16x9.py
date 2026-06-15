"""
Probe whether BF16x9 (Ozaki / CUBLAS_COMPUTE_32F_EMULATED_16BFX9) FP32-emulation is
reachable for torch.bmm on B200 + torch 2.12/cu130, and whether it's exact + faster.
Run: conda activate modal && modal run probe_bf16x9.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("bf16x9-probe", image=image)


@app.function(gpu="B200", timeout=600)
def probe():
    import os, time, torch

    mm = torch.backends.cuda.matmul
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
    print("matmul attrs:", [a for a in dir(mm) if not a.startswith("_")])
    print("fp32_precision now:", getattr(mm, "fp32_precision", "N/A"))
    print("env CUBLAS* :", {k: v for k, v in os.environ.items() if "CUBLAS" in k})

    # Representative trailing-update GEMM shape (n512 b640 panel: Y^T @ A_trail).
    B, K, M, N = 640, 512, 64, 448
    a = torch.randn(B, K, M, device="cuda")
    b = torch.randn(B, K, N, device="cuda")
    ref = torch.bmm(a.transpose(-1, -2).double(), b.double())  # FP64 reference
    refmax = ref.abs().max().item()

    def bench(label):
        for _ in range(5):
            torch.bmm(a.transpose(-1, -2), b)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            out = torch.bmm(a.transpose(-1, -2), b)
        torch.cuda.synchronize()
        ms = (time.time() - t) / 50 * 1000
        rel = (out.double() - ref).abs().max().item() / refmax
        print(f"  {label:<34} {ms:7.3f} ms   rel_err={rel:.2e}")

    mm.allow_tf32 = False
    try:
        mm.fp32_precision = "ieee"
    except Exception:
        pass
    bench("FP32 (ieee) baseline")

    mm.allow_tf32 = True
    bench("TF32")
    mm.allow_tf32 = False

    # Try torch-exposed fp32_precision emulation options (names vary by version).
    for opt in ["bf16x9", "bf16x6", "bf16x3", "16bfx9", "tf32x3", "bf16"]:
        try:
            mm.fp32_precision = opt
            bench(f"fp32_precision={opt}")
        except Exception as e:
            print(f"  fp32_precision={opt:<10} -> not supported ({type(e).__name__})")
    try:
        mm.fp32_precision = "ieee"
    except Exception:
        pass

    # Try cuBLAS emulation env-var route (must be set before first cuBLAS call ideally;
    # we test post-hoc to at least see if it's accepted).
    print("  (env-var emulation route would need setting CUBLAS_EMULATION_STRATEGY / "
          "math-mode before cuBLAS init; noting attrs above for the real integration)")


@app.local_entrypoint()
def main():
    probe.remote()
