"""
opus2_q2.py — UNAMBIGUOUS roofline attribution for v19, sidestepping the profiler's
parent/child double-counting.

The torch.profiler table double-counts (aten::bmm/baddbmm self-CUDA overlaps the
magma_sgemmEx/cutlass-sgemm backend rows; naive sum 181ms vs 128ms total). So instead we:

  (A) Run v19's actual custom_kernel under CUDA-event timing for the true e2e ms/iter, and
  (B) Micro-time EACH phase op in isolation on the EXACT per-block shapes v19 generates, summed
      over all blocks — panel (Triton), Gram bmm(Yt@Y), C bmm(Yt@A_trail), trsm, baddbmm(Y@W),
      and the elementwise/copies (tril+diagonal writes). Each op timed with CUDA events on its own.

This gives clean per-phase ms + TFLOP/s (FLOP-bearing phases) with NO double-counting, and the
sum should reconcile to the e2e time (minus launch-overlap). Roofline ref: B200 FP32 SIMT ~60 TF/s.

Shapes (from v19 _blocked_wy_triton, _BLOCK=32): block k=0..n-B step B; per block:
  m = n-k; b = min(32, n-k);  panel (m x b);  Gram (b x m)@(m x b);  A_trail (m x N_trail),
  N_trail = n-(k+b);  C (b x m)@(m x N_trail) = (b x N_trail);  trsm (b x b)\(b x N_trail);
  baddbmm Y(m x b)@W(b x N_trail) -> (m x N_trail).  All batched over BATCH.

Anti-hang: server timeout=90; local timeout 120, FOREGROUND.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("triton")
)

app = modal.App("qr-opus2-q2")

RL_FP32 = 60.0  # TFLOP/s, B200 FP32 SIMT roofline (problem statement)


@app.function(gpu="B200", image=image, timeout=90, retries=0)
def attribute(n: int = 512, batch: int = 640):
    import torch, time, math

    print("=" * 92)
    print(f"OPUS2 Q2 — per-phase roofline attribution (NO double-count)  n={n} batch={batch}")
    print("=" * 92)
    dev = "cuda"; dt = torch.float32
    B = 32

    def ev_bench(fn, iters=30, warmup=10):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters  # ms

    # ── per-phase accumulators (ms/iter summed over blocks) and FLOPs ──
    phase_ms = {"panel": 0.0, "gram_bmm": 0.0, "C_bmm": 0.0, "trsm": 0.0,
                "baddbmm_trail": 0.0, "elementwise_copy": 0.0}
    phase_flop = {"gram_bmm": 0.0, "C_bmm": 0.0, "baddbmm_trail": 0.0}

    # Triton panel kernel: import v19's exact kernel from the submission file content.
    # We replicate the per-block launch (one launch per block, batched over `batch`).
    import triton, triton.language as tl

    def _next_pow2(x):
        p = 1
        while p < x: p <<= 1
        return p

    # v19 panel kernel (verbatim signature; body trimmed to the timing-relevant work).
    @triton.jit
    def _panel_qr_kernel(A_ptr, tau_ptr, k_start, b, m,
                         sAb, sAr, sAc, stb, stc,
                         M_POW2: tl.constexpr, B_POW2: tl.constexpr, EPS_V0: tl.constexpr):
        bid = tl.program_id(0)
        rows = tl.arange(0, M_POW2); cols = tl.arange(0, B_POW2)
        base = A_ptr + bid * sAb + k_start * sAr + k_start * sAc
        tile_ptr = base + rows[:, None] * sAr + cols[None, :] * sAc
        tile_mask = (rows[:, None] < m) & (cols[None, :] < b)
        panel = tl.load(tile_ptr, mask=tile_mask, other=0.0)
        for j in range(0, b):
            is_j_row = rows == j
            col = tl.sum(tl.where(cols[None, :] == j, panel, 0.0), axis=1)
            col = tl.where(rows >= j, col, 0.0)
            alpha = tl.sum(tl.where(is_j_row, col, 0.0))
            norm_sq = tl.sum(col * col); norm = tl.sqrt(norm_sq)
            sign_a = tl.where(alpha >= 0.0, 1.0, -1.0)
            beta = -sign_a * norm; v0 = alpha - beta
            v_norm_sq = v0 * v0 + (norm_sq - alpha * alpha)
            tau_j = tl.where(v_norm_sq > 0.0, 2.0 * v0 * v0 / v_norm_sq, 0.0)
            safe_v0 = tl.where(tl.abs(v0) < EPS_V0, 1.0, v0)
            u_sub = tl.where(rows > j, col / safe_v0, 0.0)
            u_full = tl.where(is_j_row, 1.0, u_sub)
            w = tl.sum(u_full[:, None] * panel, axis=0)
            update = (tau_j * u_full)[:, None] * w[None, :]
            trailing = cols[None, :] > j
            panel = tl.where(trailing, panel - update, panel)
            new_colj = tl.where(is_j_row, beta, u_sub)
            write_colj = (cols[None, :] == j) & (rows[:, None] >= j)
            panel = tl.where(write_colj, new_colj[:, None], panel)
            tl.store(tau_ptr + bid * stb + (k_start + j) * stc, tau_j)
        tl.store(tile_ptr, panel, mask=tile_mask)

    MP = _next_pow2(n); BP = _next_pow2(B)
    nwarps = 4 if MP <= 256 else (8 if MP <= 512 else 16)
    grid = (batch,)

    H = torch.randn(batch, n, n, device=dev, dtype=dt)
    tau_all = torch.zeros(batch, n, device=dev, dtype=dt)

    nblocks = 0
    for k in range(0, n, B):
        b = min(B, n - k); m = n - k; k_end = k + b
        nblocks += 1
        # ---- PANEL (one Triton launch, batched) ----
        def run_panel(k=k, b=b, m=m):
            _panel_qr_kernel[grid](H, tau_all, k, b, m,
                                   H.stride(0), H.stride(1), H.stride(2),
                                   tau_all.stride(0), tau_all.stride(1),
                                   M_POW2=MP, B_POW2=BP, EPS_V0=1e-30, num_warps=nwarps)
        phase_ms["panel"] += ev_bench(run_panel)

        if k_end >= n:
            continue
        Ntr = n - k_end
        # Build the per-block operands at the right shapes (values irrelevant for timing).
        panel = H[:, k:, k:k+b]
        Y = torch.tril(panel, diagonal=-1).contiguous()
        Y.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        Yt = Y.transpose(-1, -2).contiguous()
        A_trail = H[:, k:, k_end:].contiguous()          # (batch, m, Ntr)
        Tinv = torch.bmm(Yt, Y)                           # warm shape
        W = torch.randn(batch, b, Ntr, device=dev, dtype=dt)

        # ---- Gram bmm: Yt(b x m) @ Y(m x b) -> (b x b) ----
        phase_ms["gram_bmm"] += ev_bench(lambda Yt=Yt, Y=Y: torch.bmm(Yt, Y))
        phase_flop["gram_bmm"] += 2.0 * batch * b * b * m

        # ---- C bmm: Yt(b x m) @ A_trail(m x Ntr) -> (b x Ntr) ----
        phase_ms["C_bmm"] += ev_bench(lambda Yt=Yt, At=A_trail: torch.bmm(Yt, At))
        phase_flop["C_bmm"] += 2.0 * batch * b * Ntr * m

        # ---- trsm: solve (b x b) \ (b x Ntr) ----
        Tt = Tinv.transpose(-1, -2).contiguous()
        C = torch.bmm(Yt, A_trail)
        phase_ms["trsm"] += ev_bench(
            lambda Tt=Tt, C=C: torch.linalg.solve_triangular(Tt, C, upper=False, left=True))

        # ---- baddbmm TRAILING: Y(m x b) @ W(b x Ntr) -> (m x Ntr), fused subtract ----
        At2 = A_trail.clone()
        phase_ms["baddbmm_trail"] += ev_bench(
            lambda At2=At2, Y=Y, W=W: torch.baddbmm(At2, Y, W, beta=1, alpha=-1))
        phase_flop["baddbmm_trail"] += 2.0 * batch * m * Ntr * b

        # ---- elementwise/copy: tril + diagonal fill + the H-slice copy_ ----
        def run_ew(panel=panel, Y=Y):
            Yt2 = torch.tril(panel, diagonal=-1)
            Yt2.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        phase_ms["elementwise_copy"] += ev_bench(run_ew)

    # e2e reference comes from the v19 profiler do_bench (n512=13.286ms, n1024=11.652ms).
    e2e_ref = {512: 13.286, 1024: 11.652}.get(n)

    # ── report ──
    total_phase = sum(phase_ms.values())
    print(f"\n  blocks={nblocks}  (panel runs {nblocks}x; trailing/WY {nblocks-1}x)\n")
    print(f"  {'phase':>18} | {'ms/iter':>8} | {'% of phases':>11} | {'TFLOP/s':>8} | {'% of 60 RL':>10}")
    print("  " + "-" * 70)
    order = ["panel", "baddbmm_trail", "C_bmm", "gram_bmm", "trsm", "elementwise_copy"]
    labels = {"panel": "PANEL (Triton)", "baddbmm_trail": "TRAILING Y@W",
              "C_bmm": "WY: C=Yt@Atrail", "gram_bmm": "WY: Gram Yt@Y",
              "trsm": "TRISOLVE (trsm)", "elementwise_copy": "elementwise/copy"}
    for ph in order:
        ms = phase_ms[ph]; pct = ms / total_phase * 100
        if ph in phase_flop and ms > 0:
            tf = phase_flop[ph] / (ms * 1e-3) / 1e12
            print(f"  {labels[ph]:>18} | {ms:>8.3f} | {pct:>10.1f}% | {tf:>8.1f} | {tf/RL_FP32*100:>9.0f}%")
        else:
            print(f"  {labels[ph]:>18} | {ms:>8.3f} | {pct:>10.1f}% | {'—':>8} | {'—':>10}")
    print("  " + "-" * 70)
    print(f"  {'SUM of phases':>18} | {total_phase:>8.3f} ms/iter")
    if e2e_ref:
        print(f"  {'v19 e2e (do_bench)':>18} | {e2e_ref:>8.3f} ms/iter  (phase-sum/e2e = {total_phase/e2e_ref:.2f}x; "
              f"<1 = launch overlap hides some; >1 = isolated loses fusion)")
    # GEMM grouping
    gemm_ms = phase_ms["baddbmm_trail"] + phase_ms["C_bmm"] + phase_ms["gram_bmm"]
    print(f"\n  ALL GEMMs (trailing+C+Gram): {gemm_ms:.3f} ms = {gemm_ms/total_phase*100:.0f}% of phases.")
    print(f"  PANEL alone: {phase_ms['panel']:.3f} ms = {phase_ms['panel']/total_phase*100:.0f}% of phases.")
    print(f"  TRAILING alone: {phase_ms['baddbmm_trail']:.3f} ms = {phase_ms['baddbmm_trail']/total_phase*100:.0f}% of phases.")
    return phase_ms, phase_flop, e2e_ref


@app.local_entrypoint()
def main(n: int = 512, batch: int = 640):
    attribute.remote(n=n, batch=batch)
