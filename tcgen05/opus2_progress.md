# opus2_progress.md — corrected gate (BATCHED-skinny K=32) + roofline attribution

Mandate (opus2): two DECISIVE measurements, then STOP. Do NOT build a megakernel.
Write ONLY `tcgen05/opus2_*`. Anti-hang: every `modal run` = server `timeout=90` + local `timeout 120`, FOREGROUND.

Q1 — corrected gate. The B9 gate tested the WRONG shape: single fat GEMMs M×M, K∈{64,128,256},
where cublasLt-78 is a tuned tcgen05+Ozaki kernel (~9µs, unbeatable). v19's ACTUAL trailing update
(`baddbmm(A_trail, Y, W)` in `_wy_trailing_trisolve`) is a BATCHED SKINNY bmm: per-matrix
[M×K]@[K×N], **K=b=32** (panel width `_BLOCK=32`), batch∈{640 (n512), 60 (n1024)}, M up to n,
N up to ~M. Compare on the batched-K32 shape: (a) torch.bmm FP32, (b) cublasLt-78 batched if reachable,
(c) tcgen05 BF16x9 batched (generalize opus_stage1/opus_bench over batch via grid.z). Verdict: does
(c) beat (a) on batched-K32 → drop-in exact-FP32 trailing replacement Y/N?

Q2 — roofline. Profile v19 at n512 b640 + n1024 b60; attribute time + TFLOP/s to panel vs trailing
bmm vs Gram/trisolve vs copies. Where is the ~7× roofline gap (v19 n512 ~8.6 TFLOP/s vs ~60 FP32 RL)?

Reps from QR sweep (n512): first block m=512,N_trail=480; mid m=256,N_trail=224. (n1024 analogous.)

---
- 2026-06-15 03:42 — Q1 GATE RAN (clean, 17s, exit 0). opus2_bench.py: batched tcgen05 via grid.z
  (M//128, N//256, BATCH). RESULTS (µs/call): n512-first(B640,512,480): bmm 316 / cuBLAS78 5.6 /
  tcgen05 18470. n512-mid(B640,256,224): bmm 85.5 / cuBLAS78 4.5 / tcgen05 5029. n1024-first(B60,
  1024,992): bmm 123 / cuBLAS78 4.5 / tcgen05 6447. n1024-mid(B60,512,480): bmm 35 / cuBLAS78 4.5 /
  tcgen05 1561. tcgen05 rel-vs-FP64 ~6-8e-7 (CORRECT/bit-exact) but 44-58x SLOWER than torch.bmm.
  >>> tcgen05 BF16x9 batched DROP-IN = decisive NO (0/4; naive single-warp scalar-load kernel is
  load-bound, does not scale over batch — same as B9). NOT a v19 trailing replacement.
- 2026-06-15 03:42 — TWO surprises worth chasing: (1) torch.bmm FP32 is SLOW on batched-K32
  (316µs n512-first) — this IS a real v19 bottleneck. (2) cublasLt BATCHED returned ~5µs (≈56x faster
  than torch.bmm) BUT rel_lt≈2e-3 (NOT the ~3e-7 of true BF16x9; 2e-3 ≈ plain bf16). Suspect the
  batched cublasLt heuristic did NOT engage compute-type-78 / silently fell back to plain BF16.
  MUST verify precision before any Q1 "cuBLAS78 is a free win" claim (B4: trailing must be FP32).
  Next: probe cublasLt-78 batched precision (force type-78 algo; compare vs strided single-GEMM loop).
- 2026-06-15 03:43 — Q1 RESOLVED (opus2_ltprobe.py). The ~5µs cuBLAS78-batch in opus2_bench was an
  ARTIFACT: strided-batched COMPUTE_78 returns rel=NaN, wavesCount=0.00 — the matmul effectively
  NO-OPS (emulated-78 doesn't support strided-batch on this K=32 layout). LT32-batch heuristic FAILS
  rc=15 (NOT_SUPPORTED). LT78-LOOP (per-matrix, the proven path) gives rel=2.95e-3 (bf16 floor, NOT
  3e-7 — type-78 doesn't fully engage at K=32) AND 2675µs (CPU-loop-bound over 640). torch.bmm FP32:
  rel=2.58e-7 (exact), 316µs. >>> Q1 FINAL: NO method beats torch.bmm FP32 on batched-K32. tcgen05
  44-58x slower; cuBLAS78 batched broken (nan), loop slower+imprecise. torch.bmm (what v19 uses) is
  BOTH fastest-exact AND incumbent. No drop-in trailing win exists on the batched-K32 shape.
- 2026-06-15 03:43 — Note for Q2: torch.bmm batched-K32 at 316µs (n512-first) is genuinely slow for
  the FLOPs (2*640*512*480*32 = 10.1 GFLOP / 316µs = 32 TFLOP/s — half FP32 RL; K=32 is too skinny to
  saturate). This is the trailing's contribution to the ~7× e2e roofline gap. Now Q2: profile v19.
- 2026-06-15 03:44 — Q2 profiler ran (modal_qr --mode profile v19, clean exit 0). do_bench: n512
  13.286ms, n1024 11.652ms. BUT torch.profiler double-counts (aten::bmm/baddbmm self-CUDA overlaps
  magma_sgemmEx/cutlass-sgemm backend rows; naive sum 181ms vs 128ms total). Built opus2_q2.py to
  micro-time each phase op in ISOLATION on v19's exact per-block shapes (CUDA events), summed over
  blocks — no double-count.
- 2026-06-15 03:47 — Q2 RESOLVED (opus2_q2.py; phase-sum reconciles to e2e: n512 0.99x, n1024 0.93x).
  CLEAN per-phase breakdown:
    n512 b640 (13.21ms): PANEL 5.34ms 40.4% (Triton, reduction-bound, no FLOP roofline) |
      TRAILING Y@W 4.08ms 30.9% @14.0 TFLOP/s = 23% RL | C-bmm(Yt@Atrail) 1.40ms 10.6% @40.9TF=68%RL |
      TRISOLVE 1.77ms 13.4% | Gram 0.26ms 2.0% @21.6TF | elementwise/copy 0.37ms 2.8%.
    n1024 b60 (10.84ms): PANEL 4.78ms 44.1% | TRAILING 3.08ms 28.4% @13.9TF=23%RL |
      C-bmm 1.25ms 11.5% @34.3TF=57%RL | TRISOLVE 1.06ms 9.7% | Gram 0.40ms @5.2TF | ew/copy 0.28ms 2.5%.
  e2e useful QR rate (4/3 n^3 batch): n512 8.6 TFLOP/s (7.0x below 60 RL), n1024 7.4 TFLOP/s (8.1x below).
  >>> WHERE THE 7x GAP LIVES: it is SPLIT, not one culprit. ~40-44% PANEL (sequential Householder,
  inherently low-FLOP/reduction-bound — expected, NOT a GEMM-roofline problem) + ~28-31% TRAILING Y@W
  running at only 23% of roofline BECAUSE K=B=32 is too skinny to saturate tensor/SIMT units. The
  C-bmm (wide N) already runs at 57-68% RL — it is NOT the problem. So the recoverable GEMM bottleneck
  = the skinny-K=32 TRAILING update (the ONE phase Q1 just proved nothing beats torch.bmm on, while
  it's stuck at 23% RL); the other ~40% is the sequential panel, which needs an algorithmic change
  (in-kernel tensor-core panel / wider B), NOT a faster trailing GEMM.

---
## FINAL — opus2 two-question verdict

Q1 (corrected gate, batched-skinny K=32 — the shape B9 skipped): tcgen05 BF16x9 batched = 44-58x
SLOWER than torch.bmm (naive single-warp scalar-load kernel, load-bound, doesn't scale over batch).
cublasLt-78 STRIDED-BATCHED is non-functional on K=32 (rel=NaN, wavesCount=0). cublasLt-78 per-matrix
LOOP is both slower (2675µs vs bmm 316µs, CPU-loop-bound) AND only bf16-precise at K=32 (rel 2.95e-3,
fails B4 FP32 gate). torch.bmm FP32 (rel 2.58e-7) is the ONLY confirmed-exact option AND the fastest
AND what v19 already uses. >>> NO drop-in exact-FP32 trailing replacement exists on batched-K32. The
v22-style "BF16x9 cublasLt beats torch on FAT trailing" win does NOT carry to v19's batched-K32 shape.

Q2 (roofline attribution): the ~7x e2e gap (n512 8.6 TFLOP/s) is SPLIT between the sequential PANEL
(~40-44%, expected, low-FLOP) and the TRAILING Y@W GEMM (~28-31% but only 23% of roofline, K=32-skinny-
bound). Trailing is the dominant RECOVERABLE GEMM bottleneck — but Q1 shows it can't be sped up by a
better GEMM library/kernel at K=32. The only levers left for 2.5ms: (1) raise the panel width B above
32 so the trailing GEMM fattens past the skinny wall (blocked by 228KB smem residence — findings B7
DEAD), or (2) a fundamentally different panel (in-kernel tensor-core, escape sequential Householder).
Both are big rewrites; neither is a drop-in. Confirms B9/C5: 2.5ms needs a fundamentally different
approach, not incremental GEMM swaps.
