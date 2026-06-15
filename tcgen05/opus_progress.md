# opus_progress.md — Stage 1 tcgen05 BF16x9 GEMM (opus agent)

Mandate: build standalone tcgen05 BF16x9 GEMM, get it correct, STOP at perf gate w/ go/no-go.
Write ONLY to `opus_*` files. Anti-hang: every `modal run` wrapped in local `timeout 120` + server `timeout=90`.

Baseline to beat (findings B6/E3, v18/v22): exact-FP32 BF16x9 cublasLt type-78. Wins on FAT trailing
(K=B=256, ~2-2.6x isolated) but LOSES on skinny (K=B=32). Stage-1 thesis: TMEM accumulator un-skinnies
the trailing GEMM at m∈{512,1024}, K=B∈{64,128,256}.

---
- 2026-06-15 02:27 — Took over from sonnet. Real project is C:\Users\Neel\modal\qr_competition (NOT Essays cwd). Sonnet now at stage1_v15.py (past v11). Current blocker (v14→v15): `cute.gemm op invalid layout of A/B/D` — the SMEM-descriptor/partition layout the MMA atom expects is wrong. Sonnet hand-rolled affine layouts + make_umma_smem_desc; verifier rejects them.
- 2026-06-15 02:27 — Decision: stop reverse-engineering the layout verifier. Use the OFFICIAL `make_trivial_tiled_mma(a_dtype,b_dtype,a_leading,b_leading,acc_dtype,cta_group,mma_tiler_mn)` + partition path (probe mma_probe_out.txt revealed this new-API signature). Plan: dump the shipped CUTLASS Blackwell dense GEMM example from the wheel for the exact known-good SMEM-layout + partition recipe, then port into opus_stage1.py.
- 2026-06-15 02:30 — Dumped canonical recipe from wheel (opus_probe_example/recipe/atom). KEY: wheel ships NO examples, but ships the full helper API. Correct flow: (1) `make_trivial_tiled_mma(BF16,BF16,OperandMajorMode.K,OperandMajorMode.K,FP32,CtaGroup.ONE,(128,256))` [NEW 8-arg API]; (2) `make_smem_layout_a/b(tiled_mma,(M,N,K),BF16,num_stages)` → swizzled+staged SMEM layout (rank-3, last=stage); (3) alloc sA/sB with those; (4) `thr=tiled_mma.get_slice(0)`; (5) `tCrA=thr.partition_A(sA_stage0)`, `tCrB=thr.partition_B(sB_stage0)`; (6) TMEM acc via `make_fragment_C(partition_shape_C((M,N)))`; (7) `cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)`. This is the standard Blackwell CuTe-DSL pattern — replaces sonnet's hand-rolled make_umma_smem_desc. Next: write self-contained opus_stage1.py smoke.
- 2026-06-15 02:31 — BREAKTHROUGH (opus_probe_tmem, build-only/no-hang): canonical recipe COMPILES. make_smem_layout_a/b give `S<1,4,3> o ((M,K),1,1,stage)` swizzled layouts; slice stage via `sA[None,None,None,0]` (rank-4). thr_mma.partition_A/B → SMEM-desc fragments `((128,16),16,1,1)`. make_fragment_C(partition_shape_C) → TMEM tensor `ptr<f32,tmem> o ((128,256),1,1):((65536,1),0,0)`. alloc_tmem(num_columns, smem_ptr, is_two_cta=None, arch='sm_100'). retrieve_tmem_ptr(dtype, alignment, smem_ptr). This is the full known-good path — sonnet's hand-rolled make_umma_smem_desc was the wrong abstraction level. Next: write opus_stage1.py full smoke with real TMEM alloc + cute.gemm + TMEM->gmem epilogue.
- 2026-06-15 02:32 — opus_probe_accum: MMA atom has `.set(Field.ACCUMULATE, bool)` (mma.py:531). Standard K-loop idiom: pass0 set ACCUMULATE=False (C=A*B), passes1-8 set True (C+=A*B). For BF16x9: 3x3 split x=x0+x1+x2 exact => x*y=Σ x_i*y_j is a PLAIN SUM of 9 products (no scaling) => accumulate all 9 in ONE TMEM accumulator, single epilogue. Writing opus_stage1.py now.
- 2026-06-15 02:34 — smoke compile err1: `cute.gemm doesn't support composed layout for A/B/D`. Fix attempt: wrap partition_A/B in make_fragment_A/B. err2: make_fragment_A: `Expected affine layout, got composed S<1,4,3>... Please use recast_ptr(ptr, S<1,4,3>, element_type) to move swizzle to the ptr`.
- 2026-06-15 02:35 — opus_probe_recast (build-only) SOLVED it: pattern = (1) allocate composed-layout SMEM tensor (reserves bytes); (2) `pA = cute.recast_ptr(sA.iterator, sA_layout.inner, BFloat16)` [swizzle->ptr]; (3) `sA_affine = cute.make_tensor(pA, sA_layout.outer)` [affine layout]; (4) partition_A(sA_affine_stage0) -> make_fragment_A => proper smem_desc operand. COMPILE OK. Loads go through sA_affine (swizzle in ptr => correct physical addressing matching the descriptor). Applying to opus_stage1.py.
- 2026-06-15 02:39 — More gemm errors: passing partition_A output (memref) -> "expects smem_desc_view"; passing make_fragment_A(partition_A) -> smem_desc_view but shape (1,2,1,1) -> "invalid layout". Fetched GROUND TRUTH: CUTLASS examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py (gh trees API). CANONICAL recipe: `tCrA = tiled_mma.make_fragment_A(sA)` DIRECTLY on the staged SMEM tensor (NOT partition_A — that's the GMEM/TMA path) -> (MMA,MMA_M,MMA_K,STAGE). acc: `acc_shape=partition_shape_C(mma_tiler[:2]); tCtAcc_fake=make_fragment_C(acc_shape); tCtAcc=make_tensor(tmem_ptr, tCtAcc_fake.layout)`. K-loop: `cute.gemm(tiled_mma, tCtAcc, tCrA[(None,None,kblk,stage)], tCrB[...], tCtAcc); tiled_mma.set(ACCUMULATE,True)` after blk0. Applied to opus_stage1.py (MMA_K=1 so slice (None,None,0,0)). Running smoke.
- 2026-06-15 02:44 — Fixed: dealloc_tmem wants the TMEM ptr (tmem_ptr) not the smem slot. THEN: smoke COMPILED + LAUNCHED (no hang — clean recover). Runtime err: "Misaligned Address" SM warp exception (cudaErrorMisalignedAddress). Kernel runs but a memory access is misaligned — likely the manual element SMEM load through the swizzled 2D view sA_ld/sB_ld (S<1,4,3> swizzle granularity), or the Ld32x32b TMEM copy. This is a tractable runtime bug (vs compile). Next: isolate — try loading via composed sA/sB tensor (nested coord) which matches MMA's expected swizzled addressing.
- 2026-06-15 02:47 — Isolated misalign: opus_probe_load proved scalar swizzled SMEM stores round-trip with ZERO error (load path fine). opus_probe_mma: MMA-only kernel (load+gemm+commit+mbar, dummy gC write, NO tmem read) MISALIGNS => bug is in the MMA/descriptor path, NOT the load or epilogue. Hypothesis: make_fragment_A needs swizzle in the LAYOUT (composed sA), not in the ptr. Reference passes composed `sA` to tiled_mma.make_fragment_A (op=None skips _verify_fragment_A); my recast-affine sA_aff produces a descriptor with wrong swizzle => MMA reads misaligned. Fix: pass composed sA/sB to make_fragment_A/B; keep affine views only for scalar loads.
- 2026-06-15 02:55 — Ruled out: relinquish timing (still misaligns), tmem ncols 256 (correct). Plain non-swizzled SMEM rejected: make_fragment_A needs >=3 modes (MMA,MMA_MN,MMA_K) so make_smem_layout_a's staged structure is MANDATORY; K=16 bf16 K-major forces K_SW32 swizzle (256-bit). CRUX HYPOTHESIS: scalar stores through the recast-SWIZZLED ptr place bytes per recast_ptr's swizzle interpretation, which may DIFFER from how the UMMA hardware descriptor swizzles => MMA reads misaligned. opus_probe_load only proved write/read self-consistency, NOT consistency with the descriptor. Reference avoids this via TMA (writes in descriptor-expected swizzle). Fix to try: write SMEM through the COMPOSED sA tensor (swizzle applied by the CuTe layout, descriptor-consistent), or use a tiled cute.copy into the composed layout.
- 2026-06-15 02:59 — Reference allocation style (layout=outer, swizzle=inner separate, byte_align=128) + make_warp_uniform: STILL misaligns (GPC1,TPC5,SM0 consistently = single MMA-issue thread). Fetched fp16_gemm_0.py (minimal Blackwell tcgen05 tutorial): ALL examples use TMA for SMEM loads. KEY DIFFERENCE found: TMEM alloc uses `utils.TmemAllocator(holding_buf_ptr, barrier_for_retrieve=NamedBarrier(id=1))` + `tmem.allocate(512)` [ALL 512 cols] + `tmem.wait_for_alloc()` + `tmem.retrieve_ptr()`. My raw arch.alloc_tmem(256)+sync_threads likely doesn't properly establish TMEM alloc => MMA writes misaligned TMEM addr => fault. Switching to TmemAllocator + 512 cols.
- 2026-06-15 03:01 — TmemAllocator FIXED THE MISALIGN. opus_probe_tmemalloc: compiles, NO misaligned-address fault anymore. But now HANGS (90s server self-kill, clean). Hang is at the raw mbarrier path (tc.commit(mbar)+mbarrier_wait by warp0/lane0) — likely wrong arrive-count/phase for the tcgen05 MMA completion signal. Reference uses PipelineUmmaAsync (not raw mbarrier) to await the MMA. Root TMEM-addr bug solved; remaining = MMA-completion sync. Next: remove mbarrier, test MMA completes (dummy write, no tmem read), then wire correct completion (PipelineUmmaAsync or proper mbarrier arrive/expect).
- 2026-06-15 03:08 — ⭐ SMOKE PASS! Single tcgen05 MMA 128x256x16 BF16->FP32: max abs err=0.0, rel=9.3e-08, C == A.float()@B.float(). GOAL 1 DONE. Fixes that mattered: (1) PipelineUmmaAsync acquire/commit must be in ONE warp-0 scope; (2) gC view rank-3 ((M,N),1,1) to match acc; (3) epilogue = ref pattern: zipped_divide into SUBTILE=4 epi-tiles, Ld32x32b.x64, rmem fragment sized to GMEM-dst per-thread shape (make_rmem_tensor(tDgC[...,0].shape)), loop over tiles. Running BF16x9 (NPASS=9) next.
- 2026-06-15 03:08 — ⭐⭐ BF16x9 PASS! 9-pass Ozaki, single TMEM accumulator: err vs FP64 = 9.19e-6, rel = 4.38e-07 (matches v18 target 3e-7, well under 1e-5 gate); err vs torch FP32 = 9.54e-6. GOAL 2 DONE — bit-exact FP32 from bf16 tensor cores via TMEM accumulation. Now GOAL 3 = PERF GATE. Need: generalize kernel to M-tiled grid, N (MMA N<=256), K=B(64/128/256) via BF16x9; benchmark tcgen05-BF16x9 vs cublasLt type-78 (v18) vs torch FP32 on m in {512,1024}, N~m, K in {64,128,256}. Reading v18 baseline next.
- 2026-06-15 03:12 — Built opus_bench.py: generalized M/N-tiled tcgen05 BF16x9 GEMM (grid=(M/128,N/256), NPASS=9*K/16, per-tile loads offset by block idx) + perf gate vs cublasLt type-78 (ctypes, COMPUTE 78) + torch FP32, on M in {512,1024} x K in {64,128,256}. Running (compiles 6 kernels; server timeout=120, local 200).
- 2026-06-15 03:13 — Perf gate CRASHED on first shape (M=512 grid=(4,2)): GPU-level fault kills the worker before Python prints traceback; Modal retries 8x (input-level, not fn retries). Single-tile kernel was correct, so bug is in the multi-block generalization — suspect the manual `gC.iterator + (m_off*Ncols+n_off)` pointer arithmetic or A/B global indexing. Isolating with a single-shape correctness probe (grid 4,1, cleaner GMEM tiling via local_tile/coordinate slice) before re-running the gate.
- 2026-06-15 03:15 — opus_probe_tile: local_tile-based tiling PASSES at M=512,N=256,K=64,grid(4,1),NPASS=36, rel-vs-FP64=1.44e-6. Confirms multi-block works + the opus_bench bug was the manual `gC.iterator+offset` arithmetic. Fix: use cute.local_tile for gA/gB/gC tiling in opus_bench. Rewriting opus_bench kernel with local_tile, then re-run gate (also covers N=512 grid bidy>0).
- 2026-06-15 03:19 — Perf gate crashes on FIRST shape even with cublasLt DISABLED => bug is tcgen05 multi-block at grid (4,2) i.e. N=512 (grid_n=2, bidy=1 block). Single-shape probe was grid (4,1). Isolating grid(4,2)/N=512 in opus_probe_tile. Suspects: bidy>0 N-tiling (local_tile mB/mC col offset) OR TMEM co-residency (8 blocks each grab all 512 cols; if 2 share an SM the 2nd alloc faults).
- 2026-06-15 03:24 — ⭐ PERF GATE RUNS. Bug was Constexpr grid passing; grid-from-shape fixed it. RESULTS (tcgen05 BF16x9 vs torch FP32, us): (512,512,64) 745 vs 5.5; (512,512,128) 1490 vs 6.4; (512,512,256) 2972 vs 9.0; (1024,1024,64) 749 vs 8.4; (1024,1024,128) 1492 vs 12.2; (1024,1024,256) 2982 vs 17.0. tcgen05 is 100-350x SLOWER. All rel ~1-4e-6 (correct). Time scales linearly with NPASS=9K/16 => dominated by SERIAL scalar SMEM loads + single-warp MMA issue (no TMA, no pipeline, 1 MMA-warp). Naive kernel = clear NO-GO as-is. Getting cublasLt-78 numbers next for the full table, then GATE VERDICT.

---
## FINAL SUMMARY — Stage 1 tcgen05 BF16x9 GEMM (opus)

### What was built (all in opus_* files; sonnet's stage1_v*/probe_* untouched)
- `opus_stage1.py` — standalone tcgen05 BF16x9 GEMM. SMOKE (1 MMA 128x256x16) + BF16x9 (9-pass).
- `opus_bench.py` — generalized M/N-tiled GEMM + perf gate (tcgen05 vs cublasLt-78 vs torch FP32).
- `opus_probe_*.py` — the de-risk probes that pinned the API/recipe.

### GOAL 1 (smoke) — ✅ PASS. Single tcgen05 MMA, rel vs A.float()@B.float() = 9.3e-08.
### GOAL 2 (BF16x9 accuracy) — ✅ PASS. 9-pass Ozaki, single TMEM accumulator:
   rel vs FP64 = 4.38e-07 (matches v18 cublasLt-78's ~3e-7; well under 1e-5 gate). Bit-exact FP32 achieved.

### GOAL 3 PERF GATE (B200) — tcgen05 BF16x9 vs cublasLt type-78 vs torch FP32 (us/call):
   shape (M,N,K)    | tcgen05 | cublasLt78 | torchFP32
   (512,512,64)     |  740.3  |    10.2    |   11.7
   (512,512,128)    | 1472.9  |     9.1    |   11.0
   (512,512,256)    | 2937.7  |     8.7    |    9.8
   (1024,1024,64)   |  745.7  |     8.8    |   10.5
   (1024,1024,128)  | 1481.9  |     9.0    |   12.2
   (1024,1024,256)  | 2955.3  |     8.8    |   17.0
   tcgen05 BF16x9 beat both baselines on 0/6 shapes. LOSES by ~70-330x.

### ⛔ GATE VERDICT: NO-GO — do NOT proceed to Stage 2 (as currently scoped).
Reasons (well-grounded, not just "my kernel is slow"):
1. cublasLt type-78 (CUBLAS_COMPUTE_32F_EMULATED_16BFX9) IS ITSELF a fully-tuned tcgen05+Ozaki
   kernel and clears the trailing GEMM in ~9 us flat across all 6 shapes. The Stage-1 thesis
   ("TMEM accumulator un-skinnies the trailing GEMM") assumed the rival was a SKINNY BATCHED bmm
   (B6's B=32 wall). But as a single contiguous-K GEMM, cuBLAS-78 already handles K=B in {64,128,256}
   excellently — there is no skinniness left for TMEM to fix at the isolated-GEMM level.
2. To even TIE cuBLAS-78 (~9 us), a hand-rolled tcgen05 BF16x9 must match NVIDIA's tuned library
   kernel (TMA + warp-spec + deep pipelining). That leaves zero headroom for the QR-fusion win that
   was the entire point — and these shapes are already near the ~8-10 us kernel-launch floor, so
   there's little absolute time to claw back.
3. My kernel's time scales PERFECTLY linearly with NPASS=9K/16 (740/1473/2938 us ≈ 1:2:4) => it is
   100% bound by serial single-warp scalar SMEM loads + 1-warp MMA issue (no TMA/pipeline/multi-warp).
   A full rewrite (TMA + PipelineTmaUmma + warp-spec, ~the dense_gemm.py persistent kernel) is the
   ONLY way to be competitive — i.e. Stage 2 effort just to MATCH a library call we already have (v18/v22).

### Caveat / where a GO could still live (for the parent's judgment)
The gate measured the ISOLATED trailing GEMM, where cuBLAS-78 is unbeatable. The ORIGINAL megakernel
thesis is about FUSION: keeping the active panel TMEM-resident and folding panel+trailing+subtract
into ONE kernel to kill launch overhead and HBM round-trips between the ~B steps of blocked QR. That
end-to-end win is NOT captured by this isolated-GEMM gate and is NOT refuted by it — but it is a much
bigger, riskier build (full warp-specialized megakernel) and cannot lean on "tcgen05 beats cuBLAS on
the trailing GEMM" as its justification, because (per this gate) it does not. Recommend: BANK the
working v19/v24 + v22 BF16x9-cublasLt large-n path; only pursue the TMEM megakernel if the parent
wants to bet on the fusion/launch-overhead win directly, eyes open that the per-GEMM lever is dead.

### Reusable assets delivered
- A CORRECT, bit-exact (4e-7) standalone tcgen05 BF16x9 GEMM in CuTe-DSL on B200 — the hard
  CuTe-DSL recipe is now fully pinned & working (TmemAllocator + PipelineUmmaAsync + make_smem_layout_a/b
  + recast/affine + make_fragment_A/B + Ld32x32b epilogue). This unblocks ANY future tcgen05 work.
- A working cublasLt-78 ctypes single-GEMM benchmark wrapper.
