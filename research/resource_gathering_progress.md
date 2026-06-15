# Resource gathering progress log

Format: `[YYYY-MM-DD HH:MM] URL — takeaway`

Session started: 2026-06-15

---
[2026-06-15 00:01] https://gau-nernst.github.io/tcgen05/ — idesc/smem-desc construction formulas, K-loop enable-input-d pattern, mbarrier PTX, 2-SM multicast commit, persistent scheduling skeleton
[2026-06-15 00:02] https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/ — CuTe C++ API: make_tiled_mma, get_slice(_0{}), make_fragment_A/B/C, TmemAllocator1Sm, make_tmem_copy SM100_TMEM_LOAD_32dp32b1x, ScaleOut::Zero/One
[2026-06-15 00:03] https://arxiv.org/abs/2512.02189 — abstract only; full PDF needed for latency/bw numbers
[2026-06-15 00:04] https://raw.githubusercontent.com/triton-lang/triton/main/python/tutorials/gluon/06-tcgen05.py — Gluon: TensorMemoryLayout + allocate_tensor_memory, tcgen05_mma(lhs,rhs_smem,acc_tmem,use_acc), tcgen05_commit(mbarrier), double-buffer pipeline; blockN=128 optimal
[2026-06-15 00:05] https://arxiv.org/html/2512.02189v2 — KEY NUMBERS: tcgen05 SI-LAT 11.0-11.4 cycles (vs Hopper 32.0), FP16→FP32 482 TFLOPS, BF16 1926 TFLOPS; TMEM 16 TB/s read BW/SM; CTA-pair = 1.27x training gain; TMEM = 1.26x training gain
[2026-06-15 00:06] https://api.github.com/repos/NVIDIA/cutlass/contents/examples/python/CuTeDSL/blackwell — 404; need gh CLI or direct file fetch
[2026-06-15 00:07] https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html — CONFIRMED: make_trivial_tiled_mma(ab_dtype, a_leading_mode, b_leading_mode, acc_dtype, cta_group, mma_tiler_mn, a_source=smem_desc) — SINGLE ab_dtype for both A and B
[2026-06-15 00:08] https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html — MmaF16BF16Op signature, Field enum (ACCUMULATE/NEGATE_A/NEGATE_B/SFA/SFB), SmemLayoutAtomKind values, Ld32x32bOp/Ld16x*Op, make_tmem_copy, make_smem_layout_atom, commit
[2026-06-15 00:09] gh api NVIDIA/cutlass/examples/python/CuTeDSL/blackwell — 404 (repo may be private or path wrong; will try alternate)
[2026-06-15 00:10] https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/ — BF16x9: up to 3x vs native FP32 on GB200, 2.4x on ecTrans; "equivalent or superior accuracy"; gains on moderate/large shapes; no code or type constants in blog
[2026-06-15 00:11] https://gau-nernst.github.io/tcgen05/part2/ — 404; single-page site
[2026-06-15 00:12] https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html — KeepCUBIN/KeepPTX/OptLevel Python types confirmed; cute.compile[OptLevel(2), KeepCUBIN, KeepPTX](kernel, args); artifacts access not documented in this page
[2026-06-15 00:13] gh tree NVIDIA/cutlass blackwell examples — empty (possibly requires auth)
[2026-06-15 00:14] https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html — alloc_tmem/dealloc_tmem/relinquish_tmem_alloc_permit/retrieve_tmem_ptr signatures confirmed; mbarrier_arrive_and_expect_tx confirmed; cvt_f32x2_bf16x2 confirmed; NO fence_view_async_tmem_load in docs
[2026-06-15 00:15] https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html — make_tiled_tma_atom full sig confirmed; tma_partition, prefetch_descriptor, update_tma_descriptor, fence_tma_desc_acquire/release all confirmed
[2026-06-15 00:16] https://deepwiki.com/gau-nernst/learn-cuda/8.1-tcgen05-instructions-and-tensor-memory — warp roles (0=mbarrier init, 1=alloc), TMEM addr encoding (row<<16|col), per-warp 32-lane restriction, mbarrier TMA→MMA→epilogue phase pattern
[2026-06-15 00:17] https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/06-tcgen05.py — COMPLETE SOURCE OBTAINED: TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma(lhs,rhs,acc,use_acc), tcgen05_commit(bar), mbarrier.init/expect/wait/invalidate, tma.async_load, fence_async_shared; pipelined pattern with double-buffer counters
[2026-06-15 00:18] https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor — tcgen05 single-thread-issue (ThrID=Layout<1> vs Hopper <128>); M=128 near 100% peak; M=64 ~50% peak; SMEM BW limits N<128 in SS mode; 2SM = 2x; paywall on CUTLASS patterns
[2026-06-15 00:19] https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/ — BF16x9 up to 3x on GB200, 2.4x real app (ecTrans); equivalent/better accuracy vs FP32; gains only on moderate/large shapes; no code in blog
[2026-06-15 00:20] https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html — COMPLETE API: PipelineTmaUmma.create(num_stages,producer_group,consumer_group,tx_count,...), PipelineUmmaAsync.create(...), PipelineState, NamedBarrier, PipelineProducer/Consumer with acquire_and_advance/wait_and_advance
[2026-06-15 00:21] https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py — KEY FINDING: make_trivial_tiled_mma supports BOTH old (ab_dtype) and new (a_dtype/b_dtype) APIs; make_smem_layout_a/b: partition shape → atom selection → tile → staging; get_tmem_load_op returns Ld16x64b/128b/Ld32x32b based on epi tile
[2026-06-15 00:22] https://arxiv.org/html/2511.13778v1 — Ozaki paper is INT8/FP64, NOT BF16x9; 3.7x QR trailing update speedup on Blackwell RTX Pro 6000; trailing update = 3 GEMMs: W←YᵀA, W←TᵀW, A←A-YW
[2026-06-15 00:23] https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul — TMA single-thread issue + mbarrier phase flip; TMEM alloc needs single warp; 128B swizzle Swizzle<3,4,3>; LBO=1024B/SBO=128B; TMA+tcgen05 = 155 TFLOPS, +swizzle = 288 TFLOPS on B200
[2026-06-15 00:24] https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-instruction — PTX TOC found (sections 9.7.17.7-12); actual PTX syntax not in fetched fragment; page too large
[2026-06-15 00:25] https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/ — 2-SM: cluster_layout_vmnk (AtomThrID=2); TMA multicast masks (same-parity CTAs); SMEM unified addr space (bit24=CTA_ID); umma_arrive_multicast_2x1SM for commit; mbarrier address trick: smem_int_mbar & 0xFEFFFFFF
[2026-06-15 00:26] WebSearch CUTLASS blackwell dense_gemm — CONFIRMED: dense_blockscaled_gemm_persistent.py + grouped_gemm.py + tutorial_gemm/fp16_gemm_0.py all exist on GitHub main; ab_dtype DEPRECATED in favor of a_dtype/b_dtype separately (CONFIRMED churn)
[2026-06-15 00:27] GitHub raw fp16_gemm_0.py — 404 (GitHub raw requires auth for private repo or path wrong)
[2026-06-15 00:28] https://ianbarber.blog/2025/07/04/cute-dsl/ — blog about Volta/Ampere CuTe, no Blackwell-specific content
[2026-06-15 00:29] https://gevtushenko.github.io/cccl/libcudacxx/ptx/instructions/tcgen05_alloc.html — navigation page only; syntax not extracted
[2026-06-15 00:30] dense_gemm.py (CuTeDSL examples/cute/blackwell) — CANONICAL RECIPE confirmed: make_fragment_A/B(sA/sB) directly on SMEM tensor; partition_shape_C(mma_tiler[:2]); make_fragment_C(acc_shape); cute.make_tensor(tmem_ptr, tCtAcc_fake.layout); cute.gemm(tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc)

---
Session complete: 2026-06-15. Wrote research/cutedsl_patterns.md + expanded research/blackwell_sources.md.
Total sources processed: 26 URLs across 6 topic areas.
