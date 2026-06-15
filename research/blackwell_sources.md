# Blackwell / tcgen05 frontier sources (public) — for the all-in megakernel

Curated from public material (GPU MODE publishes its submission corpus + requires winners to open-source;
the leaderboard CLI only exposes YOUR OWN submissions, so competitor source is mined from these instead).

## ⭐ STRATEGIC: a Triton-native TMEM/tcgen05 path (maybe NO CuTe→cubin needed)
- **Triton Gluon tcgen05 tutorial:** https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/06-tcgen05.py
  Triton 3.x "Gluon" sublanguage exposes **Tensor Memory allocation + `tcgen05.mma`** directly in Python:
  allocate TMEM, issue UMMA, pipeline MMAs. **The grader HAS Triton** (and our Triton cubins already work)
  → a Gluon kernel could ship with ZERO cubin-embed/driver-load plumbing — far simpler than CuTe-DSL.
  **EVALUATE Gluon vs CuTe-DSL FIRST in Stage 1** (build the tcgen05 BF16x9 GEMM both ways, pick the
  shippable+fast one). This could de-risk the whole effort.

## tcgen05 / TMEM practical tutorials (Stage 1 implementation refs)
- **"tcgen05 for dummies" — gau-nernst:** https://gau-nernst.github.io/tcgen05/  Basic→advanced tcgen05.
  Confirms: `tcgen05.mma` has **single-thread semantics** (one thread issues the MMA), operands live in
  **smem + Tensor Memory** (no register operands). THE hands-on intro.
- **Colfax: "Writing GEMM Kernels Using Tensor Memory for Blackwell":**
  https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
- **Colfax: "GEMM with Thread Block Clusters on Blackwell":**
  https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/
- **NVIDIA CUTLASS CuTe Blackwell tutorial examples** (reference for the CuTe→cubin route, B8):
  - examples/cute/tutorial/blackwell/01_mma_sm100.cu (single-SM UMMA)
  - examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu (2-SM / CTA-pair UMMA + TMA)

## Hardware microarchitecture (latencies, throughput, extended precision = BF16x9)
- **arXiv 2512.02189 "Microbenchmarking NVIDIA's Blackwell Architecture":**
  https://arxiv.org/html/2512.02189v2  TMEM latency/bandwidth, tcgen05 throughput, CTA-pair scheduling,
  **extended-precision (the bf16x9-style exact-FP32) support** measured. Use for the perf model.
- **SemiAnalysis "Dissecting Nvidia Blackwell — Tensor Cores, PTX, SASS":**
  https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor
- **SemiAnalysis "NVIDIA Tensor Core Evolution: Volta→Blackwell":**
  https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell

## GPU MODE competition corpus + winners (technique inspiration)
- **GPUMODE/kernelbot-data (HuggingFace):** https://huggingface.co/datasets/GPUMODE/kernelbot-data
  ~110K REAL submissions from the AMD MI300 comps (fp8-gemm, moe, mla, all2all, gemm+reducescatter,
  allgather+gemm). **AMD/HIP ISA**, so technique-inspiration (blocking, warp-spec, async pipelining)
  NOT directly portable to B200 tcgen05 — but the structural patterns transfer.
- **NVIDIA "Topping the GPU MODE Kernel Leaderboard with cuda.compute":**
  https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute/
  CCCL/CUB "speed-of-light" primitives won many B200 boards (note: parallel primitives, not dense linalg).
- **gpu-mode/reference-kernels** (problem sets) + **gpu-mode/kernelbot** (platform) on github.com/gpu-mode.

## How this plugs into our plan
- Stage 1 (tcgen05 BF16x9 GEMM beating cuBLAS on the trailing shapes): start from the Gluon tutorial AND
  the Colfax/gau-nernst TMEM refs; benchmark both shippable routes (Gluon vs CuTe→cubin).
- BF16x9 exact-FP32: implement the 3-split→9-bf16-MMA Ozaki scheme on tcgen05 (the microbenchmarking
  paper confirms the extended-precision path); cross-check vs v18's cublasLt type-78 numbers (findings B6).
- Stage 2 (fused QR megakernel): TMEM-resident trailing update + warp-spec, per `research/tcgen05_tmem.md`.

---

## Additional sources added 2026-06-15 (from research/cutedsl_patterns.md session)

### CuTe-DSL Python API docs (CONFIRMED sources)
- **CUTLASS utils_sm100 API:** https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html — make_trivial_tiled_mma (full sig confirmed), make_smem_layout_a/b, get_num_tmem_alloc_cols, get_tmem_load_op, compute_epilogue_tile_shape
- **CUTLASS tcgen05 DSL API:** https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html — MmaF16BF16Op sig, Field enum (ACCUMULATE/NEGATE_A/NEGATE_B), SmemLayoutAtomKind values, Ld32x32bOp, make_tmem_copy, commit
- **CUTLASS cpasync (TMA) API:** https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html — make_tiled_tma_atom full sig, tma_partition, update_tma_descriptor, fence_tma_desc_acquire/release
- **CUTLASS arch API:** https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html — alloc_tmem/dealloc_tmem/relinquish, mbarrier_arrive_and_expect_tx, elect_one, cvt_f32x2_bf16x2
- **CUTLASS pipeline API:** https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html — PipelineTmaUmma, PipelineUmmaAsync, PipelineTmaAsync, PipelineState, NamedBarrier, PipelineProducer/Consumer full method signatures
- **CUTLASS JIT options:** https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html — KeepCUBIN/KeepPTX/OptLevel confirmed

### Triton Gluon (COMPLETE SOURCE OBTAINED)
- **Triton Gluon 06-tcgen05.py FULL SOURCE:** https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/06-tcgen05.py
  Complete: TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma(a,b,acc,use_acc), tcgen05_commit(bar),
  mbarrier.init/expect/wait/invalidate, tma.async_load, fence_async_shared, NVMMASharedLayout, TensorDescriptor,
  both blocked and pipelined matmul kernels with double-buffer counter pattern.

### Performance numbers (CONFIRMED from arXiv 2512.02189)
- tcgen05 SI-LAT: **11.0 cycles** (m64n64k16) vs Hopper 32.0 cycles → 2.9× lower
- TMEM bandwidth: **16 TB/s read** per SM
- FP16→FP32 throughput: **482 TFLOPS** (FP32 accum halves throughput vs FP16 accum)
- CTA-pair gain: **1.27×** training speedup
- 2-SM perfect weak scaling (2× throughput)

### Warp-spec + swizzle (confirmed from Modular + SemiAnalysis)
- TMA = single-thread issue (one elected thread issues cp.async.bulk.tensor)
- SMEM 128B swizzle = Swizzle<3,4,3>; LBO=1024B, SBO=128B
- tcgen05 = single-thread issue (ThrID=Layout<1> in CuTe, vs Hopper Layout<128>)
- With TMA+tcgen05+swizzle: 288 TFLOPS achieved in Modular series

### blackwell_helpers.py (key finding)
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py
  **CONFIRMED ab_dtype is DEPRECATED** — use a_dtype/b_dtype separately in make_trivial_tiled_mma.
  Both APIs coexist in 4.5.2; new kernel code should use split API.
