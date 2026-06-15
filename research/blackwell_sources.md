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
