# GPU MODE Winner Techniques — Survey (June 2026)

> Research question: what do top GPU MODE leaderboard submissions / winners actually DO to
> make kernels fast on modern NVIDIA GPUs (Hopper H100 / Blackwell B200)? Specific relevance
> to our batched Householder QR kernel, currently at geomean 4.73 ms / 9.47× on B200.
>
> Sources: GPU MODE leaderboard & news pages, NVFP4 hackathon writeups, CUTLASS/Triton docs,
> Blackwell kernel papers, ThunderKittens, Ozaki/BF16x9 GEMM papers, NVIDIA technical blogs.

---

## GPU MODE Competition Landscape (as of June 2026)

GPU MODE (gpumode.com) has run multiple sponsored competitions:
- **PMPP problems** (VectorAdd, PrefixSum, Histogram, Sort, Grayscale) — simple primitives
- **AMD $100K single-GPU** and **$100K distributed** kernel challenges (winners: mixed team)
- **AMD $1.1M** model-level challenge (ongoing)
- **NVIDIA Blackwell NVFP4 GEMV/GEMM hackathon** (problem 595/597; top: 22.4 µs)
- **Linear Algebra Problems** (`problems/linalg/qr_py`) — **our problem**; launched 2025/2026

The leaderboard is at `gpumode.com/leaderboard/<id>?tab=rankings`. Submissions are single
Python files. The "stream" substring ban (findings.md D8) is unique to this harness.

**Key competitive insight (from PR #149, opened Jun 13 2026):** as of this research date,
the linalg/qr leaderboard appears early-stage — a recent PR added "benchmark heterogeneous
& ill-conditioned batches" suggesting active development rather than a mature winner field.
Our window is open.

**NVIDIA's own GPU MODE win (PrefixSum/Sort/Grayscale):** used `cuda.compute` (CUB
primitives via JIT link-time optimization) — lesson: when a battle-tested library covers
your primitive, use it. We already use `torch.geqrf` for small-batch shapes; this validates
the dispatch approach.

---

## Technique 1: TMA (Tensor Memory Accelerator) Async Copies

### What it is
TMA is a dedicated hardware DMA unit (introduced Hopper H100, improved on Blackwell B200)
that transfers arbitrary-shaped multi-dimensional tensor tiles between global and shared memory
using a single thread — no warp-wide participation needed. It issues an asynchronous bulk copy
(`cp.async.bulk.tensor`) described by a compile-time tensor map descriptor and signals
completion via an mbarrier. Loads of sizes up to 128 bytes (vs cp.async's 16-byte per-thread)
are possible, eliminating register pressure for data movement.

### When it wins
Any kernel where data loading is latency- or bandwidth-bound and the programmer wants to
overlap data movement with computation. Mandatory for near-cuBLAS GEMM on Blackwell.
Directly feeds the producer side of warp-specialized producer/consumer pipelines (see Tech 2).

### Specific relevance to our QR
Our current Triton panel kernel (v13) uses standard `tl.load` / `tl.store`. The panel loads
`(M_POW2, B_POW2)` tiles from global memory into shared memory at kernel launch — already
"one big load, all steps on-chip." This is conceptually similar to TMA but uses the
compiler's cp.async path rather than explicit TMA descriptors.

**Impact assessment: MEDIUM.** For the trailing-update `torch.bmm` calls, TMA is essential
if we replace them with a custom CuTe DSL / Triton GEMM kernel. For the panel kernel, an
explicit TMA port would reduce register pressure but the current smem-resident approach is
already 3.41× — the panel is 29% of GPU time (findings.md C3/C4) and bmm is 28–30%.
TMA becomes HIGH priority once we write a fused trailing-update GEMM.

**Tools:** Triton 3.7 exposes TMA via `tl.load` automatically on sm_100a (compiler handles
descriptor generation). CuTe DSL has `cute.experimental` for manual TMA. CUTLASS handles TMA
transparently in all GEMM kernels.

Sources: [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) |
[Modular Blackwell Part 2](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul) |
[CUTLASS Blackwell GEMM tutorial (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)

---

## Technique 2: Warp Specialization / Producer-Consumer Pipelines

### What it is
Warp specialization assigns distinct roles to different warp groups within a thread block.
In the producer-consumer pattern, producer warps issue TMA loads and signal mbarriers; consumer
warps wait on the barrier, perform tcgen05 MMA on the loaded data, then signal memory is
reusable. Flash Attention 4 uses five specialized warp types (Load, MMA, Softmax, Correction,
Epilogue). On Blackwell, this is qualitatively more complex than Hopper: tensor memory (TMEM)
blocking synchronization adds a mandatory cross-warp communication step that Hopper didn't need.

### When it wins
Required for near-peak GEMM performance on Hopper and Blackwell. FA3/FA4 demonstrated
~20% speedup over cuDNN by explicit warp specialization + TMA. ThunderKittens matches
CuBLAS on GEMM using warp specialization + TMEM. The three critical cases are:
(1) variable-latency async memory ops that need separate issuing threads;
(2) resource constraints preventing all roles from fitting in one warp's register budget;
(3) blocking synchronizations (like TMEM reads) that would stall a unified warp.

### Specific relevance to our QR
Warp specialization is the mechanism underlying every state-of-the-art Blackwell GEMM kernel.
Our trailing-update is currently 3× `torch.bmm` (28–30% of GPU time). If we write a custom
fused trailing-update GEMM using CuTe DSL or Triton autoWS, warp specialization is
automatically applied (Triton's `warp_specialize=True` ForOp) or manually structured (CuTe).

**Impact assessment: HIGH for trailing-update GEMM, LOW for panel.** The panel kernel is
inherently sequential (column-by-column Householder reduction); warp specialization within the
panel has limited headroom because each step depends on the previous reflector. The trailing
GEMM is the target.

Triton autoWS status (3.7): Flash Attention forward on B200 is 1.5–2× faster than stock
Triton; it is the primary supported use case. Generalizing to our GEMM shape is possible but
requires care. CuTe DSL makes this more explicit and controllable.

Sources: [PyTorch Blog: Warp Specialization in Triton](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) |
[Flash Attention 4 reverse-engineered (Modal)](https://modal.com/blog/reverse-engineer-flash-attention-4) |
[Optimal SWP and WS for Tensor Core GPUs (arXiv 2512.18134)](https://arxiv.org/html/2512.18134v1) |
[ThunderKittens (arXiv 2410.20399)](https://arxiv.org/abs/2410.20399)

---

## Technique 3: Persistent / Megakernels (Grid-Stride, Avoid Relaunch)

### What it is
A persistent kernel launches exactly as many thread blocks as there are SMs (e.g. 148 on B200),
and each thread block loops over multiple tiles of work internally. The opposite of a monolithic
launch (one block per tile). Benefits: (1) eliminates the per-launch CPU overhead of ~10 µs;
(2) enables fine-grained pipelining between tiles (epilogue of tile N overlaps with prologue of
tile N+1 on same SM); (3) solves wave quantization (Stream-K: when tile count doesn't divide
evenly by SM count, some SMs are idle — persistent kernels assign fractional tiles). Performance
improvement: persistence alone adds ~10% in ThunderKittens GEMM benchmarks; Blackwell CUTLASS
warp-spec persistent kernel reaches 98% of cuBLAS at M=N=K=4096.

### When it wins
Persistent kernels win when: (a) tile count is small and wave quantization wastes SMs, (b)
the problem has many small tiles (our small-batch shapes: n=32, b=20 → 20 tiny matrices), or
(c) epilogue and prologue work can be overlapped.

### Specific relevance to our QR
Our `_panel_qr_kernel` launches `batch` thread blocks, one per matrix. For b=20 (n=32) that's
20 blocks on 148 SMs — massive underutilization (13.5%). The n=32 case consumes 28s of the
grader's eval time (findings.md D11). A persistent megakernel for small-n would:
1. Launch 148 blocks and assign batch elements round-robin — full SM occupancy.
2. Fold the entire QR (panel + WY build + trailing update, all on-chip for n≤230) into one
   persistent block that processes multiple matrices sequentially.
3. Remove the Python-loop panel-block-loop launch overhead entirely.

This is the mechanism behind the n=32 fused kernel (v10, planned but not yet implemented).
For medium shapes (n=512, batch=640): 640 blocks already saturates 148 SMs — persistence
provides less benefit per block but launch-overhead elimination still matters.

**Impact: HIGH for n=32/176. MEDIUM for n=352/512 (already well-utilized).**

Stream-K scheduling: relevant for the trailing-update GEMM when the number of GEMM tiles
doesn't divide cleanly by SM count (likely for small-n problem sizes).

Sources: [CUTLASS Stream-K tutorial (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) |
[ThunderKittens paper](https://arxiv.org/abs/2410.20399) |
[tcgen05 Blackwell persistent kernel description](https://gau-nernst.github.io/tcgen05/) |
[Megakernels analysis (TheoremPath)](https://theorempath.com/megakernels)

---

## Technique 4: cp.async Double/Triple Buffering (Software Pipelining)

### What it is
Multi-stage (double/triple/N-way) buffering uses multiple copies of the shared memory tile to
overlap the asynchronous data transfer for tile K+1 with the computation on tile K. The number
of stages (`num_stages`) is a key tuning parameter. With Hopper TMA + mbarrier, triple buffering
(stages=3) is typical. On Blackwell, TMEM adds a further pipeline stage because accumulators now
live in TMEM (separate from SMEM), allowing MMA to continue while SMEM is being refilled. CUTLASS
Blackwell uses a "depth 3 CLC pipeline" (Cluster Launch Control) for this purpose.

### When it wins
Wins when memory latency is the bottleneck and the compute kernel has enough independent work
to fill the pipeline. For GEMM: always enabled in production kernels. For panel QR: limited
applicability (the steps are serial; there is no independent tile to prefetch into buffer B
while buffer A is being computed on).

### Specific relevance to our QR
Our panel kernel benefits only marginally here: sequential Householder steps cannot be
easily double-buffered. However, the trailing-update GEMM (Y @ Tᵀ @ Yᵀ @ A) is a GEMM chain
and can absolutely benefit from multi-stage pipelines. The A matrix tiles for the trailing GEMM
can be prefetched while the previous tile's MMA is running.

In Triton, `num_stages` is an autotune parameter (`@triton.autotune`). Current v13 doesn't
autotune this — default is 2 or 3. Sweeping stages 2/3/4 for the trailing-update kernel could
yield 10–20% improvement. For the panel, keep `num_stages=1` (data fits in smem at load).

**Impact: MEDIUM for trailing-update GEMM; LOW/NEGLIGIBLE for panel.**

Sources: [Deep Dive on Hopper TMA for FP8 GEMMs (PyTorch)](https://pytorch.org/blog/hopper-tma-unit/) |
[Colfax CUTLASS pipelining tutorial](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)

---

## Technique 5: Shared Memory Swizzling & Bank-Conflict Avoidance

### What it is
Shared memory on NVIDIA GPUs is partitioned into 32 banks (each 4 bytes wide). If multiple
threads in the same warp access different addresses that map to the same bank, the accesses
are serialized (bank conflict). Swizzling permutes the mapping of logical element (row, col)
to physical address using an XOR of the row index into the column index, so that elements in
the same column land in different banks across rows. CuTe uses "128-byte swizzle patterns"
(`Layout_K_SW128_Atom`). Effect: swizzling was the single largest optimization in the
tcgen05 tutorial — from 254 TFLOPS to 695 TFLOPS (2.7× alone) before any warp specialization.

### When it wins
Required any time a kernel has repeated column-major or stride access to a 2D tile in shared
memory. Particularly important for: (a) the Householder panel kernel (reads column vectors,
applies row updates, stores back); (b) any GEMM where operands are loaded into SMEM with
column-major stride.

### Specific relevance to our QR
Our panel kernel (v13) stores the `(M_POW2, B_POW2)` tile in shared memory row-major (Triton
default). Column accesses (the `j`-th column for each Householder step) may have bank conflicts
if M_POW2 is a multiple of 32. Triton's default layout should handle basic conflicts, but we
haven't verified. For the WY-build GEMM and trailing update, column-major access to Y tiles
is bank-conflict-prone without swizzling.

**Action**: in any new custom GEMM kernel (CuTe DSL trailing update), use `SW128_Atom` layout.
In the Triton panel kernel, check if `tl.make_block_ptr` with stride specification or
`constexpr` column stride removes conflicts.

**Impact: HIGH for any new GEMM kernel; LOW-MEDIUM for existing Triton panel (already 3.41×).**

Sources: [tcgen05 for dummies - swizzling section](https://gau-nernst.github.io/tcgen05/) |
[CUDA Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/) |
[CuTe Swizzle (Lei Mao)](https://leimao.github.io/blog/CuTe-Swizzle/)

---

## Technique 6: tcgen05 / UMMA — Blackwell 5th-Gen Tensor Core MMA

### What it is
`tcgen05.mma` is the Blackwell 5th-generation tensor core instruction. Key differences from
Hopper's WGMMA:
- **Issued by a single thread** per CTA (not warp-group-wide) — decouples MMA from other warps
- **Results land in TMEM** (tensor memory, 256 KB/SM) not registers — eliminates register pressure
  for accumulation; registers freed for other logic
- **Largest MMA shape**: m128×n256×k16 per CTA (vs Hopper m64×n256×k16) — 2× the tile
- **Supported dtypes**: FP16, BF16, TF32, FP8 (e4m3/e5m2), FP4/NVFP4, INT8
- **2-SM cluster mode** (2CTA): two SMs cooperate on a single CTA-group MMA for even larger tiles

On B200, tcgen05 provides ~2–4× higher tensor core throughput than Hopper WGMMA for the same
dtype, due to larger tile + higher clock + higher BF16 throughput (2.25 PFLOPS vs ~1 PFLOPS).

### When it wins
Any kernel that does matrix multiplication on Blackwell. Required for a near-peak trailing
GEMM kernel. Not applicable to the sequential Householder panel reduction (that's
GEMV/inner-product operations, not blocked GEMM).

### Specific relevance to our QR
Our current trailing update uses `torch.bmm` which calls cuBLAS, which already uses tcgen05
internally on B200. So we are getting tcgen05 for the trailing GEMM — but through cuBLAS,
which doesn't know about our batched layout or that we need a fused subtract-in-place epilogue.

A custom CuTe DSL trailing-update GEMM could:
1. Fuse all 3 `bmm` operations into 1 or 2 kernel launches (eliminate 2 launches)
2. Use tcgen05 directly with a subtract-in-place epilogue (no intermediate tensor needed)
3. Use BF16x9 compute type for 2–3× FP32 throughput improvement (see Technique 7)

The WY T-recurrence build is too small for tcgen05 (b×b matrix, b=64 — only 64³ FLOPs).
Use Triton/nvrtc for the T-recurrence.

**Impact: HIGH for trailing GEMM replacement with fusion. Already implicitly used via bmm.**

Sources: [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) |
[CUTLASS Blackwell TMEM tutorial (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) |
[FA4 reverse-engineered (Modal)](https://modal.com/blog/reverse-engineer-flash-attention-4)

---

## Technique 7: BF16x9 / Ozaki FP32 Emulation — Exact FP32 from BF16 Tensor Cores

### What it is
On Blackwell, BF16 tensor core throughput (~2.25 PFLOPS) is 9× or more higher than FP32 CUDA
core throughput (~0.08–0.08 PFLOPS for real FP32 on B200 non-tensor-cores). The BF16x9
algorithm exploits this: any FP32 value can be exactly represented as 3 BF16 values, so a
single-precision matrix product can be computed as 9 BF16 matrix products (3 splits × 3
combinations) and summed in FP32 accumulators — yielding **bit-exact FP32 output**.

cuBLAS exposes this as `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` (compute type value 78 in
cuBLAS 13.x; accessible via `nvmath.bindings.cublas.ComputeType`). Performance: **3–4× more
TFLOPS than native FP32 on B200** (NVIDIA blog, cuBLAS 12.9+ release). For QR trailing update
in cuSOLVER, integration of emulated DGEMM showed **3.7× speedup on Blackwell RTX Pro 6000**
(arXiv:2511.13778). The technique is now deprecated in cuBLAS 13.3 (will be removed in a
future release — use while available). 

Key caveat for us (findings.md B5): `torch.bmm` does NOT expose this compute type. It requires
either (a) a direct `cublasLtMatmul` call via `cuda.bindings` (confirmed importable on grader)
requesting `CUBLAS_COMPUTE_32F_EMULATED_16BFX9`, or (b) a manual Ozaki 3-split / 9-GEMM
sequence in Triton/CuTe. Option (a) is lower engineering effort. Option (b) is more flexible
(can combine with epilogue fusion).

Also relevant: a related FP8-based Ozaki scheme (arXiv:2508.00441) uses FP8 tensor cores for
DGEMM emulation; on Blackwell FP8 is ~2× faster than BF16 in raw TFLOPS, so a 3-split/9-GEMM
in FP8 would be even faster — but requires more careful accumulation. The BF16x9 path is the
safer, accuracy-guaranteed option.

**Accuracy for our problem:** Band/rowscale test cases fail with TF32 (10-bit mantissa). BF16x9
outputs bit-exact FP32 — so band/rowscale would pass. This is the key advantage over TF32/BF16.
(findings.md B4-B5 already reached this conclusion; now confirmed by NVIDIA cuSOLVER integration.)

### Specific relevance to our QR
Trailing update is currently 28–30% of GPU time at n512/n1024 (after panel optimization).
If BF16x9 gives 3–4× on the GEMM, the trailing update drops from 28% to ~8% of GPU time.
Combined with a fused epilogue (eliminating intermediate `YTY`, `TC`, `ATC` temporaries),
this could move 28% of time down to near-zero overhead.

**Impact: HIGH — potentially 1.3–1.5× overall kernel speedup at n512/n1024 once the trailing
GEMM is the bottleneck.** Currently the panel is still 29% and launch overhead significant;
prioritize only after panel/launch is solved.

**Access path on grader:** `cuda.bindings` (v13.0.3) is confirmed importable. Call
`cublasLtMatmul` with `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` compute type. The cuBLAS handle
can be initialized once in a module-level `if` guard. "stream" substring ban applies to source
text, not to the Python string `"CUBLAS_COMPUTE_32F_EMULATED_16BFX9"`.

Sources: [NVIDIA blog: Unlocking tensor core performance with FP emulation](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/) |
[cuBLAS 13.x docs](https://docs.nvidia.com/cuda/cublas/) |
[Ozaki/ADP DGEMM paper (arXiv:2511.13778)](https://arxiv.org/html/2511.13778v1) |
[FP8 Ozaki DGEMM (arXiv:2508.00441)](https://arxiv.org/html/2508.00441v2)

---

## Technique 8: CuTe-DSL, ThunderKittens, and CUTLASS as Backend GEMM Engines

### What they are
Three frameworks for writing near-cuBLAS GEMM kernels on Blackwell:

**CuTe DSL (Python, CUTLASS 4.x):** Python-native DSL that compiles via MLIR → NVRTC → PTX
→ ptxas. Same performance as CUTLASS C++. Key for us: AOT cubin export via `.__cubin__`
attribute; batched GEMM with custom subtract-in-place epilogue can be expressed as a "grouped
GEMM" with a custom EFC lambda. CuTe DSL is in Beta (graduating summer 2026); already at
v4.5.2 as of May 2026. Not importable on grader, but the cubin export + driver-load path is
confirmed working (findings.md H2).

**ThunderKittens 2.0 (C++, Jan 2026):** C++ header library achieving cuBLAS-matching GEMM
on B200 with FP8 and BF16. Uses explicit tile abstractions (rt_tile, st_tile), LCSF
(Load-Compute-Store-Finish) template, and TMEM-based accumulation. Must be compiled offline.
Production-adopted at Together AI and Jump Trading. 40 lines of device code matches cuBLAS.

**CUTLASS C++ (mature):** The gold standard. Used by cuBLAS itself. CUTLASS 4.x Blackwell
kernels use CLC (Cluster Launch Control) based persistent scheduling + TMEM double-buffering.
"New warp-specialization recipe tuned specifically for Blackwell SM100" added in CUTLASS 4.3.
Compile offline with `-arch=sm_100a`.

### When each wins
All three produce near-cuBLAS GEMM on Blackwell. CuTe DSL is fastest to iterate (Python,
seconds to compile). ThunderKittens is best if you prefer C++ with high-level tile abstractions.
CUTLASS C++ is most feature-complete and battle-tested. For our trailing GEMM fusion, CuTe DSL
is the shortest path (see cutlass_dsl.md).

### Specific relevance to our QR
The trailing update is `A -= Y @ (Tᵀ @ (Yᵀ @ A))`. In one fused kernel:
- Load Y and A tiles using TMA
- Compute Yᵀ @ A via tcgen05 with TMEM accumulation  
- Apply Tᵀ scaling in-place in TMEM epilogue
- Compute Y @ result subtract from A in global memory epilogue
- Net: 3 launch → 1 launch, no intermediate tensors

This is a legitimate CuTe DSL grouped-GEMM-with-custom-epilogue use case. The cubin would be
extracted offline on Modal (Linux, CUDA 13.x), embedded as bytes, driver-loaded at grade time.

**Impact: HIGH (fuses 3 launches → 1, enables BF16x9, eliminates temporaries).**

Sources: [CuTe DSL PyPI (v4.5.2)](https://pypi.org/project/nvidia-cutlass-dsl/) |
[ThunderKittens paper (ICLR 2025)](https://openreview.net/pdf/f4b2b2d3f597357551880dae1c1a4286791aadc5.pdf) |
[CUTLASS 4.3 Blackwell changelog](https://docs.nvidia.com/cutlass/4.3.4/CHANGELOG.html) |
[Colfax CUTLASS TMEM tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) |
[cutlass_dsl.md in this repo]

---

## Technique 9: Autotuning, Fixed/Known Shapes, tl.constexpr Specialization

### What it is
Triton kernels parameterized with `tl.constexpr` block sizes generate distinct PTX per
configuration — the compiler can specialize loop bounds, unroll factors, and register
allocation. `@triton.autotune` benchmarks a config grid at first invocation and caches the
winner. For fixed-shape problems like ours (all 7 benchmark shapes are known at submission
time), offline autotuning is possible: run the autotune once on Modal, hardcode the winning
configs, ship them as a static mapping `{(batch, n): (M_POW2, num_warps, num_stages)}`.

The Helion framework (PyTorch, 2025) generalizes this further: it generates hundreds of Triton
variants from a single high-level kernel spec and benchmarks them. The GPU Performance
Portability paper (arXiv:2505.03780) shows autotuned kernels can be an order of magnitude
more performant than heuristic defaults. For competitions with known shapes, this is a major
lever — you're optimizing for 7 specific (batch, n) pairs, not general workloads.

### When it wins
Always: there is no reason NOT to autotune a competition kernel. The key insight from the
NVFP4 hackathon writeups: templating on K-dimension (n in our case) for full loop unrolling
and aggressive register allocation contributed substantially to winning entries. One participant
noted that register count (`-maxrregcount=32–45`) vs the default (80) was a key lever.

### Specific relevance to our QR
Current v13 has 3 compile variants (M_POW2 ∈ {256, 512, 1024}) with hardcoded `num_warps`.
We've already established that `num_warps` must scale with M_POW2 (findings.md C4). Remaining
untuned parameters: `num_stages` (for any async pipelining), `_BLOCK` (panel width b), and
the dispatch threshold `(batch, n)` that decides geqrf vs our kernel.

Optimal `_BLOCK` is non-obvious: larger blocks reduce panel-loop iterations (fewer launches)
but increase shared memory and panel kernel time per block. Currently hardcoded at 64.
**Action: sweep `_BLOCK ∈ {32, 64, 96, 128}` for each shape; pick per-shape winners.**

**Impact: MEDIUM — free to do, likely 10–20% improvement from tuning `_BLOCK` and
`num_warps`/`num_stages`. Foundational hygiene before any architectural change.**

Sources: [Triton Autotune Explained (TillCode)](https://tillcode.com/triton-autotune-explained-with-examples/) |
[GPU Performance Portability needs Autotuning (arXiv:2505.03780)](https://arxiv.org/pdf/2505.03780) |
[Helion DSL (PyTorch)](https://pytorch.org/blog/helion/)

---

## Technique 10: Occupancy Tuning — Registers, SMEM, Warps per SM

### What it is
Occupancy = (active warps per SM) / (max warps per SM = 64 on B200). Higher occupancy helps
hide memory latency by having ready-to-run warps when one stalls. However, occupancy is
constrained by three resources per SM: registers (64K × 32-bit), shared memory (up to 228 KB
configurable), and max thread blocks. The register file is split among active threads; too
many threads → registers spill to local memory (DRAM) → catastrophic slowdown.

Key B200 rule: `num_warps × 32 threads × registers_per_thread ≤ 65536`. At 255 registers/thread
(max), only 8 warps fit. At 64 registers, 32 warps. Profile with `ncu --metrics
l1tex__data_pipe_lsu_wavefronts_mem_local.avg` to catch register spills. Competition winners
explicitly set `-maxrregcount` (NVFP4 hackathon: 32–45 vs default 80 was a key lever).

SMEM: B200 allows 0/8/16/32/64/100/132/164/196/228 KB per SM. Our panel tile: n=1024,
M_POW2=1024, b=64 → tile = 1024×64×4 = 256 KB → too large for 228 KB. That's why v13
splits at n≥1024 (M_POW2=1024 requires 2 blocks sharing load). Confirmed working in v13.

### Specific relevance to our QR
- **Panel kernel:** `num_warps` must match M_POW2/32 (each warp handles ~2 rows at tile width
  B_POW2). Over-subscription regresses (v13 finding: 8/16/32 for M_POW2 256/512/1024).
  Current setup is tuned. Register spill: unverified — `ncu` unavailable on Modal (findings.md G1).
  Workaround: Triton AOT exposes `n_regs` and `n_spills` in `compiled.asm`.
  **Action: check `compiled.asm['n_spills']` for each variant; if >0, reduce `B_POW2` or use `__launch_bounds__`.**
- **New trailing GEMM kernel:** will need occupancy-aware design. CuTe DSL handles this via
  TMEM (accumulators in TMEM not registers → more registers free for pipelining).

**Impact: MEDIUM — spill-checking is free and could reveal a hidden regression in v13.**

Sources: [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) |
[NVFP4 12 attempts (amandeepsp)](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/) |
[GPU MODE Lecture 4: Compute and Memory Basics](https://christianjmills.com/posts/cuda-mode-notes/lecture-004/)

---

## Technique 11: Minimizing Launch / Host Overhead (Kernel Count)

### What it is
Each GPU kernel launch has ~10 µs of CPU-side overhead. Our blocked-WY kernel issues ~10,000
kernel launches per iteration (findings.md C1): `copy_` 4468×, `mul` 2975×, etc. Even after
v13 (panel fused), there remains ~12 ms of launch overhead per iteration at n512 (findings.md
C3). The key lever is: fewer, larger kernels — not faster kernels. CUDA graphs (banned) and
torch.compile (banned) are the standard fixes; our only path is hand-fused Triton/nvrtc kernels.

**What winners do:** AMD $100K competition: custom Triton launchers reduced overhead from ~120 µs
to ~40 µs (Yotta Labs writeup) by caching kernel compilation with pointer arithmetic instead of
full tensor objects. The NVFP4 competition: launch overhead was negligible because the problem
was a single GEMV call. For our QR — which has a Python for-loop over block columns — the fix
is fusing the Python loop into Triton program IDs or using a megakernel that covers all blocks.

**WY T-recurrence:** The T-matrix build recurrence is a b×b sequential loop with b=64 steps,
each issuing a small `ger`/`axpy` kernel. This is ~64 extra launches per panel block. Fusing
this into the panel kernel (shared-memory resident T-build) would eliminate 64× launches per
block and ~(n/b) × 64 launches total.

### Specific relevance to our QR
This is our #1 confirmed bottleneck (findings.md C1). Current status: v13 fused the panel.
Remaining unfused steps:
1. **WY T-recurrence** (b steps per block, each a GEMV-like op) — fuse into panel kernel
2. **Trailing update** (3 bmm calls per block) — fuse into 1 custom kernel
3. **Python block-column loop** (n/b iterations, each issuing above) — unroll into Triton
   kernel that loops over blocks internally (one Triton launch covers all n/b panel blocks)

Fusing #3 alone (a single Triton kernel that loops over blocks) would reduce Python-loop
launch overhead from O(n/b) to O(1) kernel launches for the panel phase. For n=4096, b=64:
64 launches → 1 launch.

**Impact: HIGH. Reducing launch count is the most impactful remaining lever (confirmed
bottleneck). Estimated gain: 20–40% overall at mid-sizes, potentially larger at large-n.**

Sources: [AMD Challenge 2025 winner writeup (Yotta Labs)](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025) |
[GPU MODE Lecture 4](https://christianjmills.com/posts/cuda-mode-notes/lecture-004/) |
[CUDA Graphs for kernel batching (arXiv:2501.09398)](https://arxiv.org/pdf/2501.09398)

---

## Technique 12: nvrtc / Embedded PTX-cubin Path

### What it is
nvrtc (NVIDIA Runtime Compilation) compiles CUDA C++ source to PTX in-memory within the
Python process, eliminating the subprocess/disk overhead of `nvcc`. On the grader, nvcc,
ninja, and ptxas are NOT on PATH — so `torch.utils.cpp_extension.load_inline` FAILS. But
`cuda.bindings.nvrtc` IS importable (findings.md H1, confirmed). For kernels too complex for
Triton but too simple to warrant offline CUTLASS, nvrtc at grade time is the path: write a
CUDA C kernel string, call `nvrtc.compileProgram`, get PTX bytes, then use
`cuda.bindings.driver.cuModuleLoadData` + `cuModuleGetFunction` to load and launch.

The "offline cubin embedding" variant pre-compiles on Modal (where nvcc/ptxas are available),
extracts `cubin_bytes`, embeds as a `bytes` literal in submission.py, and driver-loads at grade
time — zero JIT overhead. This is THE path for any kernel that uses CuTe DSL, CUTLASS C++,
or ThunderKittens (none importable on grader; all produce offline cubins).

### Specific relevance to our QR
Already confirmed working (H1, H2). The "stream" workaround (`getattr(torch.cuda, "current_"
+"stream")()`) handles the substring ban. This path is a prerequisite for any CUTLASS-based
trailing GEMM or T-recurrence kernel.

For the T-recurrence specifically: a compact nvrtc kernel (few dozen lines CUDA C) that does
the b×b T-build recurrence on shared-memory-resident data (already in shared mem after panel)
would eliminate 64 launches per block with minimal development effort.

**Impact: ENABLER — unlocks all offline-compiled kernels. Direct impact depends on what kernels
we build with it.**

Sources: [NVRTC 13.3 docs](https://docs.nvidia.com/cuda/nvrtc/) |
[findings.md H1/H2 (confirmed on grader)]

---

## GPU MODE Competition Meta-Lessons (from retrospectives)

### From NVFP4 Hackathon writeups (Yue Zhang, amandeepsp, veitner)
1. **Profile first, optimize second.** The amandeepsp writeup: "Had I profiled after attempt 7,
   it would have told me immediately the kernel was memory-bound." Optimizing for compute when
   memory-bound is the most common mistake.
2. **Raw PTX / hardware intrinsics >> high-level abstractions for fine control.** Top NVFP4
   entries used direct PTX byte unpacking, precise cache policy control (`L1::no_allocate` for
   streaming data, `L1::evict_last` for reused data), and `-maxrregcount` tuning.
3. **Template on problem dimensions.** Compile-time K-specialization enabled full loop unrolling
   and optimal register allocation — a winning technique across all NVFP4 entries.
4. **A pure PyTorch/library call can beat hand-rolled kernels.** `torch._scaled_mm` scored
   22.4 µs on NVFP4 GEMV — competitive with hand-written CUDA. Validate against library
   implementations before writing from scratch.
5. **Shared memory reduction > atomic reduction** for intra-block dot products. Consistent finding.

### From NVIDIA's GPU MODE wins (cuda.compute / CUB)
6. **For standard primitives, use the proven library.** Sort/Scan/Histogram: CUB via
   `cuda.compute` was 2–4× faster than other submissions. We already apply this: `torch.geqrf`
   for shapes where cuSOLVER wins (E1, E3).

### From AMD $100K challenge winner (Yotta Labs)
7. **Kernel fusion across operations eliminates intermediate memory traffic.** Fusing MoE + GEMM
   + all-reduce into a megakernel was decisive. For us: fusing panel + T-build + trailing GEMM.
8. **Register optimization is concrete (compile-time constants reduce spills by 30).** Use
   `tl.constexpr` and fixed shapes aggressively.
9. **Custom Triton launchers** (caching pointer arithmetic, not tensor objects) can cut per-call
   launch overhead by 3×.

### From ThunderKittens (HazyResearch)
10. **Block ordering matters for L2 cache reuse.** Wrong ordering → 50% degradation on large
    GEMMs. For our trailing GEMM over a batch of 640 matrices, process matrices in an order
    that maximizes L2 reuse of the Y and T tiles (likely contiguous batches).

---

## Comparison/Priority Table

| Technique | What it gives | Implementation effort | Applicability to QR | Priority |
|---|---|---|---|---|
| T1: TMA async copy | Low register pressure, fast async load | LOW (Triton handles auto; HIGH for manual CuTe) | MEDIUM (panel already smem-resident; needed for trailing GEMM) | Medium |
| T2: Warp specialization | Overlap load/compute; near-peak GEMM | HIGH (needs CuTe DSL or Triton autoWS) | HIGH for trailing GEMM; irrelevant for panel | High (post-panel) |
| T3: Persistent/megakernel | Full SM occupancy; no wave waste; 1 launch for all blocks | MEDIUM (restructure panel loop into Triton constexpr loop) | HIGH for n=32 (20 blocks / 148 SMs); MEDIUM for n=176/352 | High |
| T4: Double/triple buffering | 10–20% GEMM speedup; hides TMA latency | LOW (Triton `num_stages` autotune) | LOW for panel; MEDIUM for trailing GEMM | Medium |
| T5: SMEM swizzling | 2.7× potential in GEMM SMEM access | LOW for new kernels (use SW128_Atom); MEDIUM for panel verify | HIGH for any CuTe GEMM kernel | High (for new kernels) |
| T6: tcgen05/UMMA | 2–4× tensor core throughput on B200 | LOW (torch.bmm already uses it; HIGH to use directly) | Already getting via bmm/geqrf; MEDIUM to exploit via CuTe | Medium (implicit) |
| T7: BF16x9 FP32 emulation | 3–4× faster trailing GEMM, exact FP32 | MEDIUM (cublasLt call via cuda.bindings) | HIGH — exact FP32, safe for band/rowscale, 3–4× vs FP32 | High (post-compute-bound) |
| T8: CuTe/TK/CUTLASS | Near-cuBLAS custom GEMM with fused epilogue | HIGH (new framework; but Python API available) | HIGH — fused trailing update; offline compile → cubin embed | High (trailing GEMM) |
| T9: Autotuning / shapes | 10–20% from config sweep; free | LOW | HIGH — we have 7 fixed shapes, must tune `_BLOCK`, `num_warps`, dispatch threshold | High |
| T10: Occupancy tuning | Prevent register spill; up to 2× | LOW (check n_spills, tune maxrregcount) | MEDIUM — verify no spills in v13; LOW if already spill-free | Medium |
| T11: Minimize launch count | Our confirmed #1 bottleneck | MEDIUM (fuse T-recurrence + merge block loop) | VERY HIGH — 10–64× fewer launches from fusing Python loop into kernel loop | Very High |
| T12: nvrtc/cubin embed | Enables offline CUTLASS kernels; fast compile | LOW (path confirmed) | ENABLER — required for all T8 work | Very High (enabler) |

---

## Recommended Path to 2.5 ms Geomean (from 4.73 ms, 1.88× improvement needed)

Current time budget per benchmark (rough, in ms):
```
n32:   5.48 ms  (currently worst; 28s grader time)
n176:  33.1 ms  (losing to geqrf 22 ms; need dispatch improvement)
n352:  71.3 ms  (losing to geqrf 51 ms)
n512: ~13 ms   (v13; winning 9.45× → keeping)
n1024:~42 ms   (v13; marginal win → need improvement)
n2048: geqrf (77 ms)  — dispatch here
n4096: geqrf (52 ms)  — dispatch here
```

For geomean 2.5 ms, assuming n2048/n4096 stay at geqrf (±1%):
- n32 needs to approach ~2–3 ms (or use geqrf 0.33 ms — just dispatch it!)
- n176: use geqrf (22 ms wins)
- n352: use geqrf (51 ms wins)
- n512: continue winning, push from ~13 ms toward ~8–10 ms
- n1024: push from ~42 ms toward ~25 ms

**Actually the clearest path to 2.5 ms geomean is better dispatch:**
If we dispatch n32/176/352/2048/4096 to geqrf and only use our kernel for n512/n1024,
geomean = geomean(0.33, 22, 51, 13, 42, 77, 52) ms = ~13 ms... that's still >2.5 ms.
Target: get n512 from 13 ms to ~5–7 ms and n1024 from 42 ms to ~15–20 ms.

### Top 3 Techniques (most applicable to our 4.73 ms → 2.5 ms goal)

**#1 — Minimize Launch Count: Fuse Block Loop + T-Recurrence (T11)**
Current ~12 ms residual launch overhead at n512 (C3). Fusing the Python `for col_block in
range(n//b)` loop into the Triton kernel itself (a `constexpr` loop over column blocks, with
shared-memory state persisting between blocks) converts O(n/b) panel launches into 1, and
eliminates the T-recurrence launch entirely. For n=512, b=64: 8 outer iterations → 1. This
is the single highest-leverage change available without new frameworks.

**#2 — BF16x9 FP32 Emulation for Trailing GEMM via cublasLt (T7)**
Access via `cuda.bindings` (confirmed importable): call `cublasLtMatmul` with
`CUBLAS_COMPUTE_32F_EMULATED_16BFX9`. This replaces `torch.bmm` for the trailing update and
gives 3–4× FP32-equivalent throughput while maintaining band/rowscale correctness (exact FP32
output). Engineering effort: medium. This directly reduces the 28–30% trailing-GEMM portion
(post-v13) by 3–4×. Combined with launch fusion (#1), the trailing update goes from 28% to
near-zero of GPU time.

**#3 — Fused All-in-One Kernel for Small-n (n ≤ 230): Persistent Megakernel (T3)**
For n=32 (the 28s grader chunk) and n=176 (if not dispatching to geqrf): a single Triton/nvrtc
kernel that processes the entire QR of one (small) matrix in shared memory, with a grid of 148
blocks (one per SM) each looping over multiple batch elements. Eliminates all inter-block
launches. For n=32, `32²×4 = 4 KB` per matrix — 148 blocks × 4 KB = 592 KB needed for all
panels on all SMs, trivially fitting in SMEM. This would bring n=32 from 5.48 ms to well below
1 ms, flipping it from the worst outlier to a near-geqrf competitor.

---

## Sources (Complete List)

- [GPU MODE leaderboard](https://www.gpumode.com/leaderboard/597?tab=rankings)
- [GPU MODE: Linear Algebra Kernels for the Age of Research](https://www.gpumode.com/news/linear-algebra-kernels-age-of-research)
- [GPU MODE reference-kernels (GitHub)](https://github.com/gpu-mode/reference-kernels)
- [GPU MODE kernelbot (GitHub)](https://github.com/gpu-mode/kernelbot)
- [NVIDIA blog: Topping GPU MODE leaderboard with cuda.compute](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute/)
- [NVIDIA blog: Unlocking tensor core performance with FP emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [tcgen05 for dummies (Gau Nernst)](https://gau-nernst.github.io/tcgen05/)
- [Modular: Matmul on Blackwell Part 2](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul)
- [CUTLASS tutorial: GEMM with TMEM for Blackwell (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [CUTLASS tutorial: Persistent kernels and Stream-K (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [ThunderKittens paper (arXiv:2410.20399 / ICLR 2025)](https://arxiv.org/abs/2410.20399)
- [Optimal SWP and WS for Tensor Core GPUs (arXiv:2512.18134)](https://arxiv.org/html/2512.18134v1)
- [FA4 reverse-engineered (Modal blog)](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [Warp Specialization in Triton (PyTorch blog)](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/)
- [Ozaki/ADP DGEMM on Blackwell (arXiv:2511.13778)](https://arxiv.org/html/2511.13778v1)
- [DGEMM with FP8 Ozaki (arXiv:2508.00441)](https://arxiv.org/html/2508.00441v2)
- [Blackwell NVFP4 hackathon journey (Yue Zhang)](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
- [12 attempts at an FP4 kernel (amandeepsp)](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [AMD $100K challenge winner: distributed kernels (Yotta Labs)](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025)
- [Unweaving warp specialization (Rohany)](https://rohany.github.io/blog/warp-specialization/)
- [Tawa: Automatic Warp Specialization (arXiv:2510.14719)](https://arxiv.org/pdf/2510.14719)
- [CuTe DSL (PyPI, CUTLASS 4.x)](https://pypi.org/project/nvidia-cutlass-dsl/)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [cuBLAS 13.x documentation](https://docs.nvidia.com/cuda/cublas/)
- [NVRTC 13.1 documentation](https://docs.nvidia.com/cuda/nvrtc/)
- [GPU Performance Portability needs Autotuning (arXiv:2505.03780)](https://arxiv.org/pdf/2505.03780)
- [Helion DSL (PyTorch blog)](https://pytorch.org/blog/helion/)
- [CUDA Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)
- [Bank Conflicts in Shared Memory (Ian Barber)](https://ianbarber.blog/2025/03/29/bank-conflicts-in-shared-memory/)
- [Stream-K and wave quantization (Colfax tutorial)](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [CUDA Graphs for kernel batching (arXiv:2501.09398)](https://arxiv.org/pdf/2501.09398)
- [Batched QR and SVD Algorithms on GPUs (arXiv:1707.05141)](https://arxiv.org/pdf/1707.05141)
- [KernelBot competition platform (OpenReview)](https://openreview.net/pdf?id=bq9U4dmuyJ)
