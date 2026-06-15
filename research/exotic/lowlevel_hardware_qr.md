# Exotic / Low-Level / Hardware-Layer Approaches to QR Decomposition

> Research survey for the GPU-MODE QR competition (B200, batched square Householder QR, FP32).
> Focus: interesting-over-fast, but every entry flags the speed/accuracy verdict for our problem.
> Sources current as of June 2026.

---

## 0. Context anchor

Our problem: batched blocked-WY Householder QR, return `(H, tau)` in LAPACK SGEQRF format, FP32 output.
Bottleneck (from profiling): launch-bound panel kernel and now the trailing-update `bmm`  (~28-30% of
GPU after v13). Hard correctness gates: `20·n·eps32` (factor) and `100·n·eps32` (orthogonality).
Known killers of precision: the `band` and `rowscale` correctness cases (wide dynamic range).
The B200 has: FP32 = 80 TFLOPS, TF32 ≈ 2500 TFLOPS (31x), BF16 ≈ 2250-5000 TFLOPS (28-62x),
FP8 ≈ 10000 TFLOPS (125x), FP4 ≈ 20000 TFLOPS (250x).

---

## 1. Bit-Level Tricks: Fast Inverse Square Root and Friends (Quake-style)

### Idea
The Quake III fast inverse square root (`y = 0x5F3759DF - (i >> 1)`) works by exploiting that
IEEE 754 floats encode `log2(x)` approximately in their bit pattern. One integer subtract on the
reinterpreted bits gives a first-order approximation to `1/sqrt(x)`, refined by Newton-Raphson.
For Householder: the critical inner loop computes `tau = 2 / ||v||^2` and `v /= (v[0] + sign*||v||)`.
Both are `1/sqrt(...)` or `1/x` operations on a small vector, tempting targets.

### Why interesting
It is the canonical example of representation-aware numeric tricks. On GPU, each warp independently
computes a reflector norm; a bit-twiddled approximate rsqrt followed by one Newton step might save
a warp-reduce + division sequence. Also applicable to the sign bit: the Householder sign choice
`sign(x) = copysign(1.0, x)` is a single bit manipulation, and fusing it with the norm byte-tricks
could save a conditional branch per reflector.

### Modern reality
CUDA hardware provides `__frsqrt_rn(x)` and `rsqrtf(x)`, which are single-instruction hardware
rsqrt (IEEE-rounded to 1 ULP). The Quake trick was a *software* workaround for CPUs without a
hardware rsqrt. On GPU, the hardware instruction is faster AND more accurate than any bit twiddling.
The `copysign` / `signbit` trick for the Householder sign bit is still useful: `__int_as_float(
(__float_as_int(x[0]) & 0x80000000) | __float_as_int(sigma))` avoids a branch. This is a real
micro-optimization used in hand-written CUDA Householder kernels.

### Caveats
The sign trick produces one instruction instead of a predicated branch, but branch divergence in a
warp of same-sign elements is already zero-cost (all paths identical). Only worth doing if the
Householder column has a divergent sign distribution (rare in practice).

### Speed/accuracy verdict for us
**Not a lever.** Hardware rsqrt is already optimal. The copysign bit trick is worth 1-2 instructions
per reflector in a fused panel kernel but is noise compared to the global-memory traffic we're saving
with shared-memory residence (v13). File under "polish after correctness is locked."

**Links:**
- [Fast Inverse Square Root - Wikipedia](https://en.wikipedia.org/wiki/Fast_inverse_square_root)
- [fast-inverse-sqrt on modern hardware (Algorithmica)](https://en.algorithmica.org/hpc/arithmetic/rsqrt/)
- [CUDA libdevice rsqrt](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rsqrt.html)

---

## 2. Stochastic Rounding as a Hardware Feature

### Idea
Standard IEEE 754 uses round-to-nearest-even (RNE), which can cause systematic error accumulation
(the "Staircase" phenomenon) in iterative algorithms. Stochastic rounding (SR) instead rounds up or
down with probability proportional to the distance to each neighbor, which statistically cancels
errors across many operations. "Stochastic Rounding 2.0" (arXiv:2410.10517, 2025) formalizes
conditions under which SR provably improves convergence and accuracy beyond RNE, and calls for SR as
a standard hardware feature on GPUs.

### Why interesting
If the trailing GEMM were performed in BF16 with SR (instead of RNE), the accumulated error might
cancel enough to pass our correctness gates even on band/rowscale. SR is already present on:
Graphcore IPU, IBM floating-point units, AMD mixed-precision adders, Tesla D1, AWS Trainium.
NVIDIA H100/Hopper FP8 and Blackwell FP8/FP4 use per-tensor or per-block scaling (MXFP) which
reduces quantization error but is *not* stochastic rounding. NVIDIA has not publicly documented SR
in any Blackwell tensor core unit.

### Caveats
- No confirmed hardware SR on the B200 as of June 2026.
- Emulating SR in software (random perturbation before rounding) costs ~2x per op, not a speedup.
- SR helps most with iterative (convergent) algorithms; blocked QR trailing updates are not iterative
  — they are one-shot updates, where systematic round-off is less likely to cancel.

### Speed/accuracy verdict for us
**Exotic curiosity, not actionable.** The B200 does not expose SR control at the kernel level.
If NVIDIA adds SR to future architectures (likely given industry momentum), it could help BF16
trailing updates avoid band/rowscale failures — but that is not our problem today.

**Links:**
- [Stochastic Rounding 2.0 (arXiv:2410.10517)](https://arxiv.org/pdf/2410.10517)
- [NVIDIA B200 specs](https://inworld.ai/resources/nvidia-b200-gpu-cloud)

---

## 3. Alternative Number Systems: Posit, Logarithmic, Takum

### Idea
**Posits** (John Gustafson, 2017) use a variable-length exponent field ("regime" bits) that gives
more mantissa precision near ±1 and less at extremes. They eliminate ±Inf/NaN and have a unique
"quire" accumulator for exact dot products. **Logarithmic number system (LNS)**: stores the log of
a value, turning multiply into add and divide into subtract (but addition requires a table lookup).
**Takum** (2025): a newer tapered-precision format claiming "exceptional stability at low precision."

A 2025 paper (arXiv:2412.20268, ARITH 2025) evaluated posit, takum, and bfloat16 in LU, QR, and
GMRES solvers on real-world matrices: "tapered-precision posit and takum formats show better accuracy
in direct solvers and reduced iteration counts in indirect solvers." QR in particular benefited from
takum at 16-bit precision matching or exceeding FP32 in solution accuracy for many test matrices.

LNS has been applied to QR in FPGA hardware: a 2014 paper showed "highly efficient" LNS-QR in
terms of hardware complexity and accuracy, using a CORDIC-like approach for the log-add operation.

### Why interesting
Posit with a quire accumulator could make the Householder dot products (norm^2 = vT*v) exact,
eliminating cancellation error in the pivot step — a genuinely different precision story than
"just use higher precision."

### Caveats
- **No GPU hardware support.** No B200 posit instructions; any use requires software emulation
  (catastrophically slow, 10-100x overhead for each FP32 op).
- Posit arithmetic is not associative, so auto-vectorization / tensor-core use is impossible.
- Takum is even newer and has no hardware implementation.
- Qurer accumulator (512-bit per lane) does not exist in GPU register files.

### Speed/accuracy verdict for us
**Beautiful concept, completely impractical on GPU.** The accuracy story is intriguing for FPGA/ASIC
design, but has zero bearing on our B200 submission. Interesting for the "strangest" prize essay only.

**Links:**
- [Bfloat16, Posit, Takum in Sparse Solvers (arXiv:2412.20268)](https://arxiv.org/abs/2412.20268)
- [LNS QR Architecture (ResearchGate)](https://www.researchgate.net/publication/261040336_Low_Complexity_QR-Decomposition_Architecture_Using_the_Logarithmic_Number_System)

---

## 4. Block Floating Point and Microscaling MX Formats (MXFP8, MXFP6, MXFP4)

### Idea
Block floating point (BFP) stores a group of values with a shared exponent. Microsoft's MX
(Microscaling) formats, adopted by AMD, Intel, Meta, NVIDIA, Qualcomm et al. as an open standard,
extend BFP to blocks of 32 elements with a power-of-2 shared scale per block:
- **MXFP8** (E4M3/E5M2): 8-bit per element + 1 shared E8 scale per 32 → 8.25 bits effective
- **MXFP6** (E2M3/E3M2): 6-bit per element
- **MXFP4** (E2M1): 4-bit, one sign, two exponent, one mantissa bit per element

Blackwell B200 supports MXFP8, MXFP6, MXFP4 natively in 5th-gen Tensor Cores. MXFP8 on B200 =
10 PFLOPS, MXFP4 = 20 PFLOPS (vs 80 TFLOPS FP32 = 125-250x throughput ratio).

MXFP8 per-block scaling reduces quantization error 30-50% vs per-tensor FP8, which is why NVIDIA
introduced it: per-tensor FP8 (Hopper) was too inaccurate for training without careful tuning;
per-block gives a better dynamic range profile.

### Why interesting for QR
Our band/rowscale failure stems from wide dynamic range: one row may have norms 1e-6, another 1e+6.
MXFP8 per-block scaling adaptively rescales every 32 elements, so a column with mixed magnitudes
gets individualized scale per block — much better than one global scale. If MXFP8 trailing GEMMs
can pass band/rowscale, we unlock 125x the tensor-core throughput.

### Caveats
- PyTorch (torch 2.12) does not yet expose `torch.mm(..., dtype=torch.float8_e4m3fn, scale_mode='block')`
  for Blackwell MXFP8 in a call we can use from submission.py. Access requires either CUTLASS or
  TransformerEngine, neither of which is importable on the grader (H2 in findings.md).
- Even with access, 4+1 mantissa bits (MXFP8 E4M3) vs 7+1 (BF16) still risks trailing-update error
  on rank-deficient or very-ill-conditioned matrices, which our checker exercises via nearrank.
- Applying Ozaki-style error correction on top of MXFP8 would give accuracy, but the overhead
  (many tensor-core GEMMs) may eat most of the throughput gain.

### Speed/accuracy verdict for us
**High potential, medium friction.** If accessible via embedded CUDA/PTX (H2 path in findings.md:
nvrtc → driver launch), MXFP8 trailing update could be the 10x lever IF combined with a dynamic-range
detector to fall back to FP32 for band/rowscale. This is a real research direction for the competition.
The MX standard + Blackwell native support is a genuine 2025 development making this newly feasible.

**Links:**
- [Microscaling MX formats overview (FPRox)](https://fprox.substack.com/p/ocp-mx-scaling-formats)
- [Hardware for converting to MX format (arXiv:2411.03149)](https://arxiv.org/pdf/2411.03149)
- [MXFP4 training LLMs (arXiv:2502.20586)](https://arxiv.org/pdf/2502.20586)
- [Practical FP4 MoE on Hopper (arXiv:2603.02731)](https://arxiv.org/pdf/2603.02731)

---

## 5. Error-Free Transformations and Compensated Arithmetic (2Sum, Dekker, TwoProduct)

### Idea
An **error-free transformation (EFT)** is a pair of floating-point operations that computes BOTH the
rounded result AND the exact rounding error. `2Sum(a,b)` returns `(s, e)` with `s = fl(a+b)` and
`e = a+b-s` exactly (using 3 FLOP). `TwoProduct(a,b)` (Dekker/Veltkamp split + FMA) does the same
for multiplication: `(p, e)` with `p = fl(a*b)`, `e = a*b-p` exactly.

Kahan compensated summation chains 2Sum to get O(eps) total error in a sum regardless of length
(vs. O(n*eps) naive). For a dot product: **EFTdot** chains TwoProducts and 2Sums for exact dot
products in double the working precision. The Ozaki scheme (Section 6) is built on EFTs applied
to matrices.

At the scalar level inside a fused Householder kernel: the norm^2 = sum_i v_i^2 could be computed
with compensated summation to recover FP64-accurate norm from FP32 values. The reflector formula
`tau = 2 / (||v||^2)` then uses exact-norm input. This is distinct from precision "emulation" —
it is exact correction of accumulated rounding error.

### Why interesting
The key source of Householder error is cancellation in `v[0] += sign(x[0])*||x||` (the pivot update)
when `x[0]` and `||x||` are nearly equal (i.e., the first component dominates). A 2Sum-corrected
norm avoids this cancellation. FastTwoSum requires `|a| >= |b|` which is not always guaranteed;
a 2025 paper (arXiv:2601.17198) gives more general conditions for FastTwoSum as an EFT.

### Caveats
- At the **scalar/panel** level: 3 FLOP per addition vs. 1 → 3x overhead for the norm computation
  inside the Householder kernel. Given that our v13 panel kernel is still 29% of GPU time, this
  overhead is real. But the norm is O(n) and the trailing update is O(n^2), so the relative cost is
  small (~2% overhead for a 200-step panel).
- At the **GEMM/trailing update** level: applying EFTs to each scalar inside a GEMM completely
  defeats the tensor-core execution model (tensor cores do not expose per-lane error terms).
  EFTs for GEMMs are the Ozaki scheme (Section 6), not raw 2Sum.
- FMA (`__fmaf_rn`) in CUDA gives `fl(a*b+c)` with a single rounding, which already gives
  TwoProduct-level precision for a sum-of-two-products — this is the free EFT from hardware.

### Speed/accuracy verdict for us
**Panel-level EFT is a zero-cost option** (FMA is already used by the compiler in norm/dot loops).
Explicitly adding 2Sum to the norm computation in our panel kernel could make the pivot more robust
on nearcollinear/clustered matrices. **Zero speed cost, small accuracy insurance.** Worth a 5-line
addition to the fused panel kernel.

**Links:**
- [2Sum - Wikipedia](https://en.wikipedia.org/wiki/2Sum)
- [FastTwoSum conditions (arXiv:2601.17198)](https://arxiv.org/pdf/2601.17198)
- [Accurate Algorithms docs](https://accurate-algorithms.readthedocs.io/en/latest/ch04summation.html)

---

## 6. Ozaki Scheme / Error-Free Transformation for Matrix Multiplication (THE KEY ENTRY)

### Idea
The **Ozaki scheme** (Ozaki et al., 2012–2025) performs an error-free transformation of a
matrix product `C = A*B` into a *sum* of many matrix products that can each be done *without
rounding error* (or with bounded error). The steps:

1. **Split**: Decompose `A` into `k` low-precision matrices `A_1, ..., A_k` such that
   `A = A_1 + ... + A_k` exactly (as floating-point additions). Each `A_i` has a constrained
   mantissa range (a "slice" of the significand bits).
2. **Multiply**: Compute all `k^2` products `A_i * B_j` using fast low-precision tensor cores.
   Each product is exact (or nearly so) because both inputs have limited mantissa ranges.
3. **Sum**: Accumulate with carefully controlled rounding.

For FP32 accuracy from FP16/BF16 tensor cores, roughly **2-4 BF16 GEMMs** suffice (the exact
count depends on the input dynamic range). For FP64 from FP16/BF16, 7-9 slices are needed.

**For FP32 specifically (SGEMM emulation):**
- Ootomo & Yokota (2022, arXiv:2203.03341): dynamically split FP32 inputs into **two FP16
  components** (high bits + residual), perform 3 FP16 GEMM calls (high*high, high*low, low*high),
  accumulate. Achieves 51 TFlops on A100 vs 19.5 TFlops native FP32 SIMT: **2.6x speedup with
  exact FP32 accuracy.**
- **cuBLAS BF16x9**: NVIDIA's production implementation. "An FP32 value can be exactly represented
  as **three BF16 values**." The algorithm uses 9 BF16 tensor-core GEMMs (3x3 outer product of
  splits) with FP32 accumulation to recover exact FP32 results. Available in CUDA 13.0+ via
  `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` or env var `CUBLAS_EMULATE_SINGLE_PRECISION=1`. Requires
  compute capability 10.0 (Blackwell B200 = sm_100). Achieves **2-3x speedup over native FP32
  SGEMM** on B200/GB200 (ecTrans benchmark: 2.4x). Accuracy: same or better than native FP32.
- **Ozaki + cuSOLVER QR (November 2025, arXiv:2511.13778)**: Modified `cusolverDnGeqrf` where only
  the trailing matrix update GEMMs use emulated FP64 arithmetic (via Ozaki on INT8/BF16 tensor
  cores). On Blackwell GB200 and RTX Pro 6000 Blackwell: **up to 3.7x end-to-end QR speedup** while
  maintaining FP64-level accuracy. This is a direct proof-of-concept for our trailing-update problem.
- **Ozaki Scheme II (arXiv:2504.08009, 2025)**: Chinese Remainder Theorem variant; on GH200 achieves
  56.6-80.2 TFLOPS for FP64-accurate GEMM. Cleaner theoretical structure than original Ozaki.
- **ADP (Automatic Dynamic Precision, arXiv:2511.13778)**: Auto-estimates the number of slices needed
  via an "Exponent Span Capacity" (ESC) predictor — avoids over-splitting for well-conditioned
  matrices while guaranteeing accuracy for ill-conditioned ones.
- **SGEMM-cube (arXiv:2507.23387, 2025)**: On Ascend NPU (FP16 only), 2-split Ozaki achieves
  77% of theoretical FP32-equivalent peak = 65.3 TFLOPS.

### Direct relevance to our problem
Our trailing update `A -= Y @ (T^T @ (Y^T @ A))` is three batched GEMMs. With BF16x9, each GEMM
runs at 28x higher throughput and returns identical FP32 accuracy. Our current bottleneck (post-v13)
is the `bmm` (~28-30% of GPU time). At 2.4x speedup on B200, BF16x9 would turn 30% of our time
into 12.5%, cutting total time by ~17%. If we're at 3.4x overall geomean and the GEMM portion is
30%, that's: new geomean target ≈ 3.4 / (1 - 0.30 + 0.30/2.4) = 3.4 / 0.825 = **4.1x** — a real,
meaningful gain without touching panel or launch overhead.

**Critically: BF16x9 is accuracy-exact for FP32 (not approximate) so band/rowscale do not fail.**

### Caveats
- `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` is only available "on select architectures" requiring
  compute capability 10.0+ — the B200 (sm_100) qualifies.
- Must be enabled per-GEMM or globally. Using it globally on all GEMMs (including panel small ops)
  may have overhead; best to target only the three large trailing-update bmm calls.
- The 9 BF16 GEMM calls underlying BF16x9 are launched by cuBLAS automatically; no source change
  needed beyond setting the math mode. BUT cuBLAS's BF16x9 is for single-matrix SGEMM, not
  batched bmm. Torch's `torch.bmm` may not pick up BF16x9 math mode automatically.
- Alternatively: manual BF16 Dekker split + 3 bmm calls. The split: `A_hi = bf16(A)`,
  `A_lo = A - float(A_hi)`, then `C = A_hi @ B_hi + A_hi @ B_lo + A_lo @ B_hi` (3 BF16 bmm +
  2 FP32 adds). This is implementable in pure PyTorch/Triton.
- The grader's "stream" substring check does NOT affect math mode settings.

### Speed/accuracy verdict for us
**THE HIGHEST-PRIORITY PRECISION LEVER. Directly addresses our trailing-update bmm bottleneck
with zero accuracy cost.** Either via `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction(False)`
+ cuBLAS math mode, or via manual Dekker split in Triton. Should be tried immediately after v13
stabilizes. Expected gain: 15-25% on overall geomean at zero accuracy risk.

**Links:**
- [DGEMM Using Tensor Cores (Ozaki original, PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295351/)
- [Recovering FP32 accuracy from Tensor Cores (Ootomo & Yokota 2022, arXiv:2203.03341)](https://arxiv.org/abs/2203.03341)
- [Unlocking Tensor Core Performance with FP Emulation in cuBLAS (NVIDIA blog)](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [cuBLAS 13.3 documentation (BF16x9 API)](https://docs.nvidia.com/cuda/cublas/)
- [Guaranteed DGEMM Accuracy via Ozaki (arXiv:2511.13778)](https://arxiv.org/html/2511.13778)
- [Ozaki Scheme II (arXiv:2504.08009)](https://arxiv.org/abs/2504.08009)
- [Performance enhancement of Ozaki on INT matmul (arXiv:2409.13313)](https://arxiv.org/abs/2409.13313)
- [SGEMM-cube on Ascend (arXiv:2507.23387)](https://arxiv.org/html/2507.23387v2)
- [HPCwire: Ozaki Scheme primer (April 2025)](https://www.hpcwire.com/2025/04/17/have-you-heard-about-the-ozaki-scheme-you-will/)
- [Decompositional Factorizations with FP64 Emulation in INT8 (arXiv:2509.23565)](https://arxiv.org/pdf/2509.23565)

---

## 7. Randomized Householder QR / Sketching-Based Reflectors

### Idea
**Randomized Householder QR (RHQR)** (arXiv:2405.10923, SIAM J. Sci. Comput. 2025): Apply a
random sketching matrix `Ψ` to the working matrix before computing reflectors, so reflectors are
formed from the sketched matrix rather than the full one. The key property: RHQR is mathematically
equivalent to standard Householder QR applied to `ΨW` (a sketch), but the reflectors can be
aggregated without synchronizations, reducing the communication cost. Requires a single
synchronization per iteration instead of the usual per-step synchronization in standard Householder.
Cost: half the arithmetic of standard Householder QR, while maintaining "columnwise backward stable
factorization independently of condition number" even in half-precision arithmetic.

**Randomized Householder-Cholesky QR** (arXiv:2309.05868, 2025): Combines sketching with
CholeskyQR stability; nearly as fast as CholeskyQR2 on GPU for tall-skinny matrices.

### Why interesting
If sketching can halve the arithmetic of the panel factorization (currently 29% of GPU time in v13),
that's a 14% overall speedup. The orthogonality guarantee independent of condition number is
important for nearcollinear/clustered/nearrank cases.

### Caveats
- RHQR produces the QR of `ΨA` (sketched matrix), not `A`. For our competition, the checker
  verifies `Q = householder_product(H, tau)` and `R = triu(H)` against the ORIGINAL `A`.
  The output `(H, tau)` must satisfy the SGEQRF convention for the ORIGINAL matrix.
  Using a sketched reflector would produce H/tau that are WRONG for the original A.
- This method is designed for tall-skinny matrices; our benchmark has square matrices (`n×n` with
  batch). For square matrices, the reduction in synchronizations doesn't help as much.
- No GPU speedup figures for square batched matrices in the published papers.

### Speed/accuracy verdict for us
**Architecturally incompatible with the (H, tau) output contract.** RHQR changes WHAT reflectors
are computed, not just HOW. Could be relevant if the competition output were Q (explicit), but it
is not. Flag as mathematically interesting, not actionable.

**Links:**
- [Randomized Householder QR (arXiv:2405.10923)](https://arxiv.org/abs/2405.10923)
- [SIAM publication of RHQR](https://epubs.siam.org/doi/abs/10.1137/24M1674327)
- [Householder-Cholesky QR with sketching (arXiv:2309.05868)](https://arxiv.org/abs/2309.05868)
- [Tall-skinny QR on GPU with approximate reflectors (ResearchGate)](https://www.researchgate.net/publication/338812916_Tall-and-skinny_QR_factorization_with_approximate_Householder_reflectors_on_graphics_processors)

---

## 8. Analog / In-Memory / Photonic Matrix Engines

### Idea
**Analog in-memory computing (AIMC)**: Resistive crossbar arrays (RRAM, PCM, MRAM) store matrix
weights as conductances. A matrix-vector multiply is performed in O(1) time by Ohm's law + Kirchhoff's
current law in the physical circuit. Recent 2025 work (Nature Electronics) demonstrates a 16×16
analogue matrix solver matching 32-bit floating-point precision using scaled iterative refinement.

**Photonic engines**: Programmable unitary interferometer meshes (Mach-Zehnder interferometer arrays)
can perform unitary matrix-vector products at the speed of light. Native QR factorization on
photonic meshes (arXiv:2602.20701, Feb 2026): configures the mesh through local power routing steps
to produce QR factorization in O(N log N) physical operations vs O(N^3) digital. Complexity is
O(N log N) because the mesh has O(N log N) tunable elements in a butterfly topology.

A related 2026 paper computes SVD on photonic chips (arXiv:2602.18950).

### Why interesting
The photonic O(N log N) QR complexity is genuinely exotic — it does not correspond to any digital
algorithm. For large N, this would be a qualitative change in complexity class. Photonic in-memory
compute (pSRAM, PCM-weight banks) paper (arXiv:2503.18206) shows tensor decomposition with
"constant-time acceleration vs quadratic traditional compute."

### Caveats
- **Photonic chips operate on continuous (analog) signals**. Precision is limited by photon shot
  noise, thermal noise, and phase control accuracy — typically 8-12 effective bits in current
  implementations, far short of FP32. Our checker requires ~7 accurate digits (20·n·eps32).
- **Not programmable at GPU-kernel granularity.** Photonic mesh reconfiguration takes microseconds
  (EO modulator switching), while CUDA kernels operate at nanoseconds.
- **Optical QR only works on matrices that fit in the mesh.** Current chips: 64×64 elements max.
  Our n=4096 is 4000× too large.
- **Not available as a computational resource in any GPU cluster.** No API, no batched operation
  support, no CUDA integration.

### Speed/accuracy verdict for us
**Fascinating future technology, zero relevance to the competition.** Photonic QR is the most exotic
item in this survey and merits "strangest" prize mention, but cannot be submitted.

**Links:**
- [Native QR Factorization on Photonic Meshes (arXiv:2602.20701)](https://arxiv.org/pdf/2602.20701)
- [SVD on Photonic Chips (arXiv:2602.18950)](https://arxiv.org/pdf/2602.18950)
- [Accurate RRAM analogue matrix solver (Nature Electronics 2025)](https://www.nature.com/articles/s41928-025-01477-0)
- [Achieving high precision in analog IMC (npj Unconventional Computing 2025)](https://www.nature.com/articles/s44335-025-00044-2)

---

## 9. FPGA / Systolic CORDIC QR

### Idea
CORDIC (COordinate Rotation DIgital Computer) computes Givens rotations using only shift-and-add
operations, with no multiplier required. Systolic FPGA arrays pipe QR via a triangular array of
CORDIC processing elements (PEs): new columns enter from the top, rotated rows cascade diagonally,
and R emerges from the diagonal. A Virtex5 FPGA implementation achieves 246 MHz with ~24M updates
per second for 4×4 matrices (IEEE Xplore 7082764). Enhanced vectoring CORDIC: 37.7% less hardware,
76.8% less power, 1.8x speedup vs commercial implementation.

The core advantage: CORDIC uses no multipliers, making it extremely energy-efficient and hardware-
compact on FPGAs. Every Givens rotation is computed as a cascade of ~20 shift-adds.

### Why interesting
CORDIC Givens rotations are the FPGA/ASIC answer to the question "what if we have no multipliers?"
From a mathematical angle, Givens rotations are O(n^2) sequential operations on a matrix (vs WY
blocked Householder's O(n^2) flops + O(n^2) memory traffic). On GPU, CORDIC would require ~20
serial dependent operations per rotation element, making it dramatically slower than one FMA.

### Caveats
- CORDIC is a sequential algorithm (each step depends on the previous bit-shift). GPUs are bad at
  sequential scalar loops — no SIMD benefit from CORDIC.
- Givens rotation QR is O(n^2) sequential rotations for an n×n matrix; FPGA pipelines them across
  columns simultaneously. GPU cannot pipeline across the anti-diagonal chain.
- For 2025 GPU QR, CORDIC-Givens is never competitive with Householder+BLAS on a floating-point GPU.
- The fixed-point constraint (LNS or integer CORDIC) cannot meet our FP32 output format requirement.

### Speed/accuracy verdict for us
**Elegant on FPGA, wrong tool for GPU.** CORDIC QR is the right answer to a different question
(energy-minimal, no-multiplier hardware). Architecturally incompatible with GPU tensor-core execution.

**Links:**
- [Systolic FPGA CORDIC QR (IEEE Xplore)](https://ieeexplore.ieee.org/document/7082764/)
- [FPGA QR via Givens Rotation (IEEE Xplore)](https://ieeexplore.ieee.org/document/7110554)
- [MATLAB CORDIC QR tutorial](https://www.mathworks.com/help/fixedpoint/ug/implement-hardware-efficient-qr-decomposition-using-cordic-in-a-systolic-array.html)
- [Intel CORDIC QR-RLS Application Note](https://cdrdv2-public.intel.com/650506/wp_qrd.pdf)

---

## 10. Undervolting / Clock Glitching / Cosmic Rays

### Idea
**Undervolting**: Reducing GPU supply voltage below the manufacturer guardband causes stochastic
bit-flips when signal propagation is too slow to complete before the clock edge. Research (GreenMM,
2019) shows that for matrix multiplication specifically, safe undervolting achieves 8-15% power
savings with zero accuracy loss, because ABFT (Algorithm-Based Fault Tolerance) checksums can detect
and correct errors. Below the safe threshold: bit-flips begin.

**Algorithmic fault tolerance for QR (ABFT-QR)**: Add row and column checksums before each panel
and trailing update. After the operation, verify checksums. A flipped bit produces a detectable
mismatch; single-bit correction is possible. This was demonstrated on GPU heterogeneous systems
(CPU+GPU) for Cholesky, LU, and QR.

**Cosmic rays / soft errors**: High-energy particles occasionally flip a DRAM or register bit.
For a competition GPU running 24/7 in a data center, soft error rate for an H100 is ~1 error per
GPU per 6-12 months. For a 1-second computation, probability of a bit flip ≈ 10^-8 — negligible.

**Clock glitching (overclocking)**: Intentionally marginally overclocking causes intermittent
timing violations in the FP units. Similar to undervolting. The grader does not allow custom
clock settings; this is a hardware-control discussion only.

### Why interesting
GreenMM + ABFT demonstrates that "safe zone" undervolting is a real energy lever on data-center
GPUs without correctness risk. The ABFT-QR checksum technique (2018) could protect our kernel
from any stray soft errors AND enable future undervolting for power efficiency — a practical
near-term technique.

For pure weirdness: GPU undervolting has been used as a side channel for physical side-channel
attacks on cryptographic operations, because different voltage tolerances correlate with power traces.
Timing/voltage fault injection is an active area of hardware security research.

### Speed/accuracy verdict for us
**Not actionable for the competition** (we can't control GPU voltage; the grader environment
controls hardware settings). ABFT-QR checksums ADD overhead (~10-20% extra FLOP). But the concept
is useful background: ABFT proves that QR factorization has algebraic structure exploitable for
error detection — the same structure we could use for adaptive precision switching (if trailing
update is wrong, fall back to FP32 refinement).

**Links:**
- [GreenMM: GPU Matrix Multiply via Undervolting (ResearchGate)](https://www.researchgate.net/publication/333860651_GreenMM_energy_efficient_GPU_matrix_multiplication_through_undervolting)
- [ABFT for One-Sided Decompositions on GPU (ResearchGate)](https://www.researchgate.net/publication/326834691_Fault_Tolerant_One-sided_Matrix_Decompositions_on_Heterogeneous_Systems_with_GPUs)
- [Soft Error Sensitivity in GPU Matrix Multiply (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0026271420304558)
- [ABFT QR Full Checksum Diagram](https://www.researchgate.net/figure/Full-checksum-QR-decomposition-Algorithm-1-FT-xGEQRF2-1-input-panel-P-f-size-m_fig1_326834691)

---

## 11. Quantum and Quantum-Inspired QR / Orthogonalization

### Idea
**Quantum QR**: Li & Liu (Tsinghua, Phys. Rev. A 112, 032410, Sept 2025) present a quantum algorithm
for vector set orthogonal normalization and matrix QR decomposition with polynomial speedup over
previous quantum algorithms. QR on an N×N matrix: O(N² log² N) vs previous quantum results of
O(N^{2.5} polylog(N)). Based on the quantum Gram-Schmidt process via quantum phase estimation.

**Quantum-inspired classical algorithms**: These are classical *randomized* algorithms inspired by
quantum amplitude estimation. For low-rank matrices, they achieve exponential classical speedup
using sampling-based techniques (Tang 2018, "dequantization"). For QR specifically, no published
quantum-inspired classical algorithm outperforms standard methods for general (full-rank) matrices.

**Limitations of quantum speedup**: The Zi-Ming Li paper achieves polynomial quantum speedup over
previous quantum QR algorithms, but it is NOT a speedup over the best classical algorithms.
Standard classical Householder QR is O(n^3). The quantum algorithm scales as O(N^2 log^2 N) in
system dimension, which IS better than O(N^3), BUT this is the qubit complexity — the actual
operational count and gate depth are different, and quantum computers are currently far slower
than GPUs in practice.

### Why interesting
Quantum QR is a theoretically active area. The 2025 Phys. Rev. A result shows that quantum
computers can orthogonalize vectors in a way that is fundamentally different from classical
Householder (they use quantum phase estimation and amplitude amplification). For square matrices,
the quantum algorithm requires O(log N) quantum memory (log N qubits for the matrix index) — a
dramatic compression.

The "quantum-inspired" angle: Randomized SVD and sketching (Section 7) are the practical classical
descendants of quantum-inspired ideas, and they ARE useful for our problem.

### Caveats
- No quantum computer has sufficient qubits and coherence time to run QR on n=512 matrices.
- The quantum algorithm requires quantum random access memory (QRAM) which does not exist.
- "Polynomial quantum speedup over previous quantum QR" ≠ speedup over classical QR.

### Speed/accuracy verdict for us
**Theoretically fascinating, practically irrelevant for 2026 competition hardware.** The quantum
result is a beautiful existence proof. The best "quantum-inspired" techniques for us are already
captured in Section 7 (randomized sketching), which we ruled out for the output-contract reason.

**Links:**
- [Quantum Algorithm for QR Decomposition (arXiv:2412.19090)](https://arxiv.org/abs/2412.19090)
- [Phys. Rev. A publication](https://journals.aps.org/pra/abstract/10.1103/79h3-jspt)
- [Quantum-inspired for low-rank linear systems (arXiv:2508.13108)](https://arxiv.org/pdf/2508.13108)

---

## 12. Approximate Computing for QR

### Idea
Approximate computing deliberately allows computational errors in exchange for energy or speed.
For QR, "approximate Householder reflectors" (tall-skinny variant from ResearchGate 2020) approximates
the reflector vectors before computing norms — useful when only the column space of Q matters, not
exact reflectors. A 2020 HPDC paper ("High Accuracy Matrix Computations on Neural Engines") studies
QR on GPU tensor cores/TPUs by co-designing QR with its application (e.g., least squares), noting
that "matrix factorizations (QR, LU, Cholesky) are best co-designed with their applications" rather
than computed exactly as standalone primitives.

### Why interesting
The co-design idea is genuinely different: instead of computing exact LAPACK SGEQRF output and then
using Q and R downstream, design an algorithm that computes the USE of Q directly without materializing
exact reflectors. This could eliminate the Householder pivot step entirely for some applications.

### Caveats
- Our competition output format is `(H, tau)` specifically — the checker materializes Q from reflectors
  and checks residuals against the ORIGINAL A. Any approximation in H or tau propagates directly to
  orthogonality error. Approximate reflectors are ruled out by the output contract.
- "Approximate" in this context typically means relaxed error norms — but ours are hard gates (DQ if
  exceeded), not soft metrics.

### Speed/accuracy verdict for us
**Incompatible with the output contract.** Approximate QR is a valid research direction for downstream
applications but cannot be submitted.

**Links:**
- [High Accuracy QR on Neural Engines (HPDC 2020)](https://dl.acm.org/doi/10.1145/3369583.3392685)
- [Approximate Householder reflectors on GPU (ResearchGate)](https://www.researchgate.net/publication/338812916_Tall-and-skinny_QR_factorization_with_approximate_Householder_reflectors_on_graphics_processors)

---

## 13. Mixed-Precision Iterative Refinement for QR

### Idea
Iterative refinement for least squares / linear systems: compute a low-precision factor, then
refine with a correction step in higher precision. For QR specifically: perform all trailing updates
in BF16 (fast), then do one FP32 correction pass on the residual. This is standard for linear
systems (LAPACK's `dsgesv` / `zcgesv` use FP32 factor + FP64 refinement). For QR factorization
itself, it is less studied because the output is reflectors `(H, tau)`, not a solution vector.

The bfloat16 evaluation paper (arXiv:2412.20268) notes "bfloat16 provides a range similar to FP32
but with short mantissa and thus could be used for factorization" — with iterative refinement
recovering accuracy. Mixed-precision iterative refinement for least squares (arXiv:2406.16499)
extends this to QR-based least-squares solvers.

### Why interesting
This is the "do cheap BF16 trailing update + one FP32 correction" strategy that our findings.md B2
entry mentioned. In the least-squares context, one refinement step provably recovers FP64 accuracy
from an FP16 factor. For our QR output specifically: the H and tau tensors after BF16 trailing
update would have ~3-bit mantissa error; a refinement step would need to re-factorize the residual
matrix to correct the reflectors — expensive and non-standard.

### Caveats
- Iterative refinement for `(H, tau)` output (reflector form) is not directly analogous to
  iterative refinement for a linear system solution. The reflectors encode the factorization
  implicitly, and "refining" them requires understanding their structure deeply.
- One refinement pass = double the trailing update work. Only beneficial if BF16 is 3x cheaper
  AND one refinement recovers accuracy. BF16 is ~15x cheaper on B200 (from throughput ratios),
  so the math works — but the correctness argument for reflector refinement is not in the literature.
- Our existing B2/B3 findings already ruled out naive BF16 trailing update (8/19 correctness).
  Refinement would need to be designed carefully.

### Speed/accuracy verdict for us
**Theoretically promising but complex.** The Ozaki/BF16x9 approach (Section 6) is strictly better:
same accuracy, no iterative correction needed, already validated by NVIDIA. Iterative refinement
for reflector-form QR is an open research problem, not a ready tool.

**Links:**
- [Mixed Precision Algorithms in Numerical Linear Algebra (Acta Numerica)](https://www.cambridge.org/core/journals/acta-numerica/article/mixed-precision-algorithms-in-numerical-linear-algebra/43CA701BA29251B5790C653E66F46197)
- [Mixed Precision Iterative Refinement on GPUs (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/)
- [Bfloat16, Posit, Takum in Sparse Solvers (arXiv:2412.20268)](https://arxiv.org/abs/2412.20268)
- [Mixed Precision IR for Least Squares (arXiv:2406.16499)](https://arxiv.org/pdf/2406.16499)

---

## Summary Table

| # | Topic | Wackiness | Speed/Accuracy for us | Actionability |
|---|-------|-----------|----------------------|---------------|
| 1 | Quake rsqrt / copysign bit trick | Medium | Noise-level gain | Low (2-line polish) |
| 2 | Stochastic rounding hardware | High | Not available on B200 | None |
| 3 | Posit / LNS / Takum | Very high | No GPU hardware | None (FPGA/exotic) |
| 4 | MXFP8 / MX formats | High | Up to 10x GEMM speedup if accessible | Medium (needs CUDA path) |
| **5** | **Ozaki / BF16x9 EFT-GEMM** | **Medium** | **2-3x trailing GEMM, exact FP32 accuracy** | **HIGH — do next** |
| 6 | 2Sum EFT in panel kernel | Low | Panel norm robustness | Low (5-line addition) |
| 7 | Randomized Householder QR | High | Incompatible output contract | None |
| 8 | Photonic / analog IMC | Extremely high | Not available, wrong precision | None |
| 9 | FPGA CORDIC systolic QR | High | Wrong compute model for GPU | None |
| 10 | Undervolting / ABFT / cosmic rays | High | Not controllable in competition | None |
| 11 | Quantum QR | Extremely high | Hardware doesn't exist | None |
| 12 | Approximate QR | Medium | Incompatible output contract | None |
| 13 | Mixed-precision iterative refinement | Medium | Complex, Ozaki is strictly better | Low |

---

## Single Most Promising Pick

**Ozaki scheme / BF16x9 (Section 6), specifically via cuBLAS BF16x9 math mode or a manual
Dekker-split Triton bmm.**

The cuBLAS team has already done the hard work: `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` on B200 (sm_100)
gives 2-3x trailing GEMM speedup with guaranteed bit-identical FP32 accuracy. It requires zero
algorithm changes (same (H,tau) output), zero precision risk (no band/rowscale failures), and the
underlying cusolverDnGeqrf QR trailing update was validated at 3.7x end-to-end speedup on the same
Blackwell architecture (arXiv:2511.13778, Nov 2025). Our post-v13 GEMM fraction is ~30% of GPU
time; BF16x9 converts that to ~12%, lifting overall geomean from ~3.4x to ~4.1x — a meaningful
step toward the 7x+ target.

The manual route (Dekker split + 3× BF16 bmm in Triton) is the fallback if cuBLAS math mode doesn't
propagate through torch.bmm, and is entirely within the competition constraints (no banned words,
no external libraries).
