# QR algorithms for the competition — what's viable and where the speed is

## HARD CONSTRAINT: the output contract forces Householder/WY

The checker does `Q = torch.linalg.householder_product(H, tau)` and `R = triu(H)`.
So we must return **Householder reflector data**, not just Q and R.

This **rules out the "fast" tensor-core QR methods** as drop-in replacements:
- **CholeskyQR / CholeskyQR2** (`G = AᵀA`, `R = chol(G)`, `Q = A R⁻¹`): extremely
  tensor-core friendly (it's basically 2 GEMMs + a Cholesky), the state-of-the-art
  for throughput — but it produces Q and R explicitly, **not reflectors**. Converting
  a given Q into `(H, tau)` such that `householder_product(H,tau)=Q` is itself a
  Householder QR of Q (geqrf-cost). No cheap conversion exists. **Dead end for the
  output format**, even though it's the fastest math. (Worth re-checking only if the
  rules ever accept explicit Q/R.)
- **Classical/Modified Gram–Schmidt**: same problem — gives orthonormal Q columns,
  not reflectors; also less stable.

**Conclusion: we are committed to Householder QR in compact WY form.** The game is
making blocked Householder fast, not switching algorithm families.

## The blocked WY structure (what we already have)

For each column block of width `b`:
1. **Panel factorization** — `b` sequential Householder steps on the `(m × b)` panel.
   O(b·n²) work, but it's **GEMV/rank-1, memory-bound, sequential** → the bottleneck
   for throughput. This is the part that doesn't hit tensor cores.
2. **WY build** — `Y` (reflectors) + `T` (b×b upper-tri) so the block reflector is
   `I − Y T Yᵀ`. Cheap.
3. **Trailing update** — `A_trail −= Y (Tᵀ (Yᵀ A_trail))`. **The dominant FLOPs and
   the tensor-core opportunity** (batched GEMM / `bmm`).

The two levers: (a) shrink/parallelize the panel cost, (b) push the trailing update
through tensor cores at the lowest safe precision.

## Where the speed is, by shape regime (matches the baseline numbers)

The 7 ranked shapes split into two regimes that want different implementations:

### Small n, fits in shared memory → FUSE THE WHOLE QR PER MATRIX
B200 has up to ~228 KB shared memory per SM. A full `n×n` FP32 matrix fits when
`n² · 4B ≲ 228KB`, i.e. **n ≲ ~230**.
- **n=32** (4 KB/matrix) and **n=176** (124 KB/matrix): one threadblock per matrix,
  load the matrix into shared memory once, do the *entire* QR there, write back once.
  This is the **MAGMA batched-small-QR tactic** and it crushes cuSOLVER (which
  under-parallelizes the batch — see baseline: b40 n176 = 22ms for geqrf!). It
  eliminates global-memory round trips and per-step launches.
- **n=352** (495 KB/matrix): does NOT fit in shared memory → needs the blocked path.

### Large n → BLOCKED WY + BATCHED GEMM (TF32/BF16)
n=352, 512, 1024, 2048, 4096: keep the blocked WY structure, route the trailing
update through batched GEMM at TF32 (and explore BF16 + correction). For very small
batch + huge n (n=4096 b2), batch parallelism is gone, so this is about a single
large factorization — recursive panel helps (below).

### Recursive panel (Elmroth–Gustavson) for large n
The sequential panel can be made more tensor-core-friendly by **recursively halving**
it: split the panel into left/right halves, factor the left recursively, update the
right via GEMM, factor the right recursively. This converts tall-skinny GEMV work into
squarer GEMMs. Relevant for the large-n / small-batch cases (n=2048, 4096) where the
panel is a real cost and batch parallelism can't hide it.

## Practical plan

1. **Shape dispatch** (free, from `data.shape`): fused-shared-memory kernel for
   n ≲ 230; blocked WY + batched GEMM otherwise.
2. **Precision**: TF32 trailing GEMMs everywhere first (big free win — see
   [b200_hardware.md](b200_hardware.md)); BF16+correction later.
3. **Launch overhead**: CUDA graphs to collapse the many-small-kernel block loop
   (see [profiling.md](profiling.md)) — especially for small/medium n.
4. Keep panel reflector construction in FP32 (stability; it's cheap anyway).

## Sources
- [High Performance Householder QR on GPUs Using Tensor Cores (IEEE TPDS 2024)](https://dl.acm.org/doi/10.1109/TPDS.2024.3522776)
- [Batch QR Factorization on GPUs: Design, Optimization, and Tuning (ICCS 2022, Dongarra et al.)](https://www.iccs-meeting.org/archive/iccs2022/papers/133500064.pdf)
- [Applying recursion to QR factorization — Elmroth & Gustavson (IBM J. R&D 2000)](https://www.researchgate.net/publication/297856543_Applying_recursion_to_serial_and_parallel_QR_factorization_leads_to_better_performance)
- [Analysis of Randomized Householder-Cholesky QR with Multisketching (arXiv 2309.05868)](https://arxiv.org/abs/2309.05868)
- [Batched Triangular Dense LA Kernels for Very Small Matrices on GPUs (ACM TOMS)](https://dl.acm.org/doi/abs/10.1145/3267101)
