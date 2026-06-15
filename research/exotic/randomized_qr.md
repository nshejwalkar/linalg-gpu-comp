# Randomized & Probabilistic QR Algorithms — Exotic Survey

> **Research date:** June 2026. Focus: wacky / highly interesting algorithms in the
> randomized and probabilistic family. Each entry: core idea, why interesting,
> numerical caveats, **speed/relevance verdict** for batched square Householder QR
> on B200 returning `(H, tau)` in `torch.geqrf` compact LAPACK format.
>
> Context from CLAUDE.md: we are launch-bound (10k+ tiny kernels per iter); the big
> lever is kernel COUNT not FLOPs; we win on b640n512 but lose on small/large shapes;
> the output contract forces full Householder (CholeskyQR, Gram-Schmidt cannot be
> drop-ins because the checker calls `householder_product(H, tau)`). TF32 globally
> fails band/rowscale cases; our current geomean vs geqrf is ~1.5x with dispatch.

---

## 1. Randomized Householder QR (RHQR)

**Paper:** Grigori & Timsit, "Randomized Householder QR," arXiv:2405.10923 (May 2024).
Published SIAM J. Sci. Comput. 2025. See also HAL: hal-04156310v4.
[arXiv](https://arxiv.org/abs/2405.10923) | [SIAM](https://epubs.siam.org/doi/abs/10.1137/24M1674327)

**Core idea (2-3 sentences).**
RHQR applies a sketching matrix Ψ (e.g., SRHT) to the input W before running standard
Householder QR, producing the factorization of ΨW. Under mild probabilistic assumptions
on Ψ, RHQR inherits full Householder stability — columnwise backward stable,
well-conditioned basis, condition-number-free — while cutting the per-column cost
roughly in half. A reconstruction step (recRHQR) recovers a single-synchronization
variant (one synchronization per iteration), with the same total cost as Randomized
Gram-Schmidt.

**Why interesting.**
This is perhaps the most "pure" randomized Householder algorithm: the reflectors are
still proper Householder reflectors (they live in the original space), so the output
stays in LAPACK compact form. Unlike CholeskyQR2 variants, it works on ill-conditioned
matrices independently of condition number. Numerical experiments confirm it beats
Randomized Gram-Schmidt on the hardest inputs, even in half-precision with mixed
precision operations in the sketch step.

**Numerical caveats.**
Stability is probabilistic ("with high probability") but the probability of failure
is exponentially small in the oversampling parameter. The sketch must be a valid
subspace embedding (SRHT and Gaussian sketches both qualify). Half-precision sketch
step can be used without hurting the final FP32 basis quality.

**Speed / relevance verdict for us.**
INAPPLICABLE as a drop-in replacement. RHQR is designed for tall-skinny matrices
(m >> n), where you sketch to reduce the tall dimension before applying n Householder
steps. For square n×n matrices (our workload), sketching reduces n×n to k×n with
k < n, which degrades the problem rather than helping — you still have to do n
Householder steps on the sketched system. The "half the cost per column" claim applies
to tall-skinny reduction overhead, not the n×n square case. **Curious but not useful
for us directly.**

---

## 2. Randomized Householder-Cholesky QR with Multisketching (rand_cholQR)

**Paper:** Higgins, Szyld, Boman & Yamazaki, "Analysis of Randomized Householder-Cholesky
QR Factorization with Multisketching," arXiv:2309.05868 (2023). Published
*Numerische Mathematik* 157, 1695–1737 (2025).
[arXiv](https://arxiv.org/abs/2309.05868) | [Springer](https://link.springer.com/article/10.1007/s00211-025-01492-5)

**Core idea.**
rand_cholQR combines a sparse CountSketch S₁ (one ±1 per column, O(nm) FLOPs to apply)
with a dense Gaussian sketch S₂ (O(m⁴) but mapped to fast GEMM) as a "multisketching"
preconditioning step. The sketched matrix S₂S₁V undergoes Householder QR to produce a
well-conditioned triangular factor R₀; then Q₀ = VR₀⁻¹; then a single CholeskyQR pass
refines orthogonality. Orthogonality error is bounded at O(u) for ANY numerically full-rank
matrix with high probability, removing CholeskyQR2's κ²(A) ≤ ε⁻¹ restriction.

**Why interesting.**
The multisketching trick (chain two sketches to avoid expensive dense sketch while keeping
small output) is algorithmically elegant. On an A100 GPU with tall-skinny matrices it is
4% faster than CholeskyQR2 and 56.6% faster than shifted CholeskyQR3 — achieving near
CholeskyQR2 speed with unconditional stability.

**Numerical caveats.**
Designed exclusively for tall-skinny matrices (m >> n). The final output is a Q factor,
NOT Householder vectors — so the checker's `householder_product(H, tau)` path is
unavailable. CountSketch introduces sparse randomness that can fail on adversarially
structured inputs with probability ε(m) (controlled by oversampling). The O(m⁴) Gaussian
sketch step is tolerable only when m (column count) is small.

**Speed / relevance verdict.**
INAPPLICABLE for our square-matrix output contract. Even if we relaxed the output
format, the method targets tall-skinny (1000×100-style) not square (512×512). The
multisketching concept is scientifically beautiful but our checker needs LAPACK compact
`(H, tau)` Householder form — there is no cheap conversion from a Q factor. **Purely curious.**

---

## 3. Randomized Preconditioned Cholesky-QR (rpCholesky-QR)

**Paper:** Garrison & Ipsen, "A randomized preconditioned Cholesky-QR algorithm,"
arXiv:2406.11751 (June 2024).
[arXiv](https://arxiv.org/abs/2406.11751)
[Poster (mixed precision)](https://grad.ncsu.edu/wp-content/uploads/2025/04/garrisonposter25.pdf)

**Core idea.**
rpCholesky-QR computes a random sketch of a few rows (as few as 3n) of the input matrix
to form a randomized preconditioner P, then applies CholeskyQR to the well-conditioned
matrix A·P⁻¹. The key insight is that random subsampling of rows produces a preconditioner
that flattens the condition number from κ(A)² to κ(AP⁻¹) ≈ O(1) with high probability.
The deviation from orthonormality scales with κ(AP⁻¹) rather than κ(A)² (the CholeskyQR2
penalty).

**Why interesting.**
Standard CholeskyQR2 fails when κ²(A)·u > 1 (i.e., κ(A) > ~10⁸ for FP32). rpCholesky-QR
removes this restriction while remaining nearly as fast as CholeskyQR2 for tall-and-skinny
inputs. It doesn't break down for highly singular matrices. As few as 3n sampled rows
suffice for the preconditioner — making sketch cost essentially free.

**Numerical caveats.**
Still produces Q and R in standard form, not LAPACK compact Householder vectors. The
preconditioner build requires a QR or Cholesky solve on an n×n sketch, which is O(n³).
For our square n×n input matrices this preconditioning overhead is O(n³) — the same cost
as the full factorization, so no speedup.

**Speed / relevance verdict.**
INAPPLICABLE for our output contract and matrix shape. rpCholesky-QR outputs Q×R, not
`(H, tau)`. Even ignoring the format mismatch, the preconditioner idea is only beneficial
when κ(A) is large; our benchmarks have cond ∈ {1, 2} (near-identity conditioning).
**Purely curious.**

---

## 4. Randomized LU / XR Preconditioned CholeskyQR (rLHC, rxCholesky)

**Papers:**
- Fan, Guo & Lin, "A Novel Randomized XR-Based Preconditioned CholeskyQR Algorithm,"
  arXiv:2111.11148. [arXiv](https://arxiv.org/abs/2111.11148)
- Guan & Fan, "Randomized LU-Householder CholeskyQR," arXiv:2412.06551 (Dec 2024).
  [arXiv](https://arxiv.org/abs/2412.06551)

**Core idea.**
XR-CholeskyQR replaces the expensive QR-based preconditioner in standard preconditioned
CholeskyQR with either a randomized LU factorization (rLHC) or a randomized QR on a
sketch (rQRC), both much cheaper than a full QR. Specifically: sketch A to get SA
(small matrix), compute LU of SA (or QR of SA), extract the upper triangular factor
as preconditioner P, apply CholeskyQR to A·P⁻¹. The rLHC variant adds Householder
reflectors after CholeskyQR for an extra stabilization pass. Claimed: "more stable
and faster than all existing algorithms" on tall-skinny matrices.

**Why interesting.**
The key trick (use a *cheap* factorization of the sketch, not the full QR, as the
preconditioner) shaves the preconditioner cost from O(n³) to O(k·n²) where k is the
sketch size. The rounding error analysis extends to sparse matrices (Guan & Fan 2506.04208,
June 2025). The rLHC/SSLHC3 family guarantees stability even for ill-conditioned cases
that would defeat CholeskyQR2.

**Numerical caveats.**
All methods in this family output Q and R, not LAPACK compact Householder data.
The stability guarantees require applying a CholeskyQR2-like double-pass at the end
(SLHC3/SSLHC3 naming). GPU results reported by these papers are for tall-skinny A100
workloads.

**Speed / relevance verdict.**
INAPPLICABLE (output contract). However, the underlying insight — randomized LU of a
sketch is a cheap yet good preconditioner — is conceptually reusable. If the checker
were ever relaxed to accept Q×R output, rLHC would be a strong candidate.
**Curious; preconditioner idea filed for reference.**

---

## 5. CQRRPT — CholeskyQR with Randomization and Pivoting for Tall Matrices

**Paper:** Murray, Demmel et al., "CholeskyQR with Randomization and Pivoting for Tall
Matrices," arXiv:2311.08316. Published SIAM J. Matrix Anal. Appl. 2025.
[arXiv](https://arxiv.org/abs/2311.08316) | [SIAM](https://epubs.siam.org/doi/10.1137/24M163712X)

**Core idea.**
CQRRPT uses a randomized sketch to identify column pivots cheaply (no sequential norm
updates needed), then applies CholeskyQR with randomized preconditioning to the pivoted
matrix. Randomization serves dual roles: (1) fast pivot selection from a sketch, (2)
preconditioning to allow a single CholeskyQR pass instead of CholeskyQR2. Achieves
order-of-magnitude speedup over LAPACK DGEQP3 (pivoted QR) on CPU, while rivaling
unpivoted DGEQRF.

**Why interesting.**
Solving the main bottleneck of column-pivoted QR (sequential norm updates) with a
sketch-and-rank-reveal approach is elegant and practically impactful. This makes
rank-revealing QR nearly as fast as unpivoted QR for the first time at scale.
Available in the open-source RandLAPACK library.

**Numerical caveats.**
CPU-only benchmarks in the paper (Intel Xeon Gold 6248R). No GPU results. Outputs
pivoted Q×R, not LAPACK Householder compact form. Designed for tall matrices.

**Speed / relevance verdict.**
INAPPLICABLE (output format and matrix shape). The ranked benchmark has cond ∈ {1,2}
so pivoting adds no numerical value; and the output contract requires Householder data.
**Curious for low-rank / rank-deficient variants.**

---

## 6. Randomized Strong Rank-Revealing QR (SRRQR via Sketching)

**Paper:** "Randomized strong rank-revealing QR for column subset selection and low-rank
matrix approximation," arXiv:2503.18496 (March 2025).
[arXiv](https://arxiv.org/abs/2503.18496)

**Core idea.**
A classical strong RRQR factorization is performed on a sketch SA (much smaller matrix)
to select columns; these columns are then used as pivots for the full matrix. Strong RRQR
normally requires O(n³·k) work to identify good pivot columns, but sketching reduces this
to O(k·n²) where k (sketch height) << n. GPU implementations (RTX 3090, A100) achieve
up to 8.67× speedup over LAPACK DGEQP3 for FP32 on GPU.

**Why interesting.**
Strong RRQR guarantees that the R factor has "strong rank-revealing" properties —
the pivoted columns span the range well in a quantifiable sense, not just empirically.
Applying it via sketching makes this accessible at production scale. The GPU speedup
vs DGEQP3 is impressive.

**Numerical caveats.**
The sketch only approximates the pivots — the guarantee is "randomized strong RRQR"
(holds with high probability). Output is a pivoted R factor (or Q×R with pivoting);
not LAPACK compact Householder form. Relevant mainly for low-rank / rank-deficient inputs.

**Speed / relevance verdict.**
INAPPLICABLE for our dense well-conditioned square QR (no rank deficiency, no pivoting
needed, wrong output format). Fascinating for completeness of the RandNLA picture.
**Purely curious.**

---

## 7. HQRRP — Householder QR with Randomization for Column Pivoting

**Paper:** Martinsson, Quintana-Orti et al., "Householder QR Factorization With Randomization
for Column Pivoting (HQRRP)," FLAME Working Note #78, arXiv:1512.02671. Also: "Fast Parallel
Randomized QR with Column Pivoting," IEEE IPDPS 2018.
[arXiv](https://arxiv.org/abs/1804.05138)

**Core idea.**
HQRRP replaces the sequential Golub-style column-norm update in LAPACK DGEQP3 with a
sketch-based approach: at each block step, project the trailing matrix onto a random
subspace, compute norms of the sketch, and use those to select the pivot block. The
whole operation is blocked and BLAS-3 friendly. Because norm estimation via random
projection is cheap and highly parallelizable, HQRRP achieves near-DGEQRF speed while
still being rank-revealing.

**Why interesting.**
This is the first practical algorithm that makes column-pivoted QR BLAS-3 efficient.
It removes the core serial bottleneck of QRCP (length-n dot products one column at a
time) and replaces it with a random matrix–vector multiply that can run on tensor cores.
Parallel GPU speedup of up to 6.22× over standard GPU QRCP reported.

**Numerical caveats.**
Still outputs pivoted Q×R in standard (not LAPACK compact Householder) form. Designed
for tall or square matrices but the output format mismatch remains. The pivot quality
is empirically indistinguishable from exact column pivoting.

**Speed / relevance verdict.**
INAPPLICABLE (output format). But the idea of using a random sketch to replace expensive
sequential column-norm decisions is *directly applicable* as a sub-step if we ever
needed pivoted QR. **Interesting; file for potential future use.**

---

## 8. CountSketch as Sketching Primitive — High-Performance GPU Implementation

**Paper:** "A High Performance GPU CountSketch Implementation and Its Application to
Multisketching and Least Squares Problems," SC'25 Workshops, arXiv:2508.14209.
[arXiv](https://arxiv.org/abs/2508.14209) | [ACM DL](https://dl.acm.org/doi/full/10.1145/3731599.3767544)

**Core idea.**
CountSketch is a sparse random projection: one ±1 per column, so applying it costs
O(n) per column (SpMM). This paper demonstrates a high-performance GPU implementation
of CountSketch that makes it competitive with SRHT for sketching matrices before
randomized QR (specifically in the rand_cholQR / multisketching pipeline). On an H100
SXM5, CountSketch achieves near-peak SpMM throughput and in the multisketching pipeline
(CountSketch + Gaussian) outperforms pure Gaussian sketching in wall time.

**Why interesting.**
CountSketch has the lowest sketching cost of any standard embdedding: O(nm) for an m×n
matrix vs O(nm log n) for SRHT vs O(nm·k) for Gaussian. On GPU, the sparse matrix
multiply (SpMM) maps well to tensor-core-adjacent bandwidth operations. This makes the
sketch step essentially free relative to the QR step, allowing multisketching pipelines
to have near-zero overhead for the randomization phase.

**Numerical caveats.**
CountSketch requires sketch dimension p = O(m²) to be a valid subspace embedding (unlike
Gaussian/SRHT which need only p = O(m·log m)). In practice the multisketching trick
(CountSketch → Gaussian second stage) brings the second-stage sketch dimension down to
O(m). CountSketch does not directly provide the OSE guarantee alone.

**Speed / relevance verdict.**
INAPPLICABLE for our output contract (these pipelines output Q not Householder vectors).
However: if we ever implement a sketch-and-precondition approach for our trailing GEMM
(e.g., to use low precision safely), CountSketch is the cheapest randomizer available.
**Interesting as a building block; not a standalone algorithm for us.**

---

## 9. Subsampled Randomized Hadamard Transform (SRHT) as QR Sketching Kernel

**Background / survey:** Multiple papers including Woolfe et al. 2008, Ailon-Chazelle
2009; most recently used in RHQR (Grigori & Timsit 2024, §8 above) and randomized
GMRES orthogonalization. See [arXiv:2002.00864](https://arxiv.org/pdf/2002.00864) for
iterative sketching analysis.

**Core idea.**
SRHT = random sign flip (diagonal ±1 matrix D) → Walsh-Hadamard transform (FFT-like,
O(n log n)) → uniform column subsampling. This produces a near-optimal subspace embedding
in O(n log k) time rather than O(nk) for a Gaussian sketch. When used as Ψ in RHQR, it
allows the tall-skinny sketch step to run in O(mn log k) rather than O(mnk).

**Why interesting.**
SRHT achieves condition-number-free stability (as shown in RHQR's finite precision
analysis), nearly optimal embedding dimensions (k = O(m·log m) suffices), and benefits
from cache-friendly sequential memory access patterns (the Hadamard butterfly).
In half-precision on GPU, RHQR with SRHT sketch was shown to be stable even when the
sketch is computed in FP16.

**Numerical caveats.**
SRHT requires n to be a power of 2 (or padding). The O(n log n) constant factor on GPU
is sometimes larger than a dense matrix–vector product for moderate n (GPU latency
overhead on small SRHT). For m small (our case: m = n for square matrices) the SRHT
step brings no dimension reduction.

**Speed / relevance verdict.**
INAPPLICABLE for square matrices. SRHT's dimensionality reduction power is only useful
when the "tall" dimension is being reduced; for square n×n there is no tall dimension.
**Purely interesting as a sketching primitive.**

---

## 10. Randomized Gram-Schmidt (RGS) with Sketched Orthogonality Check

**Papers:**
- Balabanov & Grigori, "Randomized Gram-Schmidt process with application to GMRES,"
  SIAM J. Sci. Comput. (2021). [arXiv](https://arxiv.org/pdf/2011.05090)
- Jang et al., "Randomized Orthogonalization Process With Reorthogonalization,"
  Numerical Linear Algebra with Applications (2025). [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/nla.70029)
- de Damas, Grigori et al., "Randomized orthogonalization and Krylov subspace methods:
  principles and algorithms," arXiv:2512.15455 (Dec 2025). [arXiv](https://arxiv.org/pdf/2512.15455)

**Core idea.**
Instead of computing inner products ⟨q_j, v⟩ in full n-dimensional space during
Gram-Schmidt, compute them in a k-dimensional sketch (k << n). If the sketching
matrix Ψ is a good subspace embedding, the sketch inner products approximate the
full ones within O(u) — enabling orthogonalization in O(kn) rather than O(n²) per
step. A key result: RGS halves the computational cost of classical/modified Gram-Schmidt
while achieving orthogonality *independently of the condition number*, unlike MGS.

**Why interesting.**
Halving orthogonalization cost with no accuracy loss is remarkable. The loss of
orthogonality in RGS is O(u) (machine precision) regardless of κ(A), whereas modified
Gram-Schmidt degrades to O(κ(A)·u). The 2025 paper (Jang et al.) introduces RGS-L2
with reorthogonalization that further tightens orthogonality to near-unit roundoff
with deterministic cost O(1) extra passes.

**Numerical caveats.**
Output of RGS is Q (orthogonal matrix), not Householder vectors — so inapplicable to
our output contract. For batched GPU use, the sketch step (O(kn) per vector) must
itself be parallelized. The batch structure of our problem (640 independent square
matrices) maps poorly to the per-vector sketch paradigm.

**Speed / relevance verdict.**
INAPPLICABLE (output contract). Conceptually very interesting as a half-cost alternative
to Gram-Schmidt orthogonalization. **Purely curious for us.**

---

## 11. Sketch-and-Precondition for Least Squares (Blendenpik / LSRN lineage)

**Papers:**
- Avron et al., "Blendenpik: Supercharging LAPACK's Least Squares Solver," SIAM J. Sci.
  Comput. 2010.
- Meng & Mahoney, "LSRN: A parallel iterative solver achieved by randomized preconditioning,"
  SIAM J. Sci. Comput. 2014.
- "GPU-Parallelizable Randomized Sketch-and-Precondition for Linear Regression using
  Sparse Sign Sketches," arXiv:2506.03070 (May 2025). [arXiv](https://arxiv.org/pdf/2506.03070)
- "Are Sketch-and-Precondition Least Squares Solvers Numerically Stable?", arXiv:2302.07202.
  [arXiv](https://arxiv.org/pdf/2302.07202)

**Core idea.**
Sketch-and-precondition solves min_x ||Ax - b|| by: (1) compute SA (S = sparse sign
sketch), (2) QR-factor SA to get R, (3) use P = R⁻¹ as preconditioner, (4) run LSQR on
(AP)z = b, x = Pz. The preconditioner from the sketched QR is nearly as good as a full
QR preconditioner but costs O(nm/ζ) to apply (sparse sketch) instead of O(nm). The 2025
GPU paper achieves substantial speedups over direct QR on overdetermined systems with
m = 10⁵–10⁶ rows.

**Why interesting.**
The insight that a sketch of SA provides almost as good a preconditioner as the full QR
of A (and is O(m/ζ) cheaper to compute) is the intellectual heart of all randomized
least-squares algorithms. Stability caveat: the most-used form is NOT backward stable for
ill-conditioned problems — Meier et al. 2023 showed that unpreconditioned LSQR on the
preconditioned system (AP)z = b is required for backward stability.

**Numerical caveats.**
The method is for overdetermined least squares (m >> n), not square systems. For square
A (our benchmark), SA is underdetermined (rank(SA) < n); preconditioning degenerates.
Runtime advantage vanishes for square inputs.

**Speed / relevance verdict.**
INAPPLICABLE for square n×n QR (different problem class). **Purely curious.**

---

## 12. RMGSQR — Tensor Core-Accelerated Randomized Modified Gram-Schmidt QR

**Paper:** "High Accuracy Low Precision QR Factorization and Least Square Solver on GPU
with TensorCore," arXiv:1912.05508 (2019). [arXiv](https://arxiv.org/abs/1912.05508)
[ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1912.05508)

**Core idea.**
RMGSQR uses a recursive divide-and-conquer structure on the columns of A, applying
modified Gram-Schmidt within each recursion level and using FP16 tensor-core GEMM for
the dominant panel operations. A hand-written CAQR (Communication-Avoiding QR) panel
fits in 256×32 GPU shared memory tiles. The recursive structure turns the O(n³) work
into a sequence of tensor-core GEMMs plus small GEMV-like operations.

**Why interesting.**
Reported speedups of 2.9× to 14.7× over cuSOLVER SGEQRF are impressive. For square
matrices specifically, the 2.9× figure was measured. The approach exposes tensor cores
to QR without requiring a randomized algorithm — it's a precision-reduction trick at
the QR level, not a sketching trick. The CAQR panel idea (shared-memory resident columns)
directly maps to what we implemented in v13 of our submission.

**Numerical caveats.**
Uses half-precision FP16 internally → "slightly lower accuracy" than FP32 SGEQRF.
The authors note "directly solving with our low precision QR may not lead to sufficient
accuracy," requiring iterative refinement for high-accuracy LS. For our competition:
the 20n·eps32 orthogonality gate and band/rowscale correctness cases would likely fail
with FP16 internal arithmetic. Not randomized in the strict sense — more a mixed-precision
structured approach.

**Speed / relevance verdict.**
POTENTIALLY RELEVANT conceptually. The shared-memory panel tile (our v13) is the same
architectural insight. The "recursive MGS + tensor-core trailing update" structure could
be adapted if we use FP16 internally but return FP32 Householder data — but our
findings.md B1-B4 already document that TF32 fails band/rowscale, so FP16 internal
would fail worse. **Architecturally interesting; precision route blocked by our gate.**

---

## 13. Probabilistic Rounding Error Analysis of Householder QR

**Paper:** Connolly & Higham, "Probabilistic Rounding Error Analysis of Householder QR
Factorization," SIAM J. Matrix Anal. Appl. (2023).
[Manchester preprint](https://eprints.maths.manchester.ac.uk/2865/1/paper.pdf)
[SIAM](https://epubs.siam.org/doi/10.1137/22M1514817)

**Core idea.**
The standard worst-case backward error for Householder QR is O(mn·u). Using a
probabilistic model where rounding errors are mean-independent and mean-zero (which
stochastic rounding guarantees), the bound improves to O(√(mn)·u) with high probability.
The proof uses matrix concentration inequalities (Bernstein-type). The square-root
improvement applies to the full QR algorithm including two-sided transformations.

**Why interesting.**
This paper provides the theoretical basis for why running Householder QR with *stochastic
rounding* (rather than round-to-nearest) achieves better numerical behavior: the error
accumulates like a random walk (√n growth) rather than worst-case (n growth). For
n = 4096, this is a 64× tighter error bound, meaning that FP16 arithmetic with stochastic
rounding could theoretically achieve errors comparable to FP32 round-to-nearest.

**Numerical caveats.**
The probabilistic bound is "with high probability" (not worst-case). Stochastic rounding
is not natively supported in CUDA/hardware — it must be emulated (each fp32 operation
costs ~3-5× extra ops to round stochastically). The numerical experiments show the actual
errors for stochastic rounding are "virtually identical" to round-to-nearest in practice,
suggesting the theoretical advantage does not manifest for reasonable n in FP32.

**Speed / relevance verdict.**
INTERESTING THEORETICALLY BUT NOT ACTIONABLE. Hardware stochastic rounding (available
in some research GPUs like Graphcore's IPU) would be needed for a real speedup.
On B200, emulating stochastic rounding would slow us down, not speed us up. **Purely curious.**

**Stochastic Rounding 2.0 (Drineas & Ipsen 2024):** arXiv:2410.10517 — advocates SR
as a foundational tool for complexity analysis, but doesn't provide GPU speedup numbers.
The SR error bound improvement (√n vs n) is the key theoretical tool for mixed-precision
analyses of iterative refinement. [arXiv](https://arxiv.org/abs/2410.10517)

---

## 14. Ozaki Scheme — FP64-Accurate GEMM via FP8 Tensor Cores

**Papers:**
- Ootomo et al., "DGEMM without FP64 Arithmetic — Using FP64 Emulation and FP8 Tensor
  Cores with Ozaki Scheme," arXiv:2508.00441 (2025). [arXiv](https://arxiv.org/abs/2508.00441)
- "Ozaki Scheme II: A GEMM-oriented emulation using integer modular technique,"
  arXiv:2504.08009 (2025). [arXiv](https://arxiv.org/pdf/2504.08009)
- Related application to QR: "High Accuracy Low Precision QR Factorization..." (1912.05508)
  reports "3.7× end-to-end QR speedup" on GPU using Ozaki-emulated GEMM.
- [GEMMul8 GitHub](https://github.com/RIKEN-RCCS/GEMMul8)

**Core idea.**
The Ozaki scheme splits each FP64 input into several lower-precision slices (3-5 slices
for FP64 accuracy, 2 slices for FP32 accuracy), performs many fast FP8/FP16/INT8
tensor-core GEMMs (one per pair of slices), and reconstructs the high-precision result.
The magic: on Blackwell B200, FP8 tensor cores run at ~80 TFLOPS while FP64 ALUs run at
~5 TFLOPS — a 16× hardware advantage that the Ozaki splitting can partially capture.
For B200 with FP8: 61 TFLOPS in fast mode (13 moduli), 65 TFLOPS in accurate mode
(12 moduli) for FP64-accurate results, vs hardware FP64 at ~5 TFLOPS.

**Why interesting.**
This is NOT a randomized algorithm — it is deterministic and bit-exact under certain
conditions. But it belongs in this survey because it is a *surprising/exotic* approach
that uses hardware designed for AI (FP8 tensor cores) to accelerate scientific FP64 GEMM
by 12-16× on consumer GPUs (RTX 4090: 16×, B200: ~2.3× on complex GEMM, up to 80 TFLOPS
on Blackwell with FP8). The QR trailing update (our biggest GEMM) is an obvious target.

**Numerical caveats.**
Not randomized — probabilistic quality is not an issue. However: (a) requires 3-5×
more kernel launches than a single GEMM call; (b) accumulation errors in the reconstruction
step require careful implementation; (c) for FP32 accuracy (not FP64), only 2-3 slices
are needed, so the overhead is lower. On B200, native FP32 tensor cores (not FP64 ALUs)
already run fast; FP32 GEMM on B200 runs at ~1 PFLOPS, so the Ozaki FP8 approach would
only help if we need FP64 accuracy from FP8 — not our case.

**Speed / relevance verdict for us.**
MARGINALLY RELEVANT. Our problem is FP32-in, FP32-out. The B200's FP32 tensor cores
already operate at ~1 PFLOPS, making native FP32 GEMM optimal. The Ozaki scheme helps
when you need FP64 accuracy from FP8 hardware. For us, it could in principle let the
trailing update run faster by computing in FP8 + reconstruction to FP32 — but the
reconstruction overhead (extra kernel launches) would likely cancel the speed gain given
that we're already launch-bound (C1 in findings.md). **File for future reference if
we ever become FLOP-bound on the trailing GEMM.**

---

## 15. Anatomy of Column-Pivoted QR Decomposition (Randomized + GPU)

**Paper:** "Anatomy of High-Performance Column-Pivoted QR Decomposition," arXiv:2507.00976
(July 2025). [arXiv](https://arxiv.org/abs/2507.00976)

**Core idea.**
A unified framework for QRCP implementations provides user-controlled choices for core
subroutines (sketch type, pivot selection, trailing update method). On H100, the best
variant achieves ~65% of cuSOLVER unpivoted QR performance. On CPU (dual EPYC 9734),
achieves 100× speedup over LAPACK DGEQP3.

**Why interesting.**
The first systematic benchmark and analysis of the full space of design choices for
randomized QRCP on modern hardware. The "65% of unpivoted cuSOLVER QR on H100" figure
is particularly relevant: it shows that randomized column pivoting has become cheap
enough to be nearly free compared to the factorization itself.

**Numerical caveats.** Pivoted QR outputs a different format from what we need (permuted
factors). No batched QR results.

**Speed / relevance verdict.** INAPPLICABLE (output contract, no batching).
**Useful for the day column pivoting is needed.**

---

## 16. Single-Pass / Pass-Efficient Randomized Algorithms

**Papers:**
- Halko, Martinsson, Tropp, "Finding Structure with Randomness," SIAM Review 2011.
  [arXiv](https://arxiv.org/pdf/0909.4061)
- Tropp et al., "Streaming low-rank matrix approximation," SIAM J. Sci. Comput. 2019.
- "Pass-Efficient Randomized Algorithms for Low-Rank Matrix Approximation Using Any
  Number of Views," arXiv:1804.07531.

**Core idea.**
Single-pass algorithms compute random sketches Y = AΩ and Z = Aᵀ×Ψ in one pass over
A (reading each element once), then reconstruct a low-rank approximation from Y and Z
without re-reading A. For streaming matrices (data arrives once), this is the only option.

**Why interesting.**
Eliminates re-reading the matrix — critical for out-of-memory or streaming scenarios.
The single-pass QR approximation (approximate Q from Y = AΩ via QR of Y, then approximate
R from Z) is elegant and achieves near-optimal error with one pass.

**Numerical caveats.**
The approximation error is O(σ_{k+1}) — the (k+1)-th singular value — not exact.
This produces a low-rank *approximation* to QR, not an exact factorization. Completely
inapplicable to exact (full-rank) QR. Not suitable for batched exact factorization.

**Speed / relevance verdict.**
INAPPLICABLE (approximate, not exact; wrong problem class). **Purely curious.**

---

## 17. Monte Carlo / Walk-on-Equations Linear Algebra

**Papers:**
- "On Advanced Monte Carlo Methods for Linear Algebra on Advanced Accelerator Architectures,"
  arXiv:2409.03095 (2024). [arXiv](https://arxiv.org/html/2409.03095v1)
- "Novel Monte Carlo Algorithm for Linear Algebraic Systems," Springer 2024.

**Core idea.**
Monte Carlo algorithms for linear algebra solve Ax = b or compute matrix functions by
sampling random walks on the matrix graph. The "Walk on Equations" algorithm treats each
equation as a node, samples paths stochastically, and accumulates contributions to
estimate the solution. Each sample is independent, enabling massive parallelism.

**Why interesting.**
Embarrassingly parallel — every random walk is independent. Theoretically appealing for
very large sparse systems. Recent work targets GPU accelerators explicitly, showing near-
ideal scaling.

**Numerical caveats.**
These methods produce *statistical estimates* of the solution with variance that decreases
as O(1/√N_samples). For QR factorization specifically, they produce estimates of individual
matrix elements, not the structured triangular-reflector form needed. Convergence requires
many samples for high accuracy (accuracy O(1/√N) means N = 10⁸ samples for 4 digits).

**Speed / relevance verdict.**
COMPLETELY INAPPLICABLE for exact dense QR. These methods are designed for solving
very large sparse linear systems, not dense factorization. **Purely curious / exotic.**

---

## 18. RandCholeskyQR2 Error Analysis for Sparse Matrices

**Paper:** Guan & Fan, "Rounding error analysis of randomized CholeskyQR2 for sparse
matrices," arXiv:2506.04208 (June 2025).
[arXiv](https://arxiv.org/abs/2506.04208)

**Core idea.**
Proves that rand_CholeskyQR2 (one sketch pass + one CholeskyQR2 pass) is stable for
sparse matrices, introducing a new matrix norm tailored to sparsity. The analysis
shows that the orthogonality error can be bounded independently of matrix density.

**Why interesting.**
First quantitative stability analysis for randomized CholeskyQR on sparse inputs.
Extends the rand_cholQR theory to a setting where sparse sketches (CountSketch) are
particularly natural.

**Numerical caveats.**
Still outputs Q×R, not Householder form. Designed for tall-skinny sparse matrices.

**Speed / relevance verdict.**
INAPPLICABLE. Sparse inputs are only in our correctness suite (not the ranked benchmark).
**Purely curious.**

---

## 19. Randomized Krylov-Schur Eigensolver with QR as Subroutine

**Paper:** de Damas & Grigori, "Randomized Krylov-Schur eigensolver with deflation,"
arXiv:2508.05400 (2025). [arXiv](https://arxiv.org/pdf/2508.05400)

**Core idea.**
Randomized Krylov-Schur (rKS) replaces the deterministic QR step in Krylov-Schur
eigensolvers with a sketch-orthogonalization process (sketch-based Gram-Schmidt or
RHQR). The low-dimensional Schur factorization is computed in sketch-space, enabling
single-synchronization iterations. This uses QR as a *subroutine*, not as a primary
operation.

**Why interesting.**
The QR step inside Krylov-Schur is typically a tiny square matrix QR (k×k where
k = restart dimension ~50-200) — precisely our problematic regime. The randomized
orthogonalization keeps this cheap while maintaining stability.

**Numerical caveats.**
This is an eigenvalue algorithm; the QR subroutine produces orthogonal bases for
restart, not LAPACK compact Householder data. Output format mismatch.

**Speed / relevance verdict.**
INAPPLICABLE directly, but the tiny-square QR sub-problem (k = 32-200) is exactly
our n=32 and n=176 shapes. If a custom randomized micro-QR kernel were viable for
these shapes, the sketch-based ideas here could inspire an approach. **Mildly interesting.**

---

## Summary Table

| # | Algorithm | Family | Output H/tau? | Square n×n? | Batched? | Speed verdict |
|---|-----------|--------|:---:|:---:|:---:|--------------|
| 1 | RHQR (Grigori-Timsit) | Rand Householder | Yes (H,tau) | No (tall-skinny) | No | Inapplicable |
| 2 | rand_cholQR multisketching | Rand Cholesky-QR | No (Q only) | No | No | Inapplicable |
| 3 | rpCholesky-QR | Rand precond. | No (Q only) | No | No | Inapplicable |
| 4 | rLHC / XR-CholeskyQR | Rand LU precond. | No (Q only) | No | No | Inapplicable |
| 5 | CQRRPT | Rand pivot+Cholesky | No (pivoted Q×R) | No | No | Inapplicable |
| 6 | Rand strong RRQR | Rand rank-reveal | No (pivoted Q×R) | No | No | Inapplicable |
| 7 | HQRRP | Rand column pivot | No (pivoted Q×R) | Maybe | No | Inapplicable |
| 8 | CountSketch GPU | Sketching primitive | N/A | N/A | N/A | Building block |
| 9 | SRHT | Sketching primitive | N/A | No | N/A | Building block |
| 10 | RGS sketched orthog. | Rand Gram-Schmidt | No (Q only) | Maybe | No | Inapplicable |
| 11 | Sketch-and-precond. (Blendenpik/LSRN) | LS solver | No | No | No | Inapplicable |
| 12 | RMGSQR (tensor core) | Mixed-prec Rec QR | Maybe adaptable | Yes | Partial | Interesting |
| 13 | Prob rounding analysis | Theoretical | N/A | Yes | Yes | Theoretical |
| 14 | Ozaki Scheme (FP8 GEMM) | Deterministic MP | N/A (GEMM sub) | Yes | Yes | Marginally rel. |
| 15 | QRCP anatomy (2025) | Rand QRCP | No | Maybe | No | Inapplicable |
| 16 | Single-pass rand QR | Approx. LRA | No (approx only) | No | No | Inapplicable |
| 17 | Monte Carlo walk-on-eqns | Probabilistic | No | No | Yes? | Inapplicable |
| 18 | RandCholeskyQR2 sparse | Rand Chol-QR | No | No | No | Inapplicable |
| 19 | Rand Krylov-Schur | Rand eigensolver | N/A (subroutine) | Yes (small k) | No | Mildly interesting |

---

## Overarching Observation: The Output Contract Wall

**Every randomized QR algorithm in the modern literature (2021–2026) targets tall-skinny
matrices and outputs Q in explicit form.** The LAPACK compact `(H, tau)` Householder format
is almost completely absent from the RandNLA literature. The one algorithm that preserves
Householder structure is RHQR (algorithm 1 above) — but it is designed for tall-skinny
and does not help for square matrices.

This means:
1. No existing randomized algorithm can be *directly substituted* into our submission.
2. The "randomization" lever for our competition must be injected *inside* the standard
   Householder pipeline, not as a replacement for it.
3. The most promising injection points are:
   (a) **Randomized panel column-norm estimation** (HQRRP-style) to potentially skip
       expensive max-norm scans in pivoted variants — but our benchmark uses cond ∈ {1,2}
       so pivoting is not needed.
   (b) **Ozaki-scheme-style GEMM** for the trailing update, if we ever become FLOP-bound.
   (c) **Sketch-based preconditioning** of the panel to reorder Householder steps —
       unexplored in the literature for exact square QR.

---

## Most Promising Pick for Speed

**Conclusion: No randomized algorithm is directly applicable.** The closest to useful is:

**Conceptual winner: The Ozaki Scheme (FP8 GEMM emulation) applied to the WY trailing update.**

Reasoning:
- Our v13 profiling shows bmm accounts for 28-30% of GPU time post-smem panel.
- The Ozaki scheme on B200 FP8 achieves ~12-16× over native FP64; for FP32 (2-3 splits),
  it could plausibly achieve 2-4× over native FP32 tensor-core GEMM.
- However: (a) our trailing GEMM is already FP32 tensor-core-accelerated via cuBLAS bmm,
  which runs at ~1 PFLOPS; (b) adding 2-3 kernel launches per GEMM call would worsen our
  launch-bound bottleneck; (c) band/rowscale correctness gates may fail if FP8 accumulation
  introduces bias.

**Final verdict: None of the randomized QR algorithms are speed-promising for our specific
competition setup.** The entire family assumes tall-skinny matrices or produces non-Householder
output. The exotic technique closest to actionable is the Ozaki GEMM scheme, but we'd need
to be FLOP-bound first (we are not). Our existing path (shared-memory panel kernel, better
WY recurrence, shape-dispatch) remains the right approach.

---

## Sources

- [arXiv:2405.10923 — Randomized Householder QR (Grigori & Timsit 2024)](https://arxiv.org/abs/2405.10923)
- [arXiv:2309.05868 — rand_cholQR with Multisketching (Higgins et al. 2023/2025)](https://arxiv.org/abs/2309.05868)
- [Springer Numerische Mathematik — rand_cholQR published 2025](https://link.springer.com/article/10.1007/s00211-025-01492-5)
- [arXiv:2406.11751 — rpCholesky-QR (Garrison & Ipsen 2024)](https://arxiv.org/abs/2406.11751)
- [arXiv:2412.06551 — Randomized LU-Householder CholeskyQR (Guan & Fan 2024)](https://arxiv.org/abs/2412.06551)
- [arXiv:2111.11148 — XR Randomized Preconditioned CholeskyQR (Fan et al. 2021)](https://arxiv.org/abs/2111.11148)
- [arXiv:2311.08316 — CQRRPT (Murray et al. 2023/2025)](https://arxiv.org/abs/2311.08316)
- [SIAM SIMAX — CQRRPT published 2025](https://epubs.siam.org/doi/10.1137/24M163712X)
- [arXiv:2503.18496 — Randomized Strong RRQR (2025)](https://arxiv.org/abs/2503.18496)
- [arXiv:1804.05138 — Fast Parallel Randomized QRCP (HQRRP, 2018)](https://arxiv.org/abs/1804.05138)
- [arXiv:2508.14209 — High Performance GPU CountSketch (2025)](https://arxiv.org/abs/2508.14209)
- [ACM DL — CountSketch SC'25 paper](https://dl.acm.org/doi/full/10.1145/3731599.3767544)
- [arXiv:2002.00864 — Optimal Iterative Sketching with SRHT](https://arxiv.org/pdf/2002.00864)
- [arXiv:2011.05090 — Randomized Gram-Schmidt (Balabanov & Grigori 2021)](https://arxiv.org/pdf/2011.05090)
- [Wiley — Randomized Orth Process with Reorthogonalization (Jang et al. 2025)](https://onlinelibrary.wiley.com/doi/full/10.1002/nla.70029)
- [arXiv:2512.15455 — Randomized Orthogonalization and Krylov (de Damas et al. 2025)](https://arxiv.org/pdf/2512.15455)
- [arXiv:2302.07202 — Sketch-and-Precondition Stability (Meier et al. 2023)](https://arxiv.org/pdf/2302.07202)
- [arXiv:2506.03070 — GPU Sketch-and-Precondition Sparse Sign (2025)](https://arxiv.org/pdf/2506.03070)
- [arXiv:1912.05508 — RMGSQR Tensor Core QR (2019)](https://arxiv.org/abs/1912.05508)
- [ar5iv HTML — RMGSQR readable version](https://ar5iv.labs.arxiv.org/html/1912.05508)
- [Manchester preprint — Prob Rounding QR (Connolly & Higham 2023)](https://eprints.maths.manchester.ac.uk/2865/1/paper.pdf)
- [SIAM SIMAX — Probabilistic rounding QR published 2023](https://epubs.siam.org/doi/10.1137/22M1514817)
- [arXiv:2410.10517 — Stochastic Rounding 2.0 (Drineas & Ipsen 2024)](https://arxiv.org/abs/2410.10517)
- [arXiv:2508.00441 — DGEMM FP8 Ozaki (2025)](https://arxiv.org/abs/2508.00441)
- [arXiv:2504.08009 — Ozaki Scheme II (2025)](https://arxiv.org/pdf/2504.08009)
- [GitHub GEMMul8 — Ozaki FP8/INT8 GEMM](https://github.com/RIKEN-RCCS/GEMMul8)
- [arXiv:2507.00976 — Anatomy of Column-Pivoted QR (2025)](https://arxiv.org/abs/2507.00976)
- [arXiv:2508.05400 — Randomized Krylov-Schur (de Damas & Grigori 2025)](https://arxiv.org/pdf/2508.05400)
- [arXiv:2506.04208 — RandCholeskyQR2 for Sparse (Guan & Fan 2025)](https://arxiv.org/abs/2506.04208)
- [GitHub RandLAPACK](https://github.com/BallisticLA/RandLAPACK)
- [Ethan Epperly blog — Neat Randomized Algorithms: Randomized Cholesky QR (2024)](https://www.ethanepperly.com/index.php/2024/06/25/neat-randomized-algorithms-randomized-cholesky-qr/)
- [arXiv:2410.09389 — Improved error analysis of CholeskyQR with randomized model (2024)](https://arxiv.org/pdf/2410.09389)
