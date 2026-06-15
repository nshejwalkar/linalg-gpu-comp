# Exotic / Interesting QR & Orthogonalization Algorithms

Survey of algorithms in the **streaming / communication-avoiding / structured** family for
the GPU-MODE QR competition (batched square Householder QR returning `(H, tau)` in `geqrf`
format, FP32, B200, batch 2–640, n 32–4096).

Verdict tags:
- **PROMISING** — worth prototyping for speed on this problem
- **INDIRECT** — ideas that apply partially or inspire a sub-component
- **NOT APPLICABLE** — structurally incompatible or wrong regime

Research date: June 2026. Sources linked inline.

---

## 1. TSQR — Tall-Skinny QR (Demmel et al., 2012)

### Idea
TSQR factors a tall-skinny matrix as a tree-reduction: partition the rows into leaves,
compute local Householder QR on each leaf, then merge R factors pairwise up the tree
(each merge is a tiny `2b × b` QR). Communication is one allreduce per b columns
instead of one per column (as in the LAPACK panel). The final R is exact; Q is implicit
in the tree of local factors.

### Why Interesting
It is the minimum-communication algorithm for the panel factorization step of rectangular QR.
For GPUs, CAQR (Communication-Avoiding QR) uses TSQR as its panel primitive and was shown
to outperform GPU CULA/MKL by up to 17× on tall-skinny inputs by keeping work compute-bound
rather than memory-bound ([LAWN 240](https://www.netlib.org/lapack/lawnspdf/lawn240.pdf);
[IEEE 2011](https://ieeexplore.ieee.org/document/6012824/)).

Recent GPU work (March 2026) benchmarks TSQR on H100 for very-few-column matrices and
finds 3× speedup over SVQB2 at n ≤ 8 columns, narrowing to ~1.3× at n=32
([arXiv:2603.20889](https://arxiv.org/html/2603.20889)).

### Caveats
TSQR's Q is stored in a non-standard "tree-of-tiles" format — converting to the compact
LAPACK `(H, tau)` Householder format requires an extra reconstruction pass
([OSTI reconstruction paper](https://www.osti.gov/servlets/purl/1236219)). This is a
meaningful overhead. TSQR shines most on tall-skinny matrices (more rows than columns);
for square matrices, the tree reduces to just one level = standard panel Householder.

### Speed / Relevance for Us
**INDIRECT.** For our square matrices, TSQR collapses to a single-level panel, giving
no advantage over our existing blocked Householder. The reflector reconstruction overhead
would add latency. However, the *CAQR idea* — blocking the panel to convert memory-bound
GEMV into compute-bound GEMM — is exactly what our v13 shared-memory-resident panel
kernel already exploits. TSQR validates this design; we don't need TSQR itself.

---

## 2. Communication-Avoiding QR (CAQR) for General Rectangular Matrices

### Idea
Full CAQR ([Demmel, Grigori, Hoemmen, Langou, 2012](https://www.netlib.org/lapack/lawnspdf/lawn240.pdf))
replaces the classical panel + trailing-update with a 2D block-cyclic schedule using TSQR
panels. In the sequential (single-GPU) variant, CAQR eliminates redundant reads of the
trailing matrix by processing blocks in a specific order that achieves the communication
lower bound (O(n²/√P) words).

### Why Interesting
On a single GPU, CAQR's main benefit is making the panel compute-bound (same as TSQR).
The trailing update is blocked GEMMs regardless — same as WY-form. The gain is primarily
for panels of large rectangular matrices where memory bandwidth is the bottleneck.

### Caveats
For square matrices (our case), the panel is already O(b × n) and trailing update
dominates for b < n. CAQR's communication savings relative to blocked Householder are
smaller for square matrices than for tall-skinny ones. Full CAQR also stores Q implicitly
in a non-standard format, requiring reconstruction.

### Speed / Relevance for Us
**INDIRECT.** The key insight — "make the panel compute-bound by keeping it in fast
memory" — is already captured in our v13 kernel. CAQR is the theoretical framework
that justifies v13's design.

---

## 3. CholeskyQR / CholeskyQR2 (Fukaya et al., 2014–2025)

### Idea
Given A (m×n), form the Gram matrix G = AᵀA via one GEMM, Cholesky-factor G = RᵀR,
and the QR is A = (AR⁻¹) · R. CholeskyQR2 applies this twice for stability.
CQRRPT (2025, SIAM JMAA) adds randomized column pivoting on top, enabling
rank-revealing QR in one communication-optimal pass
([arXiv:2311.08316](https://arxiv.org/abs/2311.08316);
[SIAM doi](https://epubs.siam.org/doi/10.1137/24M163712X)).

### Why Interesting
All operations are BLAS-3 (one SGEMM + one STRTRS); Tensor Cores can be used for the
GEMM. CholeskyQR2 achieves near-theoretical peak FLOP rates on GPU because the bottleneck
is a single large GEMM per pass. For square n×n batch: 2 passes × (one SGEMM n²k +
one Cholesky O(n³/3)) — both parallelizable across batch.

Recent paper mCQRGSI+ (Nov 2025) combines CholeskyQR with Gram-Schmidt stabilization
for ill-conditioned inputs
([MDPI Mathematics](https://www.mdpi.com/2227-7390/13/22/3608)).

Rand_cholQR (arXiv:2309.05868) adds one or two random sketch matrices as preconditioners
to guarantee FP32-level orthogonality even for ill-conditioned A, at nearly zero extra cost
([abstract](https://arxiv.org/abs/2309.05868)).

### Caveats
**THE FATAL CONSTRAINT FOR OUR COMPETITION:** CholeskyQR returns R from the Cholesky
factor and Q = AR⁻¹ — it does NOT produce Householder reflectors `(H, tau)`. Our
checker calls `householder_product(H, tau)` and there is no cheap way to synthesize
reflectors from a Q factor. This rules out all CholeskyQR variants as a drop-in.

However, CholeskyQR could be used internally if we could reconstruct (H, tau) afterwards
— but reconstruction from an arbitrary Q requires O(n³) Householder steps, negating
the speed advantage.

### Speed / Relevance for Us
**NOT APPLICABLE (output format).** The algorithm itself would be extremely fast but
cannot satisfy the `(H, tau)` output contract. Confirmed in our CLAUDE.md (finding A1).

---

## 4. Randomized Householder QR (RHQR, 2024)

### Idea
RHQR ([arXiv:2405.10923](https://arxiv.org/abs/2405.10923)) replaces the inner product
`vᵀA` in each Householder step with a *sketched* inner product `(Ψv)ᵀ(ΨA)` where Ψ is
a random subspace embedding. The key claim: RHQR of W is equivalent to standard
Householder QR of ΨW. The "left-looking" variant requires only one synchronization
per step (vs. two for classical Householder) and roughly halves computation at the cost
of sketch quality.

### Why Interesting
Halving the synchronization count in the panel is exactly our bottleneck (we are
launch-bound with ~10k tiny kernels/iter per our profiling). A sketched panel could
replace 2 barrier-requiring reductions per column with 1 cheaper sketched reduction.
On GPU, this translates to fewer `__syncthreads()` per Householder step, potentially
improving occupancy and throughput in the inner panel kernel.

The paper also shows that using half-precision for the sketching step (while keeping
FP32 for the output factors) is theoretically grounded and matches standard Householder
stability.

### Caveats
RHQR modifies the Householder vectors — the resulting H is equivalent to standard
Householder QR of a *different* (sketched) matrix. Whether our checker's `householder_product`
is numerically satisfied depends on whether the sketch distortion stays within the
`20·n·eps32` residual gate. For well-conditioned dense inputs (our ranked set) this
is very likely fine; for `clustered`/`nearcollinear` stress cases, the sketch quality
matters more. Needs testing.

The sketch itself (typically a sparse sign matrix or SRHT) adds a small overhead per
Householder step — worth it only if it buys more parallelism.

### Speed / Relevance for Us
**INDIRECT-TO-PROMISING.** The half-sync idea has merit for reducing barriers inside
our Triton panel kernel (currently each column step has two reductions: norm and dot).
Not a full algorithm swap, but a panel micro-optimization.

---

## 5. Randomized Householder-Cholesky QR (rand_cholQR / Multisketching, 2023–2024)

### Idea
rand_cholQR ([arXiv:2309.05868](https://arxiv.org/abs/2309.05868)) uses one or two
random sketch matrices as preconditioners to make CholeskyQR stable for any
numerically full-rank matrix (not just well-conditioned ones). The algorithm:
1. Sketch A → S = ΨA (cheap matmul)
2. QR of S → preconditioning factor R̃ (small matrix, cheap)
3. CholeskyQR on AR̃⁻¹ (now well-conditioned by construction)

On NVIDIA A100, this is nearly as fast as CholeskyQR2 while being numerically stable
for arbitrarily ill-conditioned inputs.

### Caveats
Same fatal constraint as all CholeskyQR variants: produces (Q, R), not (H, tau).
The sketching idea, however, could be adapted to precondition our Householder QR
(precondition → fewer iterations of panel Householder needed? Not directly, since
Householder is already exact in one pass).

### Speed / Relevance for Us
**NOT APPLICABLE (output format).** Interesting algorithm. Wrong output contract.

---

## 6. Streaming / Online / Incremental QR (Rank-1 Updating/Downdating)

### Idea
When new rows or columns are added/removed, the existing QR can be *updated* rather
than recomputed. Givens rotation-based updating: adding one row requires n Givens
rotations to restore the R factor to upper-triangular form. Downdating (removing a row)
is harder, requiring hyperbolic rotations. GPU implementations exist
([Manchester MIMS preprint](https://eprints.maths.manchester.ac.uk/2116/1/paper.pdf);
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167819114000337))
and achieve speedups of up to 13.5× over full GPU recomputation for column removal.

Streaming / one-pass QR for out-of-core matrices: Gunter & van de Geijn (2005) describe
updating the QR factorization in one sweep when rows arrive sequentially
([ACM TOMS](https://dl.acm.org/doi/10.1145/1055531.1055534)).

### Why Interesting
Conceptually beautiful: process a dataset as it arrives without storing it all. The
one-pass formulation is exactly a sequential application of Givens rotations to
accumulate R in O(n²) space.

### Caveats
Requires a "previous" factorization to update — doesn't apply to computing QR from
scratch (our case). The Householder reflector output format is also non-trivial to
maintain across updates. GPU speedups come from batching many simultaneous updates,
not from single-matrix throughput.

### Speed / Relevance for Us
**NOT APPLICABLE.** We always start from a fresh dense matrix; there is no prior
factorization to update. Wrong problem setting.

---

## 7. Givens Rotation Networks / Systolic Arrays (CORDIC-based)

### Idea
A triangular systolic array of processing elements (PEs) can zero sub-diagonal entries
column by column in parallel. The classic Sameh-Kuck wavefront: column j is zeroed by
n−j Givens rotations; rotations at different columns can be pipelined. With CORDIC
(Coordinate Rotation Digital Computer), each Givens rotation is computed via a
shift-and-add ladder without division, requiring ~3n²/2 rotations total.

### Why Interesting
Systolic arrays achieve perfectly regular, conflict-free memory access patterns and
are exceptionally hardware-friendly. Massively parallel on FPGA/VLSI. CORDIC is
area-efficient and avoids division (GPU has fast FP divide, so this matters less).

On GPU, the wavefront dependency structure maps naturally to diagonal sweeps: diagonal
k processes all independent (i,j) pairs with i+j = k simultaneously. This is how
parallel Givens QR is typically implemented on GPU
([MPI-CUDA paper](https://thesai.org/Downloads/Volume11No5/Paper_78-Parallel_QR_Factorization_using_Givens_Rotations.pdf)).

### Caveats
Total FLOP count: Givens QR is ~4n³/3 (vs. Householder's ~2n³/3 for square), about 2×
more flops. For memory-bound regimes this is fine, but our B200 is very FLOP-rich.
Each rotation touches only 2 rows → terrible arithmetic intensity compared to GEMM.
The output is Q and R, not Householder reflectors; Q accumulation requires another O(n³)
pass to generate reflector form. CORDIC saves area on FPGA but adds latency on GPU where
fast FP hardware is available.

GPU Givens implementations typically 2–5× slower than Householder for large matrices
because of poor GEMM utilization.

### Speed / Relevance for Us
**NOT APPLICABLE for this problem.** Slower than Householder for square matrices, wrong
output format (no cheap (H, tau) from Givens), and poor tensor-core utilization.

---

## 8. Butterfly / Hierarchical Matrix QR

### Idea
A butterfly matrix is one that admits an O(N log N) factorization into J = log₂ N sparse
"butterfly" factors, each with only 2 non-zeros per row
([arXiv:1502.01379](https://arxiv.org/pdf/1502.01379);
[SIAM JMDS](https://epubs.siam.org/doi/10.1137/22M1488727)).
The hierarchical factorization method identifies these factors from the matrix's structure.
For QR, if the matrix itself has butterfly structure, the Q factor can be represented
in O(N log N) space and applied in O(N log² N) time.

Block Low-Rank (BLR) QR decomposes a dense matrix into tiles and exploits off-diagonal
low-rank structure for compression. GPU implementations exist using batched small-QR
primitives ([arXiv:2208.06194](https://arxiv.org/pdf/2208.06194);
[Springer](https://link.springer.com/chapter/10.1007/978-3-031-29927-8_28)).

### Why Interesting
For matrices with inherent low-rank off-diagonal structure (arising in PDE solvers,
kernel methods, hierarchical $N$-body), H-matrix QR reduces complexity from O(n³) to
O(n log² n) or O(n log n). ButterflyQuant (2025, arXiv:2509.09679) uses learnable
butterfly transforms to achieve quantization with guaranteed orthogonality.

### Caveats
Our benchmark matrices are **dense with no guaranteed low-rank structure** — `cond=1`
or `cond=2` matrices are generically full-rank everywhere. Butterfly/H-matrix compression
requires the matrix to have exploitable structure; applying it to a generic dense matrix
would yield no compression and large overhead for structure detection.

BLR-QR literature focuses on compression/memory savings for single large matrices, not
throughput of many small dense square matrices.

### Speed / Relevance for Us
**NOT APPLICABLE.** Dense square benchmark matrices have no exploitable structure.
The ideas only pay off when the matrix is sparse, banded, or hierarchically low-rank.

---

## 9. Randomized Gram-Schmidt with Reorthogonalization (Low-Synchronization Variants)

### Idea
Classical Gram-Schmidt (CGS) and Modified Gram-Schmidt (MGS) are prone to orthogonality
loss. Three families of fixes:

(a) **Reorthogonalized CGS / MGS (CGS2, ICGSRO):** Apply the projection step twice.
Doubles cost but achieves near-machine-precision orthogonality.

(b) **Low-synchronization (low-synch) variants:** Rearrange the computation to reduce
the number of global all-reduces per column. The BCGSI+P-1S / BCGSI+P-2S algorithms
(July 2025, [arXiv:2507.21791](https://arxiv.org/abs/2507.21791)) achieve 4× speedup
over standard BCGSI+ on distributed memory, with only 1 or 2 synchronization points
per block. Designed for Krylov solver orthogonalization.

(c) **Randomized GS (RGS):** Incorporate a random sketch Ψ into the projection;
re-orthogonalize only when the sketch detects near-linear dependence. Halves
computation compared to double-reorthogonalization
([Wiley 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/nla.70029)).

(d) **Randomized-sketching for s-step GMRES:** Random sketches reduce synchronization in
block orthogonalization for Krylov methods
([arXiv:2503.16717](https://arxiv.org/pdf/2503.16717)).

### Why Interesting
Low-sync variants are directly motivated by GPU architecture: every `__syncthreads()`
or warp-level `__shfl_sync` costs cycles, and global all-reduces in distributed settings
stall pipelines. Within a single GPU kernel (our Triton panel), reducing barrier count
from 2 to 1 per Householder step could improve throughput.

None of these produce `(H, tau)` directly, but the ideas are adaptable to the panel
Householder step.

### Caveats
GS variants (even with reorthogonalization) are generally less numerically stable than
Householder QR for highly ill-conditioned matrices. Our `band` and `rowscale` stress
tests already kill TF32; GS might struggle on `nearcollinear` / `nearrank` inputs.

Low-sync research is primarily motivated by *distributed-memory* settings (high latency
of MPI_Allreduce); on a single GPU, `__syncthreads()` is cheap (~5–20 cycles), so the
gain is modest.

### Speed / Relevance for Us
**INDIRECT.** The randomized-skip-reorthogonalization trick (do one cheap sketched test,
skip the second reorthogonalization pass if not needed) could slightly accelerate our
panel without hurting correctness on structured stress inputs. Not a primary lever.

---

## 10. Recursive Blocking / Elmroth–Gustavson Recursion

### Idea
Instead of left-looking or right-looking blocked Householder, split the panel vertically
into two halves and recurse. The Elmroth–Gustavson (2000) algorithm:
- Left half: recurse → gets R₁₁, Y₁, T₁
- Update right half with (I − Y₁T₁Y₁ᵀ) (a GEMM)
- Right half: recurse → gets R₂₂, Y₂, T₂
- Merge the two WY representations

This converts all operations to BLAS-3 and naturally chooses the block size as a
power of two at each level.
([Semantic Scholar](https://www.semanticscholar.org/paper/Applying-recursion-to-serial-and-parallel-QR-leads-Elmroth-Gustavson/139bf1cf76cce570260b66b180d911cebbf4a33d))

A 2021 paper ([ACM ICPP](https://dl.acm.org/doi/10.1145/3472456.3473522)) applies
this recursion to out-of-core TensorCore-based Gram-Schmidt QR and demonstrates
improved GEMM utilization on GPUs.

### Why Interesting
Recursive blocking is essentially "auto-tuning-free" — it naturally finds the right
GEMM shapes at every level for cache/register reuse. On GPU, small panels deep in
recursion stay in shared memory; large panels at the top use DRAM-resident trailing
updates via tensor-core GEMM. The WY merge at each level is a cheap O(b²n) operation.

For our v13 Triton kernel, recursion within the panel (e.g., instead of sequential
column steps, doing 2-level Householder with a small GEMM update in the middle) could
improve intra-warp GEMM utilization.

### Caveats
Recursive algorithms need O(log n) levels of recursion, each adding a kernel launch or
barrier. On GPU, this can turn into a "recursive launch" problem unless the entire
recursion fits within one kernel. Our Triton kernel currently does sequential Householder
within a single kernel — replacing it with internal recursion is tricky without
JIT-time recursion unrolling.

The base case must still be a scalar or 2×2 Householder — same kernel launch overhead
as today's sequential approach.

### Speed / Relevance for Us
**INDIRECT (but highly relevant for our panel).** The idea of doing a 2-level internal
recursion within our Triton panel kernel — apply first b/2 Householders, do a small GEMM
update, then apply next b/2 — could improve GEMM utilization from ~17% to higher. This
is the "fused WY build" we already identified as a lever. The recursive framing gives a
clean way to implement it.

---

## 11. Tensor-Core-Accelerated QR via Recursive Modified Gram-Schmidt (RMGSQR)

### Idea
RMGSQR ([arXiv:1912.05508](https://arxiv.org/abs/1912.05508)) recursively splits
the column dimension until the GEMM shapes are large enough for Tensor Cores to be
efficient (turns tall-skinny GEMMs into nearly-square GEMMs). Panel is CAQR-based;
trailing update uses FP16 Tensor Core GEMM. Accuracy is recovered via CGLS iterative
refinement using the low-precision R as a preconditioner.

Results: 2.9×–14.7× speedup over cuSOLVER SGEQRF for large single matrices on V100.

### Why Interesting
The core insight — recursive column splitting to improve GEMM aspect ratios so Tensor
Cores engage — is exactly the tunable knob for our trailing update GEMM. Our v13
trailing update does `A -= Y @ (Tᵀ @ (Yᵀ@A))`, where Y is (n × b) and A is (n × n−kb).
For b=32, n=512, these are 512×32 and 512×480 GEMMs — the first is very tall-skinny.

The iterative refinement trick (low-precision factor → FP32 correction) matches what
CLAUDE.md identifies as a potential lever once we are compute-bound.

### Caveats
Output is standard (Q, R), not `(H, tau)`. But our algorithm is already Householder —
the RMGSQR ideas apply to *improving the trailing GEMM*, not replacing our algorithm.

Iterative refinement adds a second pass that costs ~same as the original factorization
(one CGLS iteration), so the net gain on FP32 accuracy is roughly "1 FP16 pass +
1 correction pass" vs. "1 FP32 pass". Speedup depends on FP16/FP32 ratio (~2× on
most modern GPUs).

The `band`/`rowscale` correctness gate (finding B4 in findings.md) means we must
verify refinement passes the stress cases, not just dense benchmarks.

### Speed / Relevance for Us
**PROMISING (ideas, not the algorithm itself).** The recursive-column-split idea to
improve trailing GEMM aspect ratios is directly applicable to our blocked-WY. The
iterative refinement trick (TF32 or FP16 trailing GEMM + FP32 correction pass) is
a deferred but real lever once we become compute-bound (which we are not yet).

---

## 12. Mixed-Precision Cholesky QR (Mixed-Precision "ShiftedCholQR3" family)

### Idea
ShiftedCholeskyQR (Yamamoto et al., 2019) shifts the Gram matrix G = AᵀA + sI by
a small diagonal to keep it positive definite even when G has near-zero eigenvalues,
enabling Cholesky QR to succeed on near-rank-deficient matrices
([SIAM JSciC](https://epubs.siam.org/doi/10.1137/18M1218212)).

Mixed-precision CholQR uses FP16 for the GEMM, FP32 for the Cholesky factor
([SIAM JSciC](https://epubs.siam.org/doi/10.1137/14M0973773)).

A randomized preconditioned variant
([arXiv:2406.11751](https://arxiv.org/abs/2406.11751)) adds a sketch preconditioning
step to further improve stability.

### Caveats
All produce (Q, R), not (H, tau). See constraint A1 in findings.md.

### Speed / Relevance for Us
**NOT APPLICABLE (output format).** The mixed-precision trick (FP16 Gram matrix,
FP32 Cholesky) is interesting but the output contract blocks it.

---

## 13. Approximate Householder Reflectors (Tomás & Quintana-Ortí, 2020)

### Idea
In tall-skinny QR, computing each Householder vector requires reading the entire column
(O(m) DRAM reads for each of n steps). The "approximate Householder" approach computes
v on only a *random sample* of rows to estimate the norm, then applies the exact reflector.
This reduces data transfers between GPU global memory and registers for the norm computation.
([Springer J. Supercomputing](https://link.springer.com/article/10.1007/s11227-020-03176-3))

The paper shows large speedup over MAGMA for tall-skinny matrices by avoiding the extra
global-memory pass for the norm.

### Why Interesting
For our small (n ≤ 4096) square matrices where the panel fits in shared memory, the
norm computation is already done in shared memory (our v13 kernel). So this technique
is already superseded by our shared-memory-resident panel design.

### Speed / Relevance for Us
**INDIRECT / SUPERSEDED.** Our v13 panel already loads the column to shared memory
once and does all steps on-chip — the approximate-Householder's goal (reduce global
reads per step) is already achieved.

---

## 14. Tiled QR with DAG Scheduling (PLASMA / Chameleon / Task-Based)

### Idea
Instead of the monolithic blocked Householder schedule, decompose the factorization
into a DAG of tile-level tasks (DGEQRT, DORMQR, DTSQRT, DTSMQR). A dynamic
task scheduler (STARPU, OpenMP, DAGuE) issues ready tasks out-of-order, overlapping
panels and trailing updates across tiles. PLASMA achieves near-roofline performance
on multicore CPUs by hiding the panel-factorization latency behind independent trailing
updates on other tile columns
([arXiv:0707.3548](https://arxiv.org/pdf/0707.3548)).

### Why Interesting
The "panel on one SM cluster, trailing update on others, overlap" idea is the GPU
equivalent of hiding the panel serial bottleneck. Within a single GPU, our current
architecture does: panel (sequential Python loop), then trailing update (blocked GEMM).
A fully-fused Triton kernel that pipelines these steps could achieve better SM
utilization.

### Caveats
On GPU, dynamic task scheduling is extremely hard to implement efficiently — the
overhead of task dispatch outweighs the benefit for matrix sizes below ~4096×4096.
For our batched setting (many small matrices), the tile sizes are already at the bottom
of the DAG (each matrix is one "tile"), so the DAG schedule reduces to our existing
sequential panel + batch GEMM.

### Speed / Relevance for Us
**INDIRECT.** The "overlap panel + trailing update across batch elements" idea is
relevant if different batch elements are at different stages simultaneously. Our current
implementation serializes panel and trailing update for each block column before moving
to the next — pipelining across block columns (start trailing update b for next batch
elements while this batch's panel runs) could improve GPU utilization. Difficult to
implement in Triton but conceptually clean.

---

## 15. 3D Parallel QR (3D QR, Ballard–Carson–Demmel, 2018)

### Idea
3D QR ([arXiv:1805.05278](https://arxiv.org/pdf/1805.05278)) distributes the matrix
across a 3D processor grid (P = pˣ × pʸ × pᶻ dimensions) and achieves
O(n²/P^{2/3}) communication vs. O(n²/√P) for 2D algorithms. This is communication-
optimal for parallel distributed factorization.

### Why Interesting
Theoretically elegant: achieves the communication lower bound simultaneously in
computation and bandwidth cost. For massive parallel clusters, this provides asymptotically
better scaling than CAQR.

### Caveats
We are on a single B200 GPU with unified memory — there is no inter-node communication.
3D algorithms have enormous overhead on a single device (requires complex data
redistribution that is pure overhead when P=1). Not applicable.

### Speed / Relevance for Us
**NOT APPLICABLE.** Single-GPU setting; 3D algorithms require distributed memory.

---

## 16. Tournament Pivoting QR (CAQR with Column Pivoting)

### Idea
Tournament pivoting replaces standard LAPACK-style column pivoting (one allreduce per
column to find the max) with a tournament among candidates: locally pivot within each
block, then run a tournament to select the global best. This reduces pivoting's
communication cost from O(n²) to O(n) all-reduces while still revealing rank
([LAWN 276](https://www.netlib.org/lapack/lawnspdf/lawn276.pdf);
[SIAM JMAA 2020](https://epubs.siam.org/doi/10.1137/20M1387663) for tensors).

### Why Interesting
Rank-revealing QR with near-communication-optimal pivoting. PAQR (Pivoting-Avoiding QR)
proposes to skip pivoting entirely by preconditioning with a random sketch
([Dongarra group, Netlib](https://www.netlib.org/utk/people/JackDongarra/PAPERS/PAQR.pdf)).

### Caveats
Our ranked benchmark matrices are full-rank dense matrices (cond=1 or 2); pivoting
provides no benefit and adds overhead. The column-pivoted output also differs from the
unpivoted `(H, tau)` format we need.

### Speed / Relevance for Us
**NOT APPLICABLE.** Our matrices don't benefit from pivoting (full-rank, well-conditioned);
the output format would differ.

---

## 17. High-Performance Anatomy of Column-Pivoted QR (RandLAPACK, 2025)

### Idea
A July 2025 modular framework ([arXiv:2507.00976](https://arxiv.org/abs/2507.00976))
decomposes QRCP into interchangeable subroutines (sketch, pivot selection, panel,
trailing update), enabling hardware-specific algorithm assembly. On H100 it reaches
~65% of cuSOLVER's unpivoted QR throughput; on CPU, two orders of magnitude faster
than LAPACK DGEQP3.

### Caveats
Designed for column-pivoted QR; our problem is unpivoted. The modular-subroutine
architecture idea is interesting for structuring our own code.

### Speed / Relevance for Us
**INDIRECT.** The modular approach could inform how we structure dispatch (pick best
subroutine per shape), but the specific algorithms are for rank-revealing pivoted QR.

---

## 18. Batched Sub-Warp QR for Very Small Matrices (MAGMA 2017–2022)

### Idea
For matrices too small for a full thread block (n ≤ 32), assign a *sub-warp* (4–16
threads) to each matrix. Entire n×n matrix fits in registers; no shared memory needed.
The panel loop runs in registers with warp-shuffle reductions. MAGMA's batch QR
([arXiv:1707.05141](https://arxiv.org/pdf/1707.05141);
[Netlib 2022](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batchqr-gpu-2022.pdf))
reports 1.2×–23.8× speedup over batched cuBLAS for n ≤ 100, with larger speedups at
smaller n.

### Why Interesting
This is *directly* the regime of our n=32 case (b20 n32). cuSOLVER's geqrf is
already fast there (323 µs); our blocked_wy is 16× slower there. A sub-warp kernel
that puts the full 32×32 matrix in registers and does 32 Householder steps purely
in registers + warp shuffles (no shared memory, no global memory during the panel)
could compete with or beat geqrf.

The MAGMA 2022 paper uses a "nested blocking" strategy: sub-warp level for very small
n, thread-block level for medium n.

### Speed / Relevance for Us
**PROMISING (HIGH PRIORITY for n=32 and n=176 cases).** This is the v10 "fused
small-n kernel" item in our CLAUDE.md next-steps list. A sub-warp register-level
Householder kernel for n ≤ 176 would directly attack our worst cases (b20 n32: 16×
slower; b40 n176: ~1.5× slower). The output is (H, tau) in the same Householder
convention — just write the reflectors into the lower triangle after each step.

References to read before implementing: MAGMA batched QR (arXiv:1707.05141),
Batch QR Factorization on GPUs (Netlib 2022 paper above).

---

## 19. Performant SVD via Bidiagonalization (Unified GPU Kernels, 2025)

### Idea
A 2025 paper ([arXiv:2508.06339](https://arxiv.org/pdf/2508.06339)) presents unified
GPU kernels for SVD via bidiagonalization (which requires Householder reductions).
The kernels target portability across hardware and precision using a single CUDA C++
implementation.

### Caveats
SVD bidiagonalization is a two-sided reduction; our problem is one-sided Householder
QR. The techniques (register tiling, warp shuffles for reductions) are transferable
but the algorithm itself is different.

### Speed / Relevance for Us
**INDIRECT.** The warp-level programming patterns for reductions in the panel step are
transferable. The algorithm itself is not.

---

## Summary Table

| # | Algorithm | Family | Speed Verdict | Applicable? |
|---|-----------|--------|--------------|-------------|
| 1 | TSQR | Comm-avoiding | INDIRECT | Ideas used in v13 |
| 2 | CAQR | Comm-avoiding | INDIRECT | Already applied |
| 3 | CholeskyQR2 / CQRRPT | Gram-based | NOT APPLICABLE | Wrong output format |
| 4 | RHQR (randomized Householder) | Randomized | INDIRECT | Panel barrier reduction |
| 5 | rand_cholQR / multisketching | Randomized+Cholesky | NOT APPLICABLE | Wrong output format |
| 6 | Streaming / incremental QR | Online/update | NOT APPLICABLE | Different problem |
| 7 | Givens systolic / CORDIC | Hardware-oriented | NOT APPLICABLE | 2× FLOPS, wrong format |
| 8 | Butterfly / H-matrix QR | Structured | NOT APPLICABLE | Matrices are dense |
| 9 | Low-sync Gram-Schmidt / RGS | Krylov-oriented | INDIRECT | Panel barrier reduction |
| 10 | Elmroth–Gustavson recursion | Recursive blocking | INDIRECT (high relevance) | Panel-internal GEMM lever |
| 11 | RMGSQR (TensorCore + iterative refinement) | Mixed-precision | PROMISING (ideas) | Trailing GEMM improvement |
| 12 | Mixed-precision CholQR | Mixed-precision | NOT APPLICABLE | Wrong output format |
| 13 | Approximate Householder | Comm-reducing | INDIRECT / superseded | Already done in v13 |
| 14 | Tiled QR / DAG scheduling | Task-parallel | INDIRECT | Overlap inspiration |
| 15 | 3D Parallel QR | Distributed | NOT APPLICABLE | Single GPU only |
| 16 | Tournament Pivoting QR | Rank-revealing | NOT APPLICABLE | Full-rank matrices |
| 17 | RandLAPACK anatomy QRCP | Modular framework | INDIRECT | Dispatch architecture |
| 18 | Sub-warp batched QR (MAGMA) | Small-matrix GPU | **PROMISING (HIGH PRIORITY)** | n=32/176 fix |
| 19 | SVD via bidiagonalization | Related | INDIRECT | Warp patterns only |

---

## Key Sources

- TSQR / CAQR original: [LAWN 240 (netlib)](https://www.netlib.org/lapack/lawnspdf/lawn240.pdf)
- CAQR GPU IEEE paper: [IEEE IPDPS 2011](https://ieeexplore.ieee.org/document/6012824/)
- TSQR for tall-skinny GPU (H100, 2026): [arXiv:2603.20889](https://arxiv.org/html/2603.20889)
- Randomized Householder QR: [arXiv:2405.10923](https://arxiv.org/abs/2405.10923)
- rand_cholQR multisketching: [arXiv:2309.05868](https://arxiv.org/abs/2309.05868)
- CQRRPT (SIAM 2025): [arXiv:2311.08316](https://arxiv.org/abs/2311.08316) / [SIAM JMAA](https://epubs.siam.org/doi/10.1137/24M163712X)
- mCQRGSI+ distributed GPU (Nov 2025): [MDPI Mathematics](https://www.mdpi.com/2227-7390/13/22/3608)
- Anatomy of QRCP (July 2025): [arXiv:2507.00976](https://arxiv.org/abs/2507.00976)
- RMGSQR TensorCore: [arXiv:1912.05508](https://arxiv.org/abs/1912.05508) / [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1912.05508)
- Approximate Householder GPU: [J. Supercomputing 2020](https://link.springer.com/article/10.1007/s11227-020-03176-3)
- QR updating on GPU: [MIMS eprint](https://eprints.maths.manchester.ac.uk/2116/1/paper.pdf) / [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167819114000337)
- Givens / CORDIC GPU: [MPI-CUDA paper](https://thesai.org/Downloads/Volume11No5/Paper_78-Parallel_QR_Factorization_using_Givens_Rotations.pdf)
- BLR-QR parallel: [arXiv:2208.06194](https://arxiv.org/pdf/2208.06194)
- Low-sync Block Gram-Schmidt (2025): [arXiv:2507.21791](https://arxiv.org/abs/2507.21791)
- Randomized GS + reorthogonalization (2025): [Wiley NLA](https://onlinelibrary.wiley.com/doi/full/10.1002/nla.70029)
- Elmroth-Gustavson recursion: [Semantic Scholar](https://www.semanticscholar.org/paper/Applying-recursion-to-serial-and-parallel-QR-leads-Elmroth-Gustavson/139bf1cf76cce570260b66b180d911cebbf4a33d)
- TensorCore recursion ICPP 2021: [ACM ICPP](https://dl.acm.org/doi/10.1145/3472456.3473522)
- MAGMA batched QR 2017: [arXiv:1707.05141](https://arxiv.org/pdf/1707.05141)
- Batch QR GPU design 2022: [Netlib PDF](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batchqr-gpu-2022.pdf)
- ShiftedCholeskyQR: [SIAM JSciC](https://epubs.siam.org/doi/10.1137/18M1218212)
- Tournament Pivoting QRCP: [LAWN 276](https://www.netlib.org/lapack/lawnspdf/lawn276.pdf)
- Tiled QR / PLASMA: [arXiv:0707.3548](https://arxiv.org/pdf/0707.3548)
- 3D Parallel QR: [arXiv:1805.05278](https://arxiv.org/pdf/1805.05278)
- Out-of-core QR updating: [ACM TOMS](https://dl.acm.org/doi/10.1145/1055531.1055534)
- Ranking-revealing rank via CholeskyQR + pivot 2024 IEEE IPDPS: (Fukaya, Nakatsukasa, Yamamoto)
- Mixed-precision iterative refinement survey: [arXiv:2007.06674](https://arxiv.org/pdf/2007.06674)
- Recovering FP32 accuracy from Tensor Cores: [IJHPCA 2022](https://journals.sagepub.com/doi/full/10.1177/10943420221090256)
