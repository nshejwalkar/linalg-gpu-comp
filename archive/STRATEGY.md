# QR Competition Strategy — Based on Trefethen & Bau

## What the Competition Actually Asks For

The submission must return `(H, tau)` in **LAPACK geqrf format**:
- `H`: `(batch, n, n)` — R on and above the diagonal; Householder vectors below (with `H[i,i] = 1` implicit)
- `tau`: `(batch, n)` — scalar factors for each reflector
- PyTorch checks with `torch.linalg.householder_product(H, tau)` to recover Q

**Tolerance**: `20 * n * eps32` for factorization residual, `100 * n * eps32` for orthogonality.  
These are calibrated for standard FP32 Householder QR (TF32 tensor cores can still pass for most n).

---

## The Two QR Algorithms from Trefethen & Bau

### Gram-Schmidt (Lectures 7–8): Triangular Orthogonalization
- Process columns left-to-right, making each new column orthogonal to all previous
- Classical GS (Algorithm 7.1): **numerically unstable** — errors amplified by ~1/ε_machine
- Modified GS (Algorithm 8.1): **more stable** — same math, different order of operations
- Cost: ~2mn² flops (Theorem 8.1)
- GPU-friendly? Somewhat — inner loop is GEMV + rank-1 update, but sequential

### Householder (Lecture 10): Orthogonal Triangularization  
- Apply n unitary reflections from the LEFT: `Q_n ... Q_1 A = R`
- Each reflector `Q_k = I - 2 v_k v_k^T` zeroes out below the diagonal in column k
- **Numerically backward stable** (Lecture 16, Theorem 16.1): `||Q̃R̃ - A||/||A|| = O(ε_machine)`
- Cost: `~2mn² - (2/3)n³` flops ≈ `2n³/3` for square matrices (Eq. 10.9)

**The baseline (`torch.geqrf`) uses Householder QR via cuSOLVER.**

---

## The Key GPU Optimization: Blocked WY-Form Householder

### Why Naive Householder is GPU-Unfriendly
Sequential Householder (Algorithm 10.1):
```
for k = 1 to n:
    x = A[k:m, k]
    v_k = sign(x₁)||x||e₁ + x           # O(n) flops — fast
    A[k:m, k:n] -= 2v_k(v_k^T A[k:m,k:n])  # O(n²) flops — GEMV + rank-1 update
```
Problem: n sequential GEMV operations, each touching ~n² memory. Only 10–20% of peak FLOPS.

### The WY Form (Schreiber & Van Loan, 1989)
Group b steps into a "block reflector":
```
H₁ H₂ ... H_b = I - Y T Y^T
```
where `Y` is `m × b` (the Householder vectors) and `T` is `b × b` upper triangular.

T is built column by column:
- `T[j,j] = tau_j`
- `T[0:j, j] = -tau_j * T[0:j, 0:j] @ (Y[j+1:, 0:j]^T @ Y[j+1:, j])`

### The Blocked Algorithm (LAPACK DGEQRT)

```
for k = 0, b, 2b, ..., n-b:
    # Panel factorization (b sequential steps, O(b²n) flops):
    for j = 0 to b-1:
        Apply Householder to A[k+j:, k+j], updating only panel A[k+j:, k:k+b]
    
    # Build Y from stored reflectors, compute T matrix
    
    # Trailing update (ONE big GEMM, ~O(n²b) flops):
    C = Y^T @ A[k:, k+b:]          # (b, n-k) × (n-k, n-k-b) GEMM
    A[k:, k+b:] -= Y @ (T^T @ C)   # (n-k, b) × (b, n-k-b) GEMM
```

**Work breakdown** (n=1024, b=64):
- Panel: `b * n² / 2 = 64 * 1024² / 2 ≈ 33M flops` (sequential GEMV, ~15% utilization)
- GEMM: `2n³/3 ≈ 715M flops` (tensor cores, ~85% utilization)

With b=64, **~96% of work hits tensor cores** vs ~0% for naive Householder.

---

## Key Insight for Batched QR on GPU

The input is `(batch, n, n)`. 

**cuSOLVER's batched QR** (`cusolverDnSgeqrfBatched`) often processes batch elements 
sequentially or with limited parallelism across the batch.

**torch.bmm** (cuBLAS batched GEMM) processes all batch elements simultaneously with
near-peak tensor core utilization.

**Strategy**: replace the trailing-update GEMM with `torch.bmm`, giving the full batch
parallel speedup for the dominant computational cost.

---

## Correctness Format (LAPACK geqrf convention)

At step k, given `x = A[k:, k]`:
1. `alpha = x[0]` (current diagonal)
2. `beta = -sign(alpha) * ||x||₂` (new R diagonal)  
3. `v[0] = alpha - beta`, `v[1:] = x[1:]`  
4. `tau = 2*v[0]² / ||v||²`
5. `u = v / v[0]` (normalized: `u[0] = 1`)
6. Store: `H[k,k] = beta`, `H[k+1:, k] = u[1:]`, `tau[k] = tau`
7. Apply: `H[k:, k+1:] -= tau * u * (u^T @ H[k:, k+1:])`

The reflector is `F = I - tau * u * u^T`, where `F * x = beta * e₁`.

Sign convention (Trefethen Eq. 10.5): `v = sign(x₁)||x||e₁ + x` to avoid cancellation.

---

## Priority Optimization Order

1. **Blocked WY-form** with `torch.bmm` trailing update — targets batch throughput
2. **Triton kernel for panel factorization** — removes Python loop overhead
3. **Adaptive block size** — tune b for each (n, batch) combination
4. **BF16 trailing GEMM + FP32 correction** — if tolerance analysis shows this passes
5. **Custom CUDA kernel** — full control, cuBLAS GEMM directly

---

## Test Cases to Worry About

From `reference.py`, the edge cases include:
- `rankdef`: columns zeroed out → some tau=0, algorithm must handle gracefully
- `nearcollinear`: near-singular → cancellation in Householder computation
- `clustered`: wildly varying column scales → need sign convention for stability
- `upper`: already triangular → should be fast
- `band`: sparse structure → banded QR possible (not easy to exploit)

The tolerance `20*n*eps32` is generous enough for standard FP32 Householder (like cuSOLVER uses),
but **not generous enough for pure BF16** (BF16 eps ≈ 65536 × FP32 eps).
