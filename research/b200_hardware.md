# B200 (Blackwell) — what matters for this kernel

## Our exact stack (observed from the Modal build / grader)
torch **2.12.0+cu130**, **triton 3.7.0**, cuBLAS 13.1.1.3, cuSOLVER 12.0.4.66,
CUDA runtime 13.0, CUPTI present. Grader GPU: `NVIDIA B200`.

## Throughput by precision (approximate, dense; sparse ≈ 2×)
| dtype | ~peak | ratio vs FP32 | notes |
|---|---:|---:|---|
| FP32 (CUDA cores) | ~80 TFLOPS | 1× | non-tensor-core |
| TF32 (tensor) | ~1.1–2.2 PFLOPS | ~15–30× | 10-bit mantissa, FP32 accumulate |
| FP16/BF16 (tensor) | ~2.25 PFLOPS | ~30× | BF16: 8-bit mantissa |
| FP8 (tensor) | ~4.5 PFLOPS | ~60× | e4m3/e5m2 |
| FP4 / NVFP4 (tensor) | ~9 PFLOPS | ~120× | microscaling, 5th-gen TC |

Memory: ~180–192 GB HBM3e, **~8 TB/s** bandwidth. ~148–160 SMs, up to ~228 KB
shared memory per SM (configurable). Dual-reticle (2-die) design.

## TF32 — TESTED, and it's a NET NEGATIVE here (don't use global allow_tf32)
Hypothesis was "TF32 is the biggest free lever." **Measured on B200 it is the opposite:**
- **Breaks correctness:** with `allow_tf32=True`, `band` (scaled factor residual 27.9)
  and `rowscale` (26.1) **fail** the gate of 20. The row/column-scaled cases can't take
  the 10-bit mantissa. So TF32 is NOT free.
- **Slower on every benchmark shape** (b640 n512: 114 → 159 ms; n1024: 212 → 321 ms;
  small cases worse too) — extra cast kernels + a worse cuBLAS algo pick for these
  small-K batched GEMMs, on an already launch-bound workload.

**Key inference:** since speeding the GEMMs (TF32) didn't help, the trailing GEMMs are
**not** the bottleneck. blocked_wy is **launch-bound + panel-bound**, not GEMM-FLOP-bound.
→ Optimize launches (CUDA graphs) and the panel, NOT precision. BF16/FP8/FP4 are even
lower mantissa and would fail more cases — deprioritized. (If precision is ever revisited,
it must be applied selectively to safe shapes, never globally, and never to band/rowscale.)

## Compute-bound vs memory/launch-bound — different cases, different wins
- **Small/medium n, large batch** (n=32/176/352, and the b640 n512 headline): the
  per-matrix work is tiny; these are **launch-bound and memory-bound**, NOT FLOP-bound.
  The win is *fewer kernel launches* and *fewer global-memory round trips* — i.e.,
  fused single-block-per-matrix kernels and CUDA graphs — not raw tensor-core FLOPS.
  (This is exactly why cuSOLVER geqrf is so slow here: it serializes/under-batches and
  eats launch overhead.)
- **Large n, small batch** (n=2048, 4096): the trailing GEMMs are genuinely
  **compute-bound** → the precision lever (TF32/BF16) is what moves the needle, plus a
  recursive panel to keep the panel work on tensor cores.

## Rule of thumb for "does the matrix fit in shared memory"
`n² · 4 bytes ≤ ~228 KB`  ⇒  **n ≲ 230**. Below that, a whole FP32 matrix fits in one
SM's shared memory → fuse the entire per-matrix QR into one threadblock (n=32, n=176).
At n=352+ it doesn't fit → blocked/global algorithm with batched GEMM.

## Sources
- [NVIDIA Blackwell B100/B200/GB200 breakdown — Cudo](https://www.cudocompute.com/blog/nvidias-blackwell-architecture-breaking-down-the-b100-b200-and-gb200)
- [NVIDIA B200 specs & benchmarks — Spheron](https://www.spheron.network/blog/nvidia-b200-complete-guide/)
- [B200 vs H100 deep dive — Civo](https://www.civo.com/blog/comparing-nvidia-b200-and-h100)
- [Blackwell vs Hopper tensor-core GPUs — Exxact](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
- [Cornell CVW: Blackwell B200 architecture](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/blackwell_b200)
