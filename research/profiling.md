# Profiling on Modal — what works, what doesn't, and the tricks

## What does NOT work
- **Nsight Compute (ncu) metrics** need GPU hardware performance counters, which
  require privileged access (`--privileged` / `--cap-add=SYS_ADMIN`,
  `NVreg_RestrictProfilingToAdminUsers=0`). Managed multi-tenant Modal doesn't grant
  this → ncu dies with `ERR_NVGPUCTRPERM`. No SOL/roofline/occupancy from ncu.
- **Nsight Systems (nsys)** uses CUPTI tracing; it *can* run in containers but several
  features need `--cap-add=SYS_ADMIN`/privileged, and extracting the `.qdrep`/`.nsys-rep`
  from Modal is awkward. Unreliable here — don't depend on it.
- **popcorn-cli `--mode profile`** is **not implemented for the qr task** —
  `reference_repo/eval.py` returns *"profile mode is not implemented for qr eval.py"*.

## What DOES work: torch.profiler (Kineto, CUPTI tracing)
CUPTI *tracing* (timeline + per-kernel durations) does NOT need counter privileges, so
`torch.profiler` works on Modal. It's wired into `modal_qr.py --mode profile` and prints
`key_averages().table(sort_by="cuda_time_total")` — per-kernel CUDA self time, call
counts (→ launch overhead), and CPU vs GPU time. Enough to answer our real questions.

How to read it for THIS kernel:
- **Which kernel dominates `cuda_time_total`?** GEMM (`*gemm*`/`bmm`) vs the panel's
  many small elementwise/GEMV kernels. Tells us where to spend effort.
- **High `# of Calls` on tiny kernels** = launch-bound. The blocked loop fires
  ~`(n/b) × b` Householder-step kernels + per-block GEMMs; for small n this overhead
  dominates → fix with fewer launches / fusion / CUDA graphs.
- **Gap between total wall time and summed CUDA time** = CPU/dispatch-bound (Python
  loop, kernel launch latency). Big gap → CUDA graphs will help a lot.
- Add `torch.cuda.nvtx.range_push/pop` around panel / WY / trailing phases to label
  them in the trace. Export a chrome trace (`prof.export_chrome_trace`) and open in
  Perfetto / chrome://tracing for the timeline if the table isn't enough (can return
  the bytes from the Modal fn or stash in a modal.Volume).

## The big optimization the profiler will point at: CUDA graphs
For each fixed benchmark shape, the blocked-QR call graph is **static** (same kernels,
same shapes every iteration). Capturing it once as a CUDA graph and replaying collapses
all Python + dispatch + per-kernel launch overhead into a single `cudaGraphLaunch`.
- Best payoff exactly where geqrf is weakest: the **launch-bound small/medium cases**
  (n=32/176/352) and the headline **b640 n512**.
- PyTorch APIs: `torch.cuda.graph(g)` context manager, or `torch.cuda.make_graphed_callables`.
- Constraints: static input tensor *address* (copy the incoming `A` into a persistent
  static buffer, replay, read static outputs), static shapes (one captured graph per
  distinct `(batch, n)` — cache them in a dict), no CPU-dependent control flow inside
  the captured region, no data-dependent shapes. Our blocked loop has fixed trip count
  for a given n, so it captures cleanly.
- Warm up a few iterations before capture (cuBLAS/Triton allocate workspaces/autotune).

## Always-available coarse timing
CUDA events + `clear_l2_cache()` between iters (what `eval.py` and our `_bench_fn` use)
for wall-clock per-shape numbers. For attributing time within a call, use the profiler.

## If we ever truly need ncu
Only path is a non-managed B200 box (RunPod/Lambda/own VM) where you control profiling
permissions. Almost certainly unnecessary — our bottlenecks (batch parallelism, panel
vs GEMM split, launch overhead) are all visible in torch.profiler + event timing.

## Sources
- [Accelerating PyTorch with CUDA Graphs — PyTorch blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [PyTorch CUDA Graph Capture — Lei Mao](https://leimao.github.io/blog/PyTorch-CUDA-Graph-Capture/)
- [Kernel Batching with CUDA Graphs (arXiv 2501.09398)](https://arxiv.org/pdf/2501.09398)
- [Nsight Systems in Docker — Lei Mao](https://leimao.github.io/blog/Docker-Nsight-Systems/)
- [Using Nsight Compute in containers — NVIDIA](https://developer.nvidia.com/blog/using-nsight-compute-in-containers)
