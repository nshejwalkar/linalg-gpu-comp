# Profiling Without Root & Root-Access Node Rental

*Research date: June 2026. Answers two questions for the GPU MODE `qr` B200 leaderboard competition.*
*Context: Modal managed B200 — no root, ncu/nsys blocked (ERR_NVGPUCTRPERM), proton profile mode unimplemented.*
*We are now writing custom Triton kernels and need kernel-internal metrics (occupancy, stalls, BW utilization).*

---

## Q1 — Kernel-Introspection WITHOUT Root

### The core problem

`ncu` (Nsight Compute) requires hardware perf counters, which since driver ≥418.43 (Linux) are gated behind
`CAP_SYS_ADMIN` or the kernel-module flag `NVreg_RestrictProfilingToAdminUsers=0`. Managed / multi-tenant
clouds (Modal, RunPod, Colab, most serverless) structurally cannot grant this — it is not a config tweak
on our end, it is enforced at the host level. `nsys` (Nsight Systems) can run basic CPU+CUDA timeline
tracing but its GPU-counter metrics (hardware occupancy, SM throughput %) are also blocked.

What we *can* do is use a layered stack of no-root tools that together give us most of the signal ncu would:

---

### Tool 1: Triton kernel attribute inspection (no root, zero overhead, always available)

After compiling a Triton kernel with `.warmup()`, the compiled object exposes hardware resource usage:

```python
kernel = my_triton_kernel.warmup(
    *args,
    BLOCK_SIZE=64, num_warps=4, num_stages=3,
    grid=(1,)
)
kernel._init_handles()

n_regs     = kernel.n_regs           # registers per thread
n_spills   = kernel.n_spills         # register spills to L2/DRAM — nonzero = bad
size_smem  = kernel.metadata.shared  # shared memory bytes per block
```

**Occupancy estimate** (NVIDIA Hopper/Blackwell, 64K regs per SM, 228 KB smem per SM on B200 cc10.0):

```python
WARP_SIZE    = 32
NUM_REGS_SM  = 65536     # B200 cc10.0: 64K 32-bit regs per SM
SIZE_SMEM_SM = 228 * 1024  # B200 cc10.0: 228 KB per SM (cc12.0: 128 KB)
MAX_WARPS_SM = 64        # B200 cc10.0

occ_reg  = NUM_REGS_SM // (n_regs * WARP_SIZE * num_warps)
occ_smem = SIZE_SMEM_SM // size_smem if size_smem > 0 else MAX_WARPS_SM
occ_warp = MAX_WARPS_SM // num_warps
occupancy = min(occ_reg, occ_smem, occ_warp)   # blocks per SM
```

**What it gives:** register pressure (spill risk), shared memory fit, theoretical occupancy — all the
"outer" occupancy signals from ncu's `Occupancy` section, without hardware counters.

**Does not give:** actual warp-stall reasons, memory-bandwidth utilization, instruction throughput SOL%.

**Needs root?** No.

**References:**
- [Triton fused-softmax tutorial (occupancy example)](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [TritonForge paper (uses n_regs/shared for tuning)](https://arxiv.org/html/2512.09196v1)

---

### Tool 2: `triton.testing.do_bench` (no root, latency + bandwidth SOL)

```python
import triton.testing

ms = triton.testing.do_bench(
    lambda: my_kernel[grid](a, b, out, ...),
    warmup=150,   # ms; default 25 ms is known to underestimate by ~30% (GitHub issue #2306)
    rep=500,      # ms
    return_mode="median"
)

# Compute bandwidth SOL (% of peak)
bytes_moved = A.nbytes + B.nbytes + out.nbytes
bw_achieved = bytes_moved / (ms * 1e-3) / 1e12   # TB/s
B200_PEAK_BW = 8.0  # TB/s HBM3e
bw_sol = bw_achieved / B200_PEAK_BW * 100

# Compute FLOP SOL
flops = 2 * M * N * K
B200_PEAK_TFLOPS_FP32 = 60.0   # ~60 TFLOPS FP32
flop_sol = (flops / (ms * 1e-3)) / (B200_PEAK_TFLOPS_FP32 * 1e12) * 100
```

**What it gives:** precise median latency, jitter range (20th/80th pct), and — when you compute
bytes-moved / time — the effective memory bandwidth utilization SOL% without any profiler access.
This is a cheap, root-free roofline point. Enough to tell: are we memory-bound, compute-bound,
or latency-bound (neither SOL is high)?

**Limitation:** `do_bench_cudagraph` variant avoids kernel-launch overhead noise for very small kernels.
Default warmup=25 ms is too short; **use warmup≥150 ms** (GitHub issue #2306).

**There is also `triton.testing.do_bench_cudagraph`** for capture+replay timing (avoids launch jitter).
Note: CUDA graphs are banned from *submissions* but legal in Modal profiling.

**Needs root?** No.

**Reference:**
- [triton.testing.do_bench API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
- [do_bench underestimates issue #2306](https://github.com/openai/triton/issues/2306)

---

### Tool 3: Triton `proton` profiler (no root, timing + user-annotated FLOPs/bytes)

Proton ships with Triton as `triton[profiling]`. It hooks into Triton's kernel-launch path and records
Python context, user-supplied FLOPs/bytes annotations, and kernel timing. Output is a Chrome trace or
tree-view via `proton-viewer`.

```python
import triton.profiler as proton

proton.start("qr_profile", hook="triton")

with proton.scope("panel_householder [n=512, block=64]",
                  {"flops": 2 * n * block**2, "bytes": n * block * 4 * 2}):
    my_panel_kernel[grid](A, tau, ...)

proton.finalize()
# Produces qr_profile.json; view with: proton-viewer qr_profile.json
```

The `hook="triton"` mode invokes `launch_metadata()` before each kernel launch, returning a dict
with kernel metadata. The tool registers spill counts and utilization warnings that the Triton compiler
surfaces. A forthcoming extension (not yet merged as of June 2026) will add intra-kernel profiling via
device-side buffers (Chrome trace output).

**What it gives:** per-kernel wall time with Python call-stack, manual FLOPs/bandwidth annotations
(you compute these), register-spill warnings from the Triton hook metadata. Better than torch.profiler
for Triton-centric workflows because it is Triton-aware (associates profile entries with Triton ops,
not underlying CUDA kernels).

**What it does NOT give:** warp-stall reasons, actual SOL%, hardware occupancy — anything that
requires CUPTI hardware counters.

**Needs root?** No (uses CUPTI activity API at the timing level only, not hardware counters).

**References:**
- [Proton repo (triton-lang/triton/third_party/proton)](https://github.com/triton-lang/triton/tree/main/third_party/proton)
- [CGO 2026 Proton paper](https://2026.cgo.org/details/cgo-2026-papers/23/Proton-Towards-Multi-level-Adaptive-Profiling-for-Triton)
- [Ian Barber: Profiling Triton (2025)](https://ianbarber.blog/2025/05/01/profiling-triton/)

---

### Tool 4: `torch.profiler` — features we are underusing

We already use torch.profiler for Kineto/CUPTI activity tracing (per-kernel CUDA time, launch counts).
Several flags we have NOT been exploiting:

| Flag | What it adds | Cost | Useful for us? |
|---|---|---|---|
| `record_shapes=True` | Input shapes attached to each op | small | Yes — essential to attribute bmm/gemm cost to specific panel sizes |
| `with_flops=True` | FLOPs estimate for matmul/conv ops | small | Yes — gives a computed-FLOPS figure to compare to B200 FP32 peak (roofline X-axis) |
| `profile_memory=True` | Tensor alloc/dealloc sizes, CUDA memory timeline | medium | Yes for memory debugging; less urgent now |
| `with_stack=True` | Python + TorchScript call stack per op | ~20% overhead | Useful once to attribute kernel launches to our Python panel loop |
| `schedule(wait=1, warmup=2, active=5)` | Skip compile/warmup noise | needed for accuracy | **Must use** — without this, first-step Triton compile lands in the trace |

**`with_flops` usage example:**

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    with_flops=True,
    record_shapes=True,
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=3)
) as prof:
    for step in range(6):
        run_kernel(A)
        prof.step()

print(prof.key_averages(group_by_input_shape=True)
        .table(sort_by="cuda_time_total", row_limit=20))
# Now shows "FLOPs" column for bmm/gemm ops — compare to B200 FP32 peak
```

**Limitation:** `with_flops` only estimates FLOPs for a whitelist of ops (matmul, conv). Our panel ops
(norm, mul, sub) get no FLOPs estimate. Still, this directly tells us FLOP utilization of the bmm calls.

**References:**
- [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Spheron 2026 profiling guide](https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/)

---

### Tool 5: TORCH_LOGS / TorchInductor for generated-kernel inspection

If we ever use `torch.compile()` or `torch._inductor` to lower PyTorch ops to Triton, these
environment variables expose the generated kernel code and individual-kernel benchmarks:

```bash
# See generated Triton kernel Python source
TORCH_LOGS="+inductor,output_code" python submission.py

# Unique kernel names in trace (e.g., triton_poi_fused_mul_23 instead of triton_)
TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python submission.py

# Benchmark each generated kernel individually (slow, but shows per-kernel cost)
TORCHINDUCTOR_BENCHMARK_KERNEL=1 python submission.py

# Tune more autotune configs (trades compile time for perf)
TORCHINDUCTOR_MAX_AUTOTUNE=1 python submission.py
```

**What it gives:** the actual Triton source for each fused kernel inductor generates, per-kernel
benchmark times, kernel categories (pointwise/reduction/persistent). Useful for understanding what
torch.compile does with our panel ops — compare with hand-written Triton to find missed fusions.

**Needs root?** No.

**Reference:**
- [TorchInductor GPU Profiling docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_inductor_profiling.html)

---

### Tool 6: CUPTI PC-sampling / metrics API

CUPTI's PC-sampling API (warp stall reasons, instruction-level latency) and its hardware-counter
metrics API are what ncu uses internally. Calling them directly from Python (via `pycupti` or ctypes)
hits the same `NVreg_RestrictProfilingToAdminUsers` gate as ncu does. There is no bypass: the check
is enforced in the NVIDIA kernel module, not in ncu itself.

**Verdict:** CUPTI hardware-counter path is not available without root/bare-metal. The CUPTI *activity*
API (kernel launch timestamps, memcpy sizes) does not need root and is what torch.profiler uses.

**Reference:**
- [CUPTI 13.0 docs](https://docs.nvidia.com/cupti/13.0.1/release-notes/release-notes.html)
- [NVIDIA ERR_NVGPUCTRPERM solution guide](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)

---

### Q1 Summary: What we can know without root

| Question | Tool | Root needed? |
|---|---|---|
| How long does each kernel take? | torch.profiler / do_bench | No |
| How many kernel launches? | torch.profiler (launch count) | No |
| Register count, spill count, smem size? | kernel.n_regs / .metadata.shared | No |
| Theoretical occupancy | Computed from above | No |
| FLOPs utilization of bmm? | torch.profiler with_flops + do_bench | No |
| Memory BW utilization (SOL%)? | Manually: bytes/time vs peak | No |
| Python→CUDA attribution | torch.profiler with_stack | No |
| Warp stall reasons (memory, sync, etc.)? | **ncu only** | **Yes** |
| Actual hardware SM occupancy? | **ncu only** | **Yes** |
| L1/L2 hit rates | **ncu only** | **Yes** |

For our current phase (fusing panel ops into a Triton kernel to kill the ~10k launch/iteration overhead),
the no-root tools give sufficient signal:
- `do_bench` to measure each kernel variant's latency
- kernel attribute inspection for occupancy and spill risk
- torch.profiler with_flops to confirm bmm is actually being used

The tools become insufficient if we hit a wall where a kernel *seems* well-structured but is still
underperforming and we can't explain why (e.g., suspected bank conflicts, unexpected L2 pressure, 
warp serialization on the shared-memory reductions inside the panel factorization).

---

## Q2 — Renting a Root-Access GPU Node for ncu/nsys

### Why ncu needs root (and what it takes to fix it)

The restriction lives in the NVIDIA kernel module. Two official fixes:

1. **Set as root on the host:** `echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf` then `modprobe -r nvidia && modprobe nvidia`. Persistent across reboots.
2. **Run ncu with CAP_SYS_ADMIN:** `docker run --cap-add=SYS_ADMIN ...` or on bare metal as root user.

Managed/serverless clouds (Modal, Colab, RunPod) **cannot** grant either because the kernel module is shared across tenants. Only bare-metal or VM-level access where *you* control the OS can fix this.

---

### Provider survey (June 2026)

#### RunPod
- **B200 availability:** Yes, B200 on Secure Cloud. ~$5.89–$7.99/hr per GPU.
- **H100 availability:** Yes, ~$2.69/hr on-demand.
- **Root/privileged?** Containers run as root inside the pod, but `--cap-add=SYS_ADMIN` is **not** granted. RunPod explicitly confirmed they block this for security. **ncu does NOT work on RunPod** — confirmed by community (ThunderKittens issue #73).
- **ncu usable?** **No.**
- **Sources:** [RunPod B200 guide](https://www.runpod.io/articles/guides/b200-ai-research), [ThunderKittens issue #73](https://github.com/HazyResearch/ThunderKittens/issues/73)

#### Vast.ai
- **B200 availability:** Yes, from ~$4.09/hr (marketplace, varies by host).
- **H100 availability:** Yes, various pricing.
- **Root/privileged?** Marketplace model — individual hosts control their machines. Some hosts may run privileged containers; Docker-in-Docker is **disabled** globally for security. Vast.ai has documented security vulnerabilities in its container isolation model. Whether a specific host sets `NVreg_RestrictProfilingToAdminUsers=0` is unpredictable and depends on the individual machine owner.
- **ncu usable?** **Inconsistent/uncertain** — depends on the specific host. Not guaranteed.
- **Sources:** [Vast.ai B200 pricing](https://vast.ai/pricing/gpu/B200), [Vast.ai FAQ](https://docs.vast.ai/documentation/reference/faq/technical)

#### Lambda Labs
- **B200 availability:** Yes, B200 SXM6 at $4.99–$6.69/hr/GPU (1×, 2×, 4×, 8× GPU options).
- **H100 availability:** Yes, $3.99/hr.
- **Root/privileged?** SSH access provided. Lambda docs describe **root SSH access to instances** ("users get SSH root access on bare-metal hardware"). Lambda Stack includes NVIDIA driver and CUDA pre-installed. If instances are truly bare-metal with root, the user can set `NVreg_RestrictProfilingToAdminUsers=0` and ncu should work.
- **ncu usable?** **Likely yes** on bare-metal instances with root SSH — needs verification on first boot.
- **Pricing to get ncu on B200:** ~$6.69/hr (single B200). H100 at $3.99/hr is significantly cheaper for a profiling session.
- **Sources:** [Lambda pricing](https://lambda.ai/pricing), [Lambda docs overview](https://docs.lambda.ai/public-cloud/on-demand/)

#### Verda (formerly DataCrunch)
- **B200 availability:** Yes, B200 SXM6 at **$3.99/hr** (cheapest confirmed B200 with profiling support found).
- **H100 availability:** Yes, $1.99/hr.
- **Root/privileged?** **Explicitly documented:** Verda allows querying GPU hardware counters with ncu and Nsight Systems. They set `NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/nvidia-profiling.conf` on their instances. They configure VMs so **profiling does not require sudo**.
- **ncu usable?** **Yes — explicitly confirmed by Verda documentation.**
- **Sources:** [Verda B200/B300 architecture post (profiling section)](https://verda.com/blog/nvidia-b200-and-b300-gpu-architecture-and-software-stack), [Verda pricing comparison](https://verda.com/blog/cloud-gpu-pricing-comparison)

#### Nebius
- **B200 availability:** Yes. Pricing listed at ~$3.95–$5.50/hr/GPU depending on source/date (pricing volatile June 2026).
- **H100 availability:** Yes, H100 SXM at ~$3.98/hr.
- **Root/privileged?** Nebius advertises "bare-metal class performance" and self-serve GPU instances. No explicit documentation found confirming `NVreg_RestrictProfilingToAdminUsers=0` pre-configured. Being VM-based or bare-metal affects this.
- **ncu usable?** **Unknown — not documented.** Contact needed.
- **Sources:** [Nebius pricing on ComputePrices.com](https://computeprices.com/providers/nebius)

#### Voltage Park
- **B200 availability:** Offers "Hopper, Blackwell and Grace Blackwell GPUs." H100 at $1.99/hr confirmed. B200 pricing not explicitly listed (sales/contract model for Blackwell).
- **H100 availability:** Yes, **$1.99/hr** — cheapest H100 found.
- **Root/privileged?** Explicitly provides **bare-metal access for direct hardware control.** This means root access on Linux, which allows setting the NVreg flag. 
- **ncu usable?** **Likely yes** on H100 bare-metal. B200 availability and pricing unclear.
- **Sources:** [Voltage Park review 2026](https://tooldirectory.ai/tools/voltage-park), [Voltage Park neocloud](https://www.voltagepark.com/neocloud)

#### Crusoe
- **B200 availability:** H100/H200 confirmed; B200 capacity building out (stranded-gas model).
- **H100 availability:** Yes (pricing by request / contract-focused).
- **Root/privileged?** A Crusoe support article specifically covers enabling `NVreg_RestrictProfilingToAdminUsers=0` for non-admin users — **strongly implies they offer root or at least a configured environment for ncu.**
- **ncu usable?** **Likely yes** — Crusoe support article specifically addresses enabling GPU performance counters for non-admin users.
- **Sources:** (Crusoe support article returned HTTP 403; referenced via search results confirming the article exists)

#### Spheron
- **B200 availability:** Yes, B200 SXM6 spot from ~$2.12/hr, on-demand ~$6.02/hr.
- **H100 availability:** Yes, from ~$1.43–$2.01/hr.
- **Root/privileged?** Explicitly claims "Running these tools on bare-metal H100 instances on Spheron gives you full ncu and nsys access without the ERR_NVGPUCTRPERM error." Bare-metal model with NVIDIA capability pass-through to container runtime.
- **ncu usable?** **Yes — explicitly stated in Spheron documentation/blog.**
- **Sources:** [Spheron GPU profiling guide (2026)](https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/)

#### Atlas Cloud
- **B200 availability:** Mentioned in hardware list; not in the live pricing table (H100/H200/GB200 confirmed).
- **H100 availability:** Yes, $1.95–$2.40/hr bare-metal.
- **Root/privileged?** "Bare Metal GPU Servers" — bare metal implies root access.
- **ncu usable?** **Likely yes** — bare metal. B200 availability unclear.
- **Sources:** [Atlas Cloud bare metal](https://www.atlascloud.ai/bare-metal)

#### Together AI / Hyperbolic
- **B200 availability:** Together AI yes (HGX B200 clusters, pricing unclear/enterprise). Hyperbolic: H100/H200 only as of June 2026.
- **Root/privileged?** Together: cluster/managed offering, likely not bare-metal root for single tenants. Hyperbolic: SSH access provided, root likely.
- **ncu usable?** **Unknown / probably no** for Together (managed orchestration). Hyperbolic possible with SSH root.
- **Sources:** [Together AI GPU clusters](https://www.together.ai/gpu-clusters), [Hyperbolic homepage](https://www.hyperbolic.ai/)

---

### Provider comparison table

| Provider | B200 $/hr | H100 $/hr | Root/bare-metal? | ncu works? | Notes |
|---|---|---|---|---|---|
| **Verda** | $3.99 | $1.99 | VM (profiling pre-configured) | **Yes (documented)** | Best ncu+B200 option; sets NVreg=0 |
| **Lambda Labs** | $4.99–$6.69 | $3.99 | Bare-metal root SSH | **Likely yes** | Root SSH; set NVreg=0 manually |
| **Voltage Park** | ~$2/hr (H100) | **$1.99** | Bare-metal | **Likely yes (H100)** | Cheapest H100+root; B200 unclear |
| **Spheron** | $2.12 spot / $6.02 OD | $1.43+ | Bare-metal cap pass-through | **Yes (documented)** | Explicitly claims ncu works |
| **Crusoe** | TBD | By request | Root implied | **Likely yes** | Support article covers NVreg fix |
| **Nebius** | ~$3.95–$5.50 | ~$3.98 | Unclear | **Unknown** | Contact required |
| **Vast.ai** | ~$4.09 | Varies | Host-dependent | **Inconsistent** | Marketplace; no guarantees |
| **RunPod** | $5.89–$7.99 | $2.69 | No (CAP_SYS_ADMIN blocked) | **No (confirmed)** | Avoid for profiling |
| **Atlas Cloud** | Not listed | $1.95–$2.40 | Bare-metal | **Likely yes (H100)** | H100 bare-metal confirmed |

---

### Do we need B200 for ncu, or would H100 give transferable insights?

**Short answer: H100 with ncu gives ~80% of the insight, at 2–4× lower cost, and is far more widely available.**

**Architecture comparison (Hopper H100 vs Blackwell B200):**

| Parameter | H100 (Hopper, cc9.0) | B200 (Blackwell, cc10.0) |
|---|---|---|
| SMs | 132 | 160 |
| Max warps/SM | 64 | 64 |
| Registers/SM | 64K 32-bit | 64K 32-bit |
| Shared mem/SM | 228 KB (max) | 228 KB/SM (cc10.0), 128 KB (cc12.0) |
| HBM BW | 3.35 TB/s (H100 SXM) | 8.0 TB/s |
| FP32 TFLOPS | ~60 | ~134 |
| L2 cache | 50 MB | 126 MB |
| New B200-only features | — | Thread block clusters up to 16, TMEM, TMA enhancements |

**What transfers from H100 ncu to B200 tuning:**
- Register count and spill diagnosis — **directly transfers** (same 64K regs/SM, same WARP_SIZE=32)
- Shared memory bank conflict analysis — **transfers** (same 128-byte banks, same access patterns)
- Warp stall categories (mem dependency, sync, etc.) — **transfers conceptually**, proportions shift
- Whether a kernel is memory-bound vs compute-bound — **transfers as a pattern**, numbers scale
- Tile size / block size decisions for occupancy — **transfers** (same constraints)

**What does NOT transfer cleanly:**
- Absolute bandwidth utilization (B200 HBM3e is 2.4× faster — a memory-bound kernel behaves differently)
- L2 hit rates (B200 has 2.5× larger L2)
- TMA / TMEM usage (Blackwell-specific features)
- Warp-stall *ratios* in pipelined kernels (B200 has better instruction-level parallelism)

**Verdict:** For our Householder panel kernel, the critical question is "why are warp schedulers stalling?"
— is it memory latency, register pressure, bank conflicts, or synchronization? These are architectural
patterns that are **fundamentally the same on H100 and B200**. The absolute numbers differ (B200 has
lower memory-latency pressure due to faster HBM) but the *diagnosis* of what to fix transfers directly.
H100 ncu is sufficient to guide optimization; the fix will then be tested on B200 via Modal.

---

## Recommendation

### Verdict: Stick with torch.profiler + Triton tools now; rent only if we hit a real wall.

**Reasoning:**

1. **We have not exhausted the no-root tools.** We currently use torch.profiler only for launch-count and
   GPU wall time. We have NOT used: `with_flops=True` (to see bmm FLOP utilization), `record_shapes=True`
   (to attribute cost to specific panel sizes), `with_stack=True` (to trace which Python call causes each
   kernel), or computed bandwidth SOL% from do_bench. These together give a near-complete picture of
   *where* the bottleneck is at the kernel-type level.

2. **Triton kernel attribute inspection is free and immediate.** Once we write our fused panel Triton kernel,
   we can inspect `n_regs`, `n_spills`, `metadata.shared` directly, compute occupancy, and confirm we are
   not spilling or over-using shared memory before even timing.

3. **ncu adds warp-stall detail we don't need yet.** The main optimization target is clear: fuse the ~10k
   tiny panel ops into one Triton kernel. That is an architectural fix, not a microarchitecture fix.
   ncu's warp-stall analysis is the right tool *after* the fused kernel exists and is still underperforming.

4. **When to rent:** Rent when we have a Triton kernel that runs correctly and fast, but its throughput
   (measured via do_bench + manual BW SOL%) is significantly below roofline with no obvious fix from
   the no-root tools. That is the point where warp-stall reasons and actual occupancy from ncu matter.

### If/when we rent: **Verda H100 first, then B200 if needed.**

- **Verda H100 at $1.99/hr:** Cheapest confirmed ncu-capable option. Verda explicitly pre-configures
  `NVreg_RestrictProfilingToAdminUsers=0`. Profile the panel kernel here. H100 insights transfer ~80%
  to B200 for the register/shared-memory/stall diagnosis we need.
- **Verda B200 at $3.99/hr** (or Lambda B200 at ~$5.00/hr): Only if H100 insights suggest a
  Blackwell-specific behavior (e.g., TMA, async pipelining, TMEM) that we want to confirm on the
  actual target. A 1-2 hour ncu session costs $4–$10 — entirely reasonable.
- **Voltage Park H100 at $1.99/hr** is another option with confirmed bare-metal, though it may require
  more setup (setting NVreg manually as root, installing Nsight Compute separately).
- **Avoid RunPod** for profiling (ncu blocked). RunPod is fine for iteration/testing.

### Concrete rental workflow when the time comes

```bash
# On Verda H100 instance (ncu already enabled, no sudo needed):
pip install triton torch   # or use their pre-built image

# Profile the fused panel kernel:
ncu --set full -o panel_profile.ncu-rep python -c "
import torch, triton
# [paste kernel + bench code]
"

# Key sections to look at first:
# 1. GPU Speed of Light → are we compute-bound or memory-bound?
# 2. Occupancy → theoretical vs achieved occupancy
# 3. Warp State Statistics → which stall reason dominates?
# 4. Memory Workload Analysis → L1/L2/HBM utilization

# Transfer the .ncu-rep report file locally and open in Nsight Compute UI
```

**Estimated cost for a full profiling session:** 2–4 hours on Verda H100 = $4–$8. If B200 needed: $8–$16. Entirely worth it once we have a candidate kernel.

---

## Sources

- [Proton CGO 2026 paper](https://2026.cgo.org/details/cgo-2026-papers/23/Proton-Towards-Multi-level-Adaptive-Profiling-for-Triton)
- [Triton proton GitHub (third_party/proton)](https://github.com/triton-lang/triton/tree/main/third_party/proton)
- [Ian Barber: Profiling Triton (2025)](https://ianbarber.blog/2025/05/01/profiling-triton/)
- [TritonForge paper (profiling-guided Triton optimization)](https://arxiv.org/html/2512.09196v2)
- [Triton fused-softmax tutorial (n_regs/occupancy code)](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [triton.testing.do_bench API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
- [do_bench warmup underestimate issue #2306](https://github.com/openai/triton/issues/2306)
- [Triton persistent matmul tutorial (proton context manager)](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
- [TorchInductor GPU Profiling docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_inductor_profiling.html)
- [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA ERR_NVGPUCTRPERM solution](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
- [CUPTI 13.0 docs](https://docs.nvidia.com/cupti/13.0.1/release-notes/release-notes.html)
- [NVIDIA Blackwell Tuning Guide (B200 SM/smem specs)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [Spheron GPU profiling 2026 guide](https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/)
- [Spheron B200 cloud pricing 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [B200 cloud pricing comparison (getdeploying.com)](https://getdeploying.com/gpus/nvidia-b200)
- [Verda B200/B300 architecture post (ncu profiling confirmed)](https://verda.com/blog/nvidia-b200-and-b300-gpu-architecture-and-software-stack)
- [Verda cloud GPU pricing comparison](https://verda.com/blog/cloud-gpu-pricing-comparison)
- [Lambda Labs GPU pricing](https://lambda.ai/pricing)
- [Lambda Labs on-demand docs](https://docs.lambda.ai/public-cloud/on-demand/)
- [RunPod B200 guide](https://www.runpod.io/articles/guides/b200-ai-research)
- [RunPod review (ncu blocked confirmed)](https://deploybase.ai/articles/runpod-review)
- [ThunderKittens issue #73 (RunPod ncu blocked)](https://github.com/HazyResearch/ThunderKittens/issues/73)
- [Vast.ai B200 pricing](https://vast.ai/pricing/gpu/B200)
- [Voltage Park review 2026](https://tooldirectory.ai/tools/voltage-park)
- [Voltage Park neocloud (bare-metal claim)](https://www.voltagepark.com/neocloud)
- [Nebius GPU pricing (ComputePrices)](https://computeprices.com/providers/nebius)
- [Atlas Cloud bare-metal GPUs](https://www.atlascloud.ai/bare-metal)
- [Together AI GPU clusters](https://www.together.ai/gpu-clusters)
- [Hyperbolic GPU rental](https://www.hyperbolic.ai/)
- [Red Hat: Triton kernel profiling with Nsight tools](https://next.redhat.com/2025/11/19/triton-kernel-profiling-with-nvidia-nsight-tools/)
- [DeepSeek DeepGEMM issue #46 (ncu in cloud)](https://github.com/deepseek-ai/DeepGEMM/issues/46)
