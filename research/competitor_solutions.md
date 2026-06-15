# Competitor solutions — analysis (nvfp4 GEMM family, B200). Source: user-pasted winning code, 2026-06-15.
(Full code is in the conversation; this is the distilled, actionable synthesis. More solutions incoming.)

## ⭐⭐ LESSON 1 — SHIPPING: `load_inline` (raw CUDA + inline PTX) WORKS on the grader. OVERTURNS findings H1.
gau.nernst's `nvfp4_gemm` (NVIDIA) AND `modal_nvfp4_dual_gemm` (**Modal B200 — our exact infra**) both ship by:
```python
from torch.utils.cpp_extension import load_inline
load_inline("mod", cpp_sources="", cuda_sources=CUDA_SRC, is_python_module=False, no_implicit_headers=True,
  extra_cuda_cflags=["-O3","-gencode=arch=compute_100a,code=sm_100a","--use_fast_math",
                     "--expt-relaxed-constexpr","--relocatable-device-code=false"],
  extra_ldflags=["-lcuda"])
op = torch.ops.my_module.gemm   # registered via TORCH_LIBRARY in the CUDA source
```
⇒ **the grader has nvcc + builds sm_100a at runtime.** My H1 ("no nvcc / load_inline fails / Ninja required") was probed on the **Modal MIRROR (`modal_qr.py` image)**, NOT the real grader — wrong for the grader. This means: write the QR kernel in **RAW CUDA C++ + inline PTX** (tractable — opus already hand-wrote tcgen05 PTX), NO CuTe-DSL, NO cubin embedding.
⚠️ **MUST re-verify on the `qr` grader** (a minimal `load_inline` hello-kernel via popcorn `--mode test`) before committing — the qr comp *might* use a different image. This is the #1 next action once the submission rate-limit clears.
- "stream" ban: the winners' CUDA sources are substring-clean (`cp.async.bulk`, `shared::cluster`, `cudaFuncAttribute...` — no "stream"). So raw-CUDA tcgen05 is compatible.

## ⭐⭐ LESSON 2 — the "megakernel" = warp specialization + multi-stage mbarrier pipeline (NO side-streams!)
This is how the overlap/pipelining is done WITHOUT the banned streams. gau.nernst's `nvfp4_gemm` skeleton:
- 1 block = `BLOCK_M/32` epilogue warps + 1 TMA warp + 1 MMA warp (e.g. BLOCK_M=128 → 6 warps). `__launch_bounds__(BLOCK_M+2*WARP_SIZE)`.
- **TMA warp** (`warp_id==NUM_WARPS-2`, `elect_sync()`): for each k-tile, `cp.async.bulk.tensor.3d` A/B(/scales) into a **NUM_STAGES smem ring**, then `mbarrier.arrive.expect_tx`. Issues the first NUM_STAGES loads up front, then waits on the MMA-done mbarrier before reusing a stage → it runs AHEAD (this is look-ahead).
- **MMA warp** (`warp_id==NUM_WARPS-1`): `mbarrier_wait`(tma_phase) → loop `tcgen05.mma` (accumulate into TMEM) → `tcgen05.commit.mbarrier::arrive` (MMA-done). `enable_input_d = (first k) ? iter_k : 1` = accumulate-after-first (our K-loop ACCUMULATE pattern).
- **Epilogue warps** (`tid<BLOCK_M`): `mbarrier_wait`(mainloop) → `tcgen05.ld.sync.aligned.32x32b` TMEM→regs → activation/convert → vectorized `st.global`. Pipelined: prefetch next `tcgen05.ld` chunk while computing current.
- Phase tracking: `phase ^= 1` each time the ring wraps. NUM_STAGES = (smem_budget − static) / stage_bytes.
→ **The TMA-warp-runs-ahead + ring buffer IS the look-ahead pipeline I wanted, intra-kernel, no streams.** Maps to QR: a megakernel with a **panel warp** (sequential Householder) overlapped against **TMA+MMA warps** (trailing GEMM) via the same ring+mbarrier scheme → hides the 40% panel behind the trailing compute.

## LESSON 3 — the raw-PTX toolkit to copy (gau.nernst `nvfp4_gemm` is the cleanest reference)
All as `asm volatile` helpers (re-request gau.nernst nvfp4_gemm from chat when building):
- **mbarrier:** `mbarrier.init.shared::cta.b64`; `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` (spin-wait); `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`.
- **TMA:** host `cuTensorMapEncodeTiled` (rank-3, `CU_TENSOR_MAP_SWIZZLE_128B`, dtype `16U4_ALIGN8B` for fp4 — for us BF16 use the bf16 dtype); device `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes`. Cache hints `EVICT_FIRST/LAST/NORMAL`.
- **tcgen05:** `alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem],ncols`; `mma.cta_group::1.kind::...block_scale` (for us `.kind::f16` BF16); `cp.cta_group::1.32x128b.warpx4` (scales→TMEM); `ld.sync.aligned.{32x32b|16x128b|16x256b}.xN.b32` (TMEM→regs, N regs); `commit.cta_group::1.mbarrier::arrive::one`; `dealloc`.
- **Descriptors:** `desc_encode(x)=(x&0x3FFFF)>>4`; SMEM desc = `desc_encode(addr) | (desc_encode(SBO)<<32) | (1<<46) | (2<<61)` (128B-swizzle); i_desc bit layout for MMA (atype/btype/MMA_N>>3<<17/MMA_M>>7<<27). `elect_sync()` (one-thread-issue).

## LESSON 4 — 2-SM CTA pair = ~2× (advanced; adopt after a working 1-SM kernel)
gau.nernst dual_gemm + macto: `__cluster_dims__(2,1,1)`, `tcgen05.mma.cta_group::2` (MMA_M=256, two SMs cooperate), multicast TMA (`.multicast::cluster` + cta_mask), `tcgen05.commit...multicast::cluster`. macto also caches `cuTensorMapEncodeTiled` host-side per (ptr,shape) — relevant since QR re-runs many iters.

## LESSON 5 — precision
These are nvfp4 (4-bit block-scaled) → NOT our case. For QR FP32-exact: BF16 `tcgen05.mma.kind::f16` + the **BF16x9 9-pass Ozaki** (opus already built + validated this, rel 4.4e-7). Open question: does the qr grader's factor_rtol≈1e-3 permit fewer passes / a cheaper scheme? Risky for rankdef/nearrank — default to exact BF16x9.

## PLAN (revised by these lessons)
1. ⚠️ **Verify `load_inline` on the qr grader** (minimal hello-kernel, `--mode test`) — gates everything. (rate-limit pending.)
2. If yes → **raw-CUDA path** (drop CuTe-DSL): port opus's working tcgen05 BF16x9 MMA into gau.nernst's warp-spec + TMA-ring + mbarrier skeleton → a fast standalone trailing GEMM, ship via load_inline, A/B vs cuBLAS on v19's batched shape.
3. Then → the **QR megakernel**: add a panel warp; overlap the sequential panel with the TMA/MMA-pipelined trailing (the LESSON-2 ring). Target n512/n1024 (7/12 of qr_v2).
4. dual-GEMM/deltanet (incoming) for: the 2-SM pattern (#4) and the sequential-chunk-recurrence handling (deltanet ≈ our sequential panel).
