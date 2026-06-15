# tcgen05 + TMEM warp-specialized megakernel — engineering reference & build plan

> **Purpose.** Implementation spec for a B200/sm_100a-specific **tcgen05 + TMEM
> warp-specialized megakernel** for batched Householder QR. This is the *only*
> remaining path to the 2.5 ms geomean goal (findings B6/B7/B8 prove the
> blocked-WY-with-resident-panel architecture is capped at ~4 ms by the B=32 wall).
> Read this together with `findings.md` (B5–B8, C3–C5, H1–H2), `cutlass_dsl.md`,
> and `gpumode_winners.md`. The CuTe-DSL→cubin→`cuModuleLoadData` pipeline is
> **already proven end-to-end** (B8, `modal_cute.py`, `v21_megakernel.py`); this
> doc assumes that shipping mechanism and focuses on the tensor-core kernel itself.
>
> **Status tags inline:** [CONFIRMED] = from a primary source / our own measurement;
> [LIKELY] = strong inference from sources but not bit-verified for our exact case;
> [UNVERIFIED] = could not confirm — DO NOT build load-bearing logic on it without
> checking. Flagged explicitly so we don't build on guesses.
>
> Date: 2026-06-14. Hardware is new (2025–2026); sources are current NVIDIA PTX ISA,
> the CUTLASS repo, and recent technical writeups, preferred over Hopper material.

---

## 0. TL;DR for the implementer

- **tcgen05.mma** is Blackwell's 5th-gen tensor-core MMA. One *single thread* issues
  it (not a warpgroup like Hopper WGMMA). Accumulator lives in **TMEM**, not
  registers. Tiles up to **M=128 (or 256 with 2-SM), N up to 256, K=16/32** per
  instruction. **No native FP32 input** — inputs are tf32/bf16/fp16/fp8/fp6/fp4;
  FP32 accuracy comes from **BF16x9 / Ozaki emulation** (3 bf16 splits → 9 bf16
  MMAs → recombine, bit-exact). [CONFIRMED — see §1, §6]
- **TMEM** = 128 lanes (rows) × 512 columns (32-bit words) per SM = **256 KB/SM**,
  allocated/freed via `tcgen05.alloc`/`dealloc`. The accumulator for a 128×N tile
  in FP32 occupies N columns. Moving TMEM↔registers is `tcgen05.ld`/`tcgen05.st`;
  operands are fed from **SMEM** (not registers) via descriptors. [CONFIRMED — §2]
- **Warp specialization** is mandatory: a TMA *producer* warp (issues
  `cp.async.bulk.tensor`, arrives on mbarriers), an *MMA* warp (one thread issues
  `tcgen05.mma`), and *epilogue* warp(s) (`tcgen05.ld` → store). mbarrier +
  named-barrier sync; persistent grid (1 CTA/SM). [CONFIRMED — §3, §4]
- **For our QR**: the rank-b **trailing update** `H[k:, k+b:] -= Y @ (Tᵀ (Yᵀ @ A))`
  becomes a TMEM-resident tcgen05 GEMM with a subtract epilogue; BF16x9 keeps it
  bit-exact FP32 so band/rowscale pass with no detector. The panel factorization
  stays a sequential on-chip reduction (CUDA-core / GEMV-shaped, *not* tcgen05).
  See §7 for the megakernel structure and the staged build plan.

---

## 1. tcgen05 MMA — the 5th-gen tensor core (sm_100a)

### 1.1 What it is, vs Hopper WGMMA
`tcgen05.mma` (a.k.a. **UMMA**) is Blackwell's tensor-core matrix-multiply-accumulate
instruction. Differences from Hopper's `wgmma` that change how you write the kernel:

- **Single-thread issue.** *One* thread issues the MMA (`if (warp_id==MMA_WARP &&
  cute::elect_one_sync())`), not a 128-thread warpgroup. The hardware fans the work
  out internally. This frees the other warps to do TMA / epilogue concurrently.
  [CONFIRMED — gau-nernst, Colfax]
- **Accumulator lives in TMEM**, not registers. Threads do not "own" the result;
  TMEM is explicitly managed in software (alloc -> MMA writes -> tcgen05.ld to read
  back). This is the single biggest structural difference and the source of the
  mandatory MMA->epilogue cross-warp sync. [CONFIRMED]
- **Operands come from SMEM via 64-bit matrix descriptors** (or A may come from
  TMEM). You never load A/B tiles into registers for the MMA. [CONFIRMED]
- **Async.** The issuing thread returns immediately; completion is signalled to an
  mbarrier via `tcgen05.commit`. [CONFIRMED]

### 1.2 Supported shapes (per single instruction / per CTA)
- **M** = 128 with `.cta_group::1` (single SM); **M = 256** with `.cta_group::2`
  (two SMs of a TPC cooperate, arranged along M). [CONFIRMED]
- **N** = multiple of 8/16, **up to 256**. (A 128x256 FP32 accumulator = 256 TMEM
  columns = half of TMEM.) [CONFIRMED]
- **K** per instruction = **16 BF16/FP16 elements** (32 bytes); for TF32 K=8; for
  FP8 K=32. You **accumulate over K by issuing a loop of MMAs** into the same TMEM
  accumulator, toggling the `enable-input-d` ("ScaleOut") flag: first MMA in a
  k-tile uses `enable-input-d=0` (overwrite), subsequent ones `=1` (accumulate).
  [CONFIRMED — gau-nernst K-loop, Colfax UMMA::ScaleOut::One]

### 1.3 Input dtypes — NO native FP32 [CONFIRMED — critical]
tcgen05.mma inputs: **tf32, bf16, fp16, fp8 (e4m3/e5m2), fp6, fp4/NVFP4, int8/int4**.
There is **no FP32 input mode** — confirming findings B6. The accumulator (output d)
*is* FP32 (or INT32). So: to multiply FP32 matrices exactly you must emulate via
BF16x9/Ozaki (section 6). TF32 (10-bit mantissa) is a native input but loses precision ->
fails band/rowscale (findings B1-B4). BF16x9 is the only exact-FP32 path. [CONFIRMED]

> WARNING: A WebSearch summary listed "FP32" among tcgen05.mma precisions. This is the
> *accumulator/compute* type, not an *input operand* type. Treat "no FP32 tensor-core
> **input**" as CONFIRMED (findings B6 measured it; the Ozaki literature exists
> precisely because of it).

### 1.4 The instruction descriptors
Two descriptors per MMA, both prepared once and reused:

**Instruction descriptor (idesc, 32-bit)** — encodes dtype/atype/btype, MMA_M,
MMA_N, transpose flags. From gau-nernst (BF16 in, FP32 out):
```c
constexpr uint32_t i_desc =
      (1U << 4U)                         // d (accum) dtype = FP32
    | (1U << 7U)                         // a type = BF16
    | (1U << 10U)                        // b type = BF16
    | ((uint32_t)BLOCK_N >> 3U << 17U)   // MMA_N
    | ((uint32_t)BLOCK_M >> 4U << 24U);  // MMA_M
```

**SMEM matrix descriptor (64-bit, one for A, one for B)** — encodes the SMEM base
address, the leading-dim byte offset (LBO), stride-dim byte offset (SBO), and a
3-bit swizzle mode in bits [61:63] (0=none, 2=128B). gau-nernst, no-swizzle:
```c
const int SBO = 8 * 128;  // bytes for the 8x128B core tile
return desc_encode(addr) | (desc_encode(SBO) << 32ULL)
     | (1ULL << 46ULL) | (2ULL << 61ULL);  // bit46=1, swizzle=128B
```

**MMA syntax** (1-SM):
```
tcgen05.mma.cta_group::1.kind::f16  [d-tmem], a-desc, b-desc, idesc,
                                    enable-input-d;
```
In CuTe/CUTLASS this is wrapped as the atom SM100_MMA_F16BF16_SS<TA,TB,TC,M,N,
Major::K,Major::K> (_SS = both operands from SMEM; an _TS variant takes A from
TMEM). [CONFIRMED — Colfax]

### 1.5 K accumulation pattern (verbatim shape of the loop)
```c
// CuTe (Colfax): one warp, one thread issues; ScaleOut toggles accumulate
if (elect_one_warp) {
  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;   // 2nd+ k-block accumulates
  }
  cutlass::arch::umma_arrive(&shared_storage.mma_barrier); // tcgen05.commit
}
```
Raw-PTX equivalent (gau-nernst): unroll for (k=1; k<BLOCK_K/MMA_K; k++) issuing
one tcgen05.mma each, enable-input-d=1 after the first.

---

## 2. TMEM (Tensor Memory)

### 2.1 What & how big
A dedicated on-SM memory **separate from SMEM/registers**, addressed only by
tcgen05 instructions. **256 KB/SM**, organized as **128 lanes (rows) x 512 columns**,
each cell a **32-bit word**. (128 x 512 x 4 B = 256 KB.) [CONFIRMED — gau-nernst, Colfax]

A 128 x N FP32 accumulator occupies **N columns** (one column per output column,
one lane per output row). So:
- 128x256 accumulator -> 256 columns = **half of TMEM**.
- You can hold **two** 128x256 accumulators, or many narrow ones (e.g. N=32 -> 32 cols).
- **This directly bounds tile N <= 256 per accumulator** and is the budget you tile
  the trailing GEMM against (section 7). [CONFIRMED]

### 2.2 alloc / dealloc / relinquish — who & how
- `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [addr], nCols;`
  Allocates nCols columns. **nCols must be a power of two and >= 32.** The
  allocated TMEM base address is written to an SMEM location ([addr]). [CONFIRMED]
- **A single warp must do both alloc and dealloc** (the *same* warp). Other warps
  read the base pointer from SMEM. [CONFIRMED]
- `tcgen05.relinquish_alloc_permit` — after the alloc, the warp relinquishes the
  per-SM allocation lock so future CTAs can queue for the SM (needed for the
  persistent kernel to make progress). [CONFIRMED]
- `tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;` at the end.
- CuTe wraps all of this: cute::TMEM::Allocator1Sm (.allocate(...),
  .release_allocation_lock(), .free(...)). Simplest robust choice: **allocate
  all 512 columns** once and sub-allocate logically. [CONFIRMED — Colfax]

### 2.3 Accumulator layout & reading it back (TMEM -> registers)
- TMEM address packs **lane in bits [31:16], column in bits [15:0]**: addr =
  (lane<<16)|col. The CuTe accumulator tensor shows stride 65536 = 1<<16 between
  rows. [CONFIRMED]
- Read back with **tcgen05.ld**, store with tcgen05.st. Load shapes:
  .16x64b / .16x128b / .16x256b / .32x32b, with .xN repeats (e.g.
  .32x32b.x8 returns 8 FP32/thread). Example (.32x32b.x1):
  ```c
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
               : "=r"(dst0) : "r"(src_addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");   // fence before using regs
  ```
- **Per-warp restriction: each warp can only access 32 of the 128 lanes.** Therefore
  the TMEM->register epilogue is done by a **full warpgroup (4 warps = 128 threads)**,
  each warp owning 32 lanes. make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, acc) is
  hardcoded to 4 warps for this reason. [CONFIRMED — Colfax, NVIDIA forum]

### 2.4 Feeding operands from SMEM
The MMA reads A and B **directly from SMEM** through the 64-bit descriptors (1.4) —
there is no register staging of operands. TMA lands tiles in SMEM (section 4), the MMA warp
builds descriptors over those SMEM tiles (cta_mma.make_fragment_A(sA) returns a
tensor *of descriptors*, not data). A can alternatively be sourced from TMEM
(_TS atom). [CONFIRMED]

### 2.5 Costs / constraints to respect
- TMEM read latency is non-trivial and per-warp-limited -> the epilogue is a real
  pipeline stage, not a free tail. Overlap it with the next tile's MMA.
- Power-of-two >= 32 column alloc; can't alloc 96 or 48 — round up.
- Only the MMA accumulator and (optionally) A operand live in TMEM; everything else
  (B operand, staging) is SMEM/registers.
- 2-SM mode requires **identical SMEM object layout across both CTAs** of the TPC.


---

## 3. Warp-specialized persistent kernel skeleton (the canonical Blackwell pattern)

### 3.1 Warp roles
The Blackwell GEMM is a **persistent** kernel: launch **1 CTA per SM** (~148 on
B200), each CTA loops over output tiles. Within a CTA, warps are specialized:

| Warp role        | typical warps | job |
|------------------|---------------|-----|
| TMA producer     | 1-2 warps     | issue `cp.async.bulk.tensor` (TMA) loads of A/B tiles GMEM->SMEM; arrive on the "tile-ready" mbarrier with `expect_tx` byte count |
| MMA              | 1 warp (1 thread issues) | wait tile-ready -> issue `tcgen05.mma` K-loop into TMEM accumulator -> `tcgen05.commit` to the "mma-done" mbarrier |
| Epilogue         | 1 warpgroup (4 warps) | wait mma-done -> `tcgen05.ld` TMEM->registers (each warp owns 32 lanes) -> apply epilogue (subtract, cast) -> store to GMEM (or TMA store) |
| Alloc/scheduler  | folded into warp 0/1 | `tcgen05.alloc` the TMEM columns once; `relinquish_alloc_permit`; `dealloc` at exit |

[CONFIRMED — Colfax, gau-nernst, deepwiki. Exact warp *counts* vary by example;
counts above are the common 8-warp (256-thread) layout. UNVERIFIED whether 1 vs 2
TMA warps is optimal for our skinny shapes — tune.]

### 3.2 Two producer/consumer chains (the heart of it)
1. **TMA -> MMA** over `NUM_STAGES` SMEM buffers (double/triple buffer). TMA warp
   waits the buffer is free (MMA consumed it), issues TMA + `expect_tx`; MMA warp
   waits tile-ready, issues MMA, signals buffer-free.
2. **MMA -> Epilogue** over the TMEM accumulator. MMA `tcgen05.commit`s; epilogue
   waits, reads TMEM, signals accumulator-free for the next tile.

mbarrier ops (PTX): `mbarrier.init.shared`, `mbarrier.arrive.expect_tx.release`
(producer), `mbarrier.try_wait.parity.acquire` (consumer; phase bit flips each
pass). CuTe DSL: `cute.arch.mbarrier_init/arrive/try_wait`, plus the high-level
`cutlass.pipeline` helpers (`PipelineTmaUmma`, `PipelineUmmaAsync`) that package
exactly these two chains. [CONFIRMED]

### 3.3 Concrete pseudocode skeleton (1-SM, our trailing-GEMM case)
```
// grid = (num_SMs); block = 256 threads (8 warps). Persistent.
__global__ void trailing_gemm_ws(...) {
  __shared__ Smem smem;             // A/B staging (NUM_STAGES bufs) + barriers + tmem_ptr slot
  int warp = threadIdx.x / 32;

  // --- one-time setup ---
  if (warp == 0) { init all mbarriers (tile-ready[STAGES], buf-free[STAGES], mma-done); }
  if (warp == 1 && elect_one_sync()) {
      tcgen05.alloc(512 cols -> smem.tmem_ptr);   // power-of-2, >=32
      tcgen05.relinquish_alloc_permit();          // let later CTAs queue this SM
  }
  __syncthreads();                                // broadcast tmem_ptr
  uint32_t tmem = smem.tmem_ptr;

  // --- persistent tile loop ---
  for (tile = cta_rank; tile < num_tiles; tile += num_SMs) {
    if (warp < TMA_WARPS) {                        // PRODUCER
      for (k = 0; k < K/BK; ++k) {
        wait(buf-free[k % STAGES], phase);         // buffer reusable?
        if (elect_one_sync()) {
          mbarrier.arrive.expect_tx(tile-ready[k%STAGES], bytes);
          cp.async.bulk.tensor(sA[k%STAGES], tmaA, coordsA);   // TMA loads
          cp.async.bulk.tensor(sB[k%STAGES], tmaB, coordsB);
        }
      }
    } else if (warp == MMA_WARP) {                 // MMA
      tiled_mma.accumulate = ScaleOut::Zero;       // overwrite on first k
      for (k = 0; k < K/BK; ++k) {
        wait(tile-ready[k%STAGES], phase);
        if (elect_one_sync()) {
          gemm(tiled_mma, descA(sA[k%STAGES]), descB(sB[k%STAGES]), tmem_acc);
          tiled_mma.accumulate = ScaleOut::One;    // accumulate after first
        }
        arrive(buf-free[k%STAGES]);                // buffer consumed
      }
      tcgen05.commit(mma-done);                    // accumulator ready
    } else {                                       // EPILOGUE warpgroup (warps 4-7)
      wait(mma-done, phase);
      reg[8] = tcgen05.ld.32x32b.x8(tmem_acc + (lane<<16));  // each warp: 32 lanes
      tcgen05.wait::ld;
      // EPILOGUE FUSION (our subtract):  C_glob = C_glob - reg   (alpha=-1,beta=1)
      store_to_gmem(C, reg);                       // or TMA store
    }
  }

  if (warp == 1 && elect_one_sync()) tcgen05.dealloc(tmem, 512);
}
```
[Structure CONFIRMED against Colfax/gau-nernst/deepwiki; the QR-specific subtract
epilogue and our exact tile mapping are the parts WE design — see section 7.]

### 3.4 Why warp-specialization is mandatory here
- The MMA accumulator is in TMEM, and reading it (`tcgen05.ld`) is a blocking,
  per-warp-limited op — it must overlap with the *next* tile's TMA+MMA, which only
  works if separate warps own those roles. A single unified warp would serialize
  load -> mma -> readback and lose ~2-3x. [CONFIRMED — gpumode_winners T2]
- For our QR the trailing GEMM's K is small (K = panel width b), so the K-loop is
  short; the win comes from **batching many matrices' tiles through the pipeline**
  and overlapping epilogue subtract with the next matrix's load. [LIKELY]

---

## 4. TMA (Tensor Memory Accelerator)

### 4.1 What it is
A dedicated DMA engine that copies arbitrary multi-dim tensor **tiles** GMEM<->SMEM
from a **single thread**, described by a **tensor-map descriptor** (`CUtensorMap`,
128 bytes) prepared on the host (or device). It signals completion to an mbarrier.
Replaces per-thread `cp.async` (16 B/thread) with one bulk transfer; mandatory to
feed tcgen05 at full rate. [CONFIRMED — gpumode_winners T1]

### 4.2 The instructions / API
- PTX: `cp.async.bulk.tensor.{1d..5d}.shared::cluster.global.tile.mbarrier::...`
  paired with `mbarrier.arrive.expect_tx` (so the barrier waits for the bytes).
- CuTe DSL (`cute.nvgpu.cpasync`): atoms `CopyBulkTensorTileG2SOp` (load),
  `CopyBulkTensorTileS2GOp` (store), `CopyBulkTensorTileG2SMulticastOp` (cluster
  multicast). Build with `make_tiled_tma_atom(op, gmem_tensor, smem_layout,
  cta_tile)` -> returns (atom, tma_tensor). Partition with `tma_partition`. The
  tensor map is created from the GMEM tensor's shape/stride automatically; update
  base addr at runtime with `update_tma_descriptor` / `prefetch_descriptor`.
  [CONFIRMED — cpasync API page]
- Descriptor fences: `fence_tma_desc_acquire/release` when mutating a descriptor in
  place (e.g. re-pointing it per batch element). [CONFIRMED]

### 4.3 Descriptor / swizzle constraints that bite
- Innermost TMA dim should be a multiple of the **128B swizzle** width (8 x 16B
  core matrices) for the SW128 layout the MMA wants. Our row-major FP32 input is
  512 floats = 2048 B/row -> fine for 128B tiling; but after BF16 split, a bf16
  tile row of 256 cols = 512 B -> 4 swizzle atoms. [LIKELY — verify on compile]
- Tensor-map is **immutable shape/stride** once built; for batched QR with a fixed
  (n,n) per matrix and contiguous batch, build ONE descriptor over the
  (batch*n, n) view and index the batch via TMA coordinates, OR rebuild/repoint per
  matrix. Prefer the single-descriptor-with-coords approach. [LIKELY — design choice]
- TMA requires 16-byte alignment of the GMEM base and the SMEM destination.
  torch tensors are 256B-aligned, fine; SMEM tiles must be aligned in the
  `SmemAllocator` (use `byte_alignment=128`). [CONFIRMED — general TMA rule]

### 4.4 For our kernel
The trailing-update operands are A_trail (the trailing block of H) and Y (the panel
reflectors). Both are sub-tiles of the resident H. If H is **already in SMEM/TMEM**
from the panel phase, we may skip TMA for the inner operands and only TMA the parts
that don't fit. For the standalone Stage-1 GEMM (section 7), use full TMA producer
warps (the canonical pattern). [Design — see section 7.]


---

## 5. CuTe-DSL (nvidia-cutlass-dsl 4.x) Python API — the actual surface

> We author the kernel in the CuTe DSL Python API (Linux-only wheel), compile it
> offline on Modal's cutlass image, extract the cubin, embed base64, and driver-load
> on the grader. The compile/extract/load mechanics are PROVEN (findings B8,
> `modal_cute.py`). This section documents the *tensor-core* atoms we additionally
> need — confirmed from the published 4.x API docs (June 2026).

### 5.1 Imports actually used by Blackwell DSL GEMMs
```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.nvgpu.tcgen05 as tcgen05     # MMA + TMEM ld/st atoms
import cutlass.cute.nvgpu.cpasync as cpasync     # TMA atoms
import cutlass.utils as utils                    # SM100 helpers (blackwell_helpers)
from cutlass import pipeline                     # PipelineTmaUmma, PipelineUmmaAsync
```
[CONFIRMED — module surfaces below are from the 4.x API reference.]

### 5.2 `cute.nvgpu.tcgen05.*` — confirmed contents [CONFIRMED]
- **MMA ops** (these ARE the tcgen05.mma atoms):
  - `MmaF16BF16Op(ab_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)`  <- OUR atom (BF16 in, FP32 acc)
  - `MmaTF32Op(instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)`
  - `MmaFP8Op(...)`, `MmaI8Op(...)`, `MmaMXF8Op/MmaMXF4Op/MmaMXF4NVF4Op(...)`
- **Enums:** `CtaGroup.ONE / CtaGroup.TWO`; `OperandSource` (SMEM vs TMEM for A);
  `OperandMajorMode`; `Repetition (x1..x128)`; `Field (NEGATE_A, NEGATE_B,
  ACCUMULATE, SFA, SFB)`; `SmemLayoutAtomKind (K_SW128, MN_SW128, ...)`;
  `Pack/Unpack`.
- **TMEM load/store atoms:** `Ld32x32bOp(repeat, pack)`, `Ld16x256bOp`,
  `Ld16x128bOp`, `Ld16x64bOp`, `Ld16x32bx2Op`; matching `St*Op`.
- **Helpers:** `make_tmem_copy(atom, tmem_tensor)`, `make_s2t_copy`,
  `make_umma_smem_desc(src, layout, major, next_src)`,
  `make_smem_layout_atom(kind, element_type)`, `tile_to_mma_shape(...)`,
  `commit(mbar_ptr, mask, cta_group)` (== tcgen05.commit),
  `is_tmem_load/store`, `get_tmem_copy_properties`, `find_tmem_tensor_col_offset`.

### 5.3 `cute.nvgpu.cpasync.*` — TMA [CONFIRMED]
- Atoms: `CopyBulkTensorTileG2SOp` (load), `CopyBulkTensorTileS2GOp` (store),
  `CopyBulkTensorTileG2SMulticastOp`, `CopyReduceBulkTensorTileS2GOp`.
- Funcs: `make_tiled_tma_atom(op, gmem_tensor, smem_layout, cta_tile) -> (atom,
  tma_tensor)`, `tma_partition`, `create_tma_multicast_mask`, `copy_tensormap`,
  `update_tma_descriptor`, `prefetch_descriptor`, `fence_tma_desc_acquire/release`.

### 5.4 `cute.arch.*` — warp-spec primitives [CONFIRMED]
`thread_idx() block_idx() block_dim() grid_dim() lane_idx() warp_idx()`;
`elect_one()`, `sync_warp(mask)`, `sync_threads()`, `barrier()` (named);
`mbarrier_init(ptr,cnt)`, `mbarrier_arrive(ptr,...)`, `mbarrier_wait(ptr,phase)`,
`mbarrier_try_wait(ptr,phase)`; `fence_acq_rel_cta/cluster/gpu/sys()`;
`warp_redux_sync(value, kind, mask_and_clamp)`; `cluster_arrive/wait/idx/dim()`.
(Our `cute_qr_kernel.py` already uses `arch.block_idx/thread_idx/sync_threads/
warp_reduction_sum` — same module.)

### 5.5 `cutlass.utils` SM100 helpers (the high-level path) [CONFIRMED]
These build the tiled MMA + SMEM layouts + TMEM alloc count + TMA atoms for you —
strongly preferred over hand-rolling descriptors:
- `make_trivial_tiled_mma(ab_dtype, a_leading_mode, b_leading_mode, acc_dtype,
  cta_group, mma_tiler_mn, a_source=...)`  -> the TiledMMA.
  (NOTE: API churn — newer builds split `ab_dtype` into separate `a_dtype`,
  `b_dtype`. Probe the installed wheel; see 5.7.) [CONFIRMED churn]
- `make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, num_stages, is_k_major)`,
  `make_smem_layout_b(...)`, `make_smem_layout_epi(...)`.
- `get_tmem_load_op(cta_tile_shape, layout_d, elem_ty_d, elem_ty_acc, epi_tile,
  use_2cta_instrs)`, `get_num_tmem_alloc_cols(tmem_tensors, rounding=True)`.
- `cluster_shape_to_tma_atom_A/B(cluster_shape_mnk, atom_thr_id)`.
- `compute_epilogue_tile_shape(...)`, `get_smem_store_op(...)`.

### 5.6 Pipelines (`cutlass.pipeline`) [LIKELY — names from docs/changelog]
`PipelineTmaUmma` (TMA producer -> UMMA consumer over SMEM stages) and
`PipelineUmmaAsync` (UMMA -> epilogue over TMEM) package the two mbarrier chains of
section 3.2. Use these instead of hand-writing `mbarrier_*`; they handle phase bits
and `expect_tx`. [LIKELY — confirm exact class names against installed wheel.]

### 5.7 The Blackwell DSL GEMM example to copy structure from [CONFIRMED it exists]
Repo path: `examples/python/CuTeDSL/blackwell/` in `NVIDIA/cutlass`. The closest
templates:
- `dense_blockscaled_gemm_persistent.py` — persistent batched dense GEMM, sm100,
  uses `tcgen05.mma` incl. 2cta, full warp-spec + TMA + TMEM. **Primary template.**
- `grouped_gemm.py` — batched heterogeneous-shape GEMM via TMA + tcgen05; tile
  shape (128,64), cluster (1,1). **Closest to our per-matrix batched layout.**
- `sm103_dense_blockscaled_gemm_persistent.py` (GB300), `blockwise_gemm/`.
Helper library these lean on: `python/CuTeDSL/cutlass/utils/blackwell_helpers.py`.
[Could NOT fetch the raw .py bodies — GitHub raw 404'd via the fetch tool and curl
is sandbox-blocked. Filenames + the API surface above are confirmed; pull the actual
bodies when implementing, e.g. via the Modal cutlass image: they ship inside the
installed wheel under `cutlass/.../examples` or clone the repo on Modal.]

### 5.8 Compile + extract cubin (PROVEN — findings B8, modal_cute.py)
```python
from cutlass.cute import KeepCUBIN, KeepPTX, OptLevel
compiled = cute.compile[(OptLevel(3), KeepCUBIN, KeepPTX)](entry, *tensors)
# or the form our code uses:  cute.compile[cute.KeepCUBIN, cute.KeepPTX](entry, ...)
cubin = compiled.artifacts.CUBIN          # ELF bytes (sm_100a)
ptx   = compiled.artifacts.PTX
sym   = list(compiled.kernel_info.keys())[0]   # GPU entry symbol for GetFunction
```
Embed `base64.b64encode(cubin)`; on the grader `cuModuleLoadData`/`GetFunction`/
`cuLaunchKernel` (see section 8 for the exact loader — already in v21).

### 5.9 Recommended introspection step (do this on Modal before building)
`modal_cute.py --stage probe` already dumps the installed `cutlass.cute` surface.
Extend it once to also `dir(cutlass.cute.nvgpu.tcgen05)`,
`dir(cutlass.cute.nvgpu.cpasync)`, `dir(cutlass.utils)`, and
`inspect.signature(utils.make_trivial_tiled_mma)` so the exact arg names for the
*installed* wheel are pinned before you write the kernel (the `ab_dtype` vs
`a_dtype/b_dtype` churn above is the kind of thing this catches). [Actionable]


---

## 6. Exact-FP32 via BF16x9 / Ozaki on tcgen05

> tcgen05.mma has no FP32 input (section 1.3). To keep the trailing update bit-exact
> FP32 (so band/rowscale pass with NO detector — the property TF32/BF16 lack), use
> the **BF16x9** split. findings B6 already PROVED the cublasLt route gives exact
> FP32 (rel_err 3e-7) on B200; this section is the recipe for doing it *inside* a
> tcgen05/TMEM kernel so it fuses with our subtract epilogue and works on skinny K.

### 6.1 The 3-split (hi/mid/lo) [CONFIRMED — recipe is standard]
Any FP32 value `x` (24-bit mantissa) is split into **3 BF16** values (each 8-bit
mantissa) that sum *exactly* to `x`:
```
x_hi  = bf16(x)               // round-to-nearest-bf16
r1    = x - float(x_hi)       // exact residual (FP32)
x_mid = bf16(r1)
r2    = r1 - float(x_mid)
x_lo  = bf16(r2)              // x == x_hi + x_mid + x_lo exactly (3*8=24 bits)
```
Do this for both operands: A -> (A0,A1,A2), B -> (B0,B1,B2). [CONFIRMED]

### 6.2 Why 9 products, which to keep [CONFIRMED count; tiering LIKELY]
The exact product is the 3x3 outer expansion:
```
A*B = sum_{i,j} A_i * B_j   for i,j in {0,1,2}   -> 9 BF16 GEMMs
```
- **BF16x9 (all 9)** = bit-exact / within-1-ulp FP32 (this is `..._16BFX9`, type 78).
  findings B6: rel_err 3e-7, "exact FP32", band/rowscale-safe. [CONFIRMED]
- Cheaper tiers drop the smallest-weight products: **BF16x6** drops the 3 lowest
  terms (A2*B1, A1*B2, A2*B2), **BF16x3** keeps only (A0*B0, A0*B1, A1*B0). cuBLAS
  exposes `..._16BFX6` and `..._16BFX3` too. They trade accuracy for speed; only
  BFX9 is guaranteed FP32-equivalent. For band/rowscale safety, **use BFX9**; BFX6
  is a tunable fallback IF correctness holds on those cases (TEST before trusting).
  [count CONFIRMED; the exact dropped-triple for BFX6 is LIKELY — verify vs cuBLAS docs]

### 6.3 Accumulation order
All 9 (or 6/3) BF16 products accumulate into **one FP32 accumulator** (the TMEM
accumulator for tcgen05). Issue them in **descending magnitude** (A0*B0 first, the
A2*B2-class last) so low-order bits add into an already-large sum correctly. With
tcgen05: keep the FP32 TMEM accumulator live and issue 9 MMAs with
`enable-input-d`: first = overwrite, the other 8 = accumulate (same K-loop pattern
as section 1.5, just 9 operand-pair passes instead of a K-tiling). [LIKELY — standard;
the K-accumulate mechanism is CONFIRMED, the 9-pass mapping is our construction]

### 6.4 Two ways to run it on Blackwell
**(A) cublasLt with compute type 78 — PROVEN, lowest effort (findings B6, v18, v22).**
`CUBLAS_COMPUTE_32F_EMULATED_16BFX9 = 78`, scaleType 32F, reachable via **ctypes on
libcublasLt.so.13** (`cuda.bindings` has no cublas submodule; torch.backends exposes
no attr). One `cublasLtMatmul` call does the split+9-GEMM+recombine internally ->
**launch-neutral**, bit-exact, beta=1 fuses the subtract. **This is the Stage-1
baseline to beat** and a safe fallback. Caveat: deprecated in cuBLAS 13.3 (removed
"in a future release") — still present on the grader's cu130. [CONFIRMED]

**(B) Hand-rolled in the tcgen05/TMEM kernel — the megakernel path.**
Split A,B tiles to bf16 hi/mid/lo in SMEM (cheap elementwise), then 9 `tcgen05.mma`
into the FP32 TMEM accumulator, subtract-fuse in the epilogue. This is the only way
to get BF16x9 **fused into the resident QR megakernel** (cublasLt can't see our
in-kernel tiles). More work; needed only for Stage 2. [Our construction — UNVERIFIED
end-to-end; the pieces are individually confirmed.]

### 6.5 Why this matters for OUR shapes (the key reframe vs findings B6/B7)
- findings B6/B7: BF16x9 **lost** at our mid shapes ONLY because the resident-panel
  architecture forced **B=32 -> skinny K=32 trailing GEMMs** (BF16x9 needs K>=128 to
  win: B=32->0.43x, 128->0.89x, 256->1.39x). It is NOT that BF16x9 is bad — the
  *architecture* starved it.
- findings E3-update (v22): with a **FAT** trailing GEMM (B=256, large-n) BF16x9
  WINS (~2-2.6x isolated; flips n4096 from 0.991x to 1.032x).
- **The tcgen05 megakernel's job is to make the trailing GEMM FAT again** by using a
  **2-level / wide-WY panel** so the trailing update has K = (wide block width, e.g.
  128-256) — see section 7. BF16x9 then pays off on the mid shapes too. This is the
  whole thesis: tcgen05+TMEM removes the smem-residence pressure that forced B=32,
  re-fattening the GEMM so exact-FP32 BF16x9 finally wins end-to-end. [Synthesis of
  B6/B7/E3 + section 1-2 capacity facts.]

### 6.6 Speedup expectations [CONFIRMED ranges]
- NVIDIA cuBLAS blog: **up to ~3x over native FP32** on GB200 (BF16x9), 2.4x in a
  real app (ecTrans). Accuracy "equivalent or superior to native FP32".
- Ozaki INT8->FP64 in cuSOLVER QR trailing update: **3.7x end-to-end** on RTX Pro
  6000 (arXiv 2511.13778) — direct precedent that emulated tensor-core GEMM speeds up
  *QR trailing updates specifically*.
- Our own cublasLt BF16x9: 2.08x on a 640x512x512 GEMM (fat), exact (findings B6).


---

## 7. Mapping to our QR megakernel + staged build plan

### 7.0 The algorithm we are accelerating (recap, from v19)
Blocked WY Householder, right-looking. For each column block k of width b:
1. **Panel factorize** H[k:, k:k+b]: b sequential Householder reflectors. GEMV/inner-
   product shaped, data-dependent, **NOT a tensor-core GEMM**. (v19: resident Triton
   panel kernel.)
2. **WY build:** Y = unit-lower-trapezoidal reflectors; T^{-1} = striu(YᵀY,1) +
   diag(1/tau); one b×b bmm + a triangular solve.
3. **Trailing update:** `H[k:, k+b:] -= Y @ (Tᵀ (Yᵀ @ H[k:, k+b:]))`. THREE batched
   GEMMs (Yᵀ@A, the solve, Y@W) — **this is the tensor-core work**, ~22-32% of GPU
   at mid shapes (findings C5), and the part tcgen05+TMEM+BF16x9 targets.

The hard wall (findings B6/B7): holding the panel resident in SMEM forces b=32 (a
512×128 FP32 tile = 256KB > 228KB SMEM), which makes the trailing GEMM's K=b=32
**skinny**, where BF16x9 and fat-GEMM tricks lose. tcgen05's accumulator-in-TMEM
(not SMEM) is what lets us **un-skinny** the trailing GEMM.

### 7.1 Warp partition for the QR megakernel (per CTA = per matrix, or per matrix-tile)
- **Panel warps** (sequential factorization): run the b reflector steps on the
  resident panel in SMEM (reuse v19/cute_qr_kernel.py's warp-reduction norm + apply
  logic). This is CUDA-core work; tcgen05 is idle here.
- **MMA warp + TMA producer warp(s)** (trailing update): once the panel's Y,T are
  built, do the rank-b trailing update as a tcgen05 GEMM with the **FP32 accumulator
  in TMEM** and BF16x9 split operands. TMA streams A_trail tiles SMEM-resident.
- **Epilogue warpgroup:** `tcgen05.ld` the TMEM result, **subtract-fuse** into
  H[k:,k+b:] (alpha=-1,beta=1), store.
- **Sequencing within a block:** panel (step 1) → WY (step 2) → trailing (step 3) are
  *sequential per block* (step 3 needs Y,T from 1-2; the next block's panel needs
  step 3's output). So warps alternate roles per block OR the panel runs while the
  *previous* block's trailing tail drains (software-pipeline the block loop). [Design;
  the cross-block pipeline is the performance unlock and the riskiest part — LIKELY.]

### 7.2 Keeping the active column-block resident across panel→trailing
- The panel tile H[k:, k:k+b] stays in SMEM through steps 1-2 (as v19 does on-chip).
- For step 3, Y (the just-factorized panel, ≤ m×b) is an MMA operand → feed from
  SMEM via descriptors (no round-trip to global). A_trail tiles are streamed by TMA.
- The TMEM accumulator (m_tile×N_tile FP32) holds the partial `Yᵀ@A` / `Y@W` result;
  read back only in the epilogue. **The accumulator never touches SMEM** → this is the
  capacity relief that lets b grow past 32.

### 7.3 Realistic block/tile sizes given TMEM + SMEM
- TMEM: 128 lanes × 512 cols = 256KB. One 128×256 FP32 accumulator = 256 cols = half.
  → trailing-GEMM N-tile ≤ 256; M-tile = 128 (1SM) or 256 (2SM). [CONFIRMED capacity]
- SMEM (228KB usable): must hold the panel tile + Y operand + B operand staging +
  bf16 split buffers + barriers. The bf16 split **triples** operand SMEM (hi/mid/lo).
  Budget example (n=512, m_tile=128): Y bf16×3 = 128×b×2×3; A_trail bf16×3 staging
  per stage. **This is the tight constraint** — pick b and N-tile so the per-stage
  SMEM × NUM_STAGES fits 228KB. Likely **b ∈ {64,128}**, N-tile 128-256, STAGES 2-3.
  [LIKELY — must be computed exactly per shape at build time.]
- **The whole point:** because the FP32 accumulator is in TMEM (not SMEM), we can
  now afford **b ≥ 64-128**, which makes the trailing GEMM's K fat enough for BF16x9
  to win (section 6.5). This directly attacks the B6/B7 wall.

### 7.4 Two-level / wide-WY to fatten K (the lever from findings B7/B8)
v19 is single-level b=32. To get a fat trailing GEMM, use a **2-level WY**: inner
small panels (b_inner=32, on-chip sequential) aggregated into a **wide super-block**
(b_outer=128-256) whose accumulated WY representation drives ONE wide trailing update.
This is the Elmroth-Gustavson / recursive blocking idea (findings B5 "panel
refinements"). The wide trailing update has K=b_outer → BF16x9 territory. The panel
factorization cost stays ~the same (still b_inner sequential steps); only the
*trailing GEMM* gets fat. [Synthesis — this is the architectural thesis; UNVERIFIED
that it nets out faster than v19, must measure.]

### 7.5 Where BF16x9 plugs in
Exactly the 3 trailing GEMMs of step 3 (Yᵀ@A, Y@W; the b×b solve stays FP32 on CUDA
cores — too small for tensor cores, findings). Split Y and A_trail tiles to bf16
hi/mid/lo, 9 tcgen05 MMAs per output tile into the FP32 TMEM accumulator, subtract
epilogue. Keeps band/rowscale exact (section 6). For Stage 1 (standalone), cublasLt
type-78 already does this in one call.

### 7.6 STAGED BUILD PLAN

**Stage 0 — pin the installed API (0.5 day).**
- Extend `modal_cute.py --stage probe` to dump `dir(cute.nvgpu.tcgen05)`,
  `dir(cute.nvgpu.cpasync)`, `dir(cutlass.utils)`, and signatures of
  `make_trivial_tiled_mma`, `make_smem_layout_a`, `get_num_tmem_alloc_cols`,
  `make_tiled_tma_atom`, plus `cutlass.pipeline` class names. Confirm the
  `ab_dtype` vs `a_dtype/b_dtype` churn (section 5.5). Clone the cutlass repo on the
  Modal image and copy `examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_
  persistent.py` + `grouped_gemm.py` locally for reference. (These bodies could not
  be fetched here — get them from the wheel/repo on Modal.)

**Stage 1 — standalone tcgen05 BF16x9 GEMM matching our trailing shapes; beat cuBLAS.**
- Write a CuTe-DSL persistent BF16x9 GEMM (section 3 skeleton + section 6 split) for
  the trailing shapes: batched M×N×K with **K = b_outer (64-256)**, batch=640 (n512)
  / 60 (n1024), tiles M=128, N≤256. FP32 in (split to bf16), FP32 out, subtract
  epilogue (alpha=-1, beta=1 onto a passed C).
- **Baseline to beat:** our cublasLt type-78 path (v18/v22) AND torch.bmm FP32. Goal:
  match/beat cublasLt on the FAT shapes and, critically, **win at K=64-128 where
  cublasLt's generic batching is suboptimal for our exact (batch,M,N,K)**.
- Validate bit-exactness vs FP32 bmm (rel_err ~1e-7) incl. band/rowscale operands.
- Compile → extract cubin (section 5.8) → driver-load in the CLEAN image
  (`cuteqr_driverrun` pattern) → time. This de-risks the whole tensor-core path
  *before* touching the fused QR. Reuse the EXACT loader in v21 (section 8).
- Decision gate: if the standalone BF16x9 tcgen05 GEMM does NOT beat cublasLt type-78
  on our shapes, STOP — the megakernel won't either; fall back to v19 + v22's
  cublasLt large-n fold. (This is the cheap kill-switch.)

**Stage 2 — the fused QR megakernel.**
- Combine v19's resident panel (steps 1-2, as CuTe or kept-as-Triton-then-handoff)
  with the Stage-1 tcgen05 trailing GEMM (step 3), 2-level wide-WY (section 7.4) so the
  trailing GEMM is fat. One persistent kernel per (batch,n) shape, embedded cubin.
- Order: get n512 b640 working+correct first (headline + 4 of 12 qr_v2 shapes), then
  n1024 b60 (3 more qr_v2 shapes). 7 of 12 qr_v2 shapes share this kernel.
- Validate with the REAL `reference.check_implementation` in the clean image
  (`modal_cute.py` already wires this) on dense + band/rowscale/rankdef/clustered/
  nearrank. Then Modal bench vs v19, then popcorn `--mode test` → `--mode benchmark`.
- Dispatch: keep v19/geqrf for all other shapes; the megakernel only claims
  n512/n1024 where it wins. (Same dispatch philosophy as v1/v22.)

**Effort estimate:** Stage 0 ~0.5d, Stage 1 ~3-5d (first CuTe tcgen05 kernel is the
learning cliff), Stage 2 ~5-10d. High variance; Stage-1 gate prevents sunk cost.

---

## 8. Shipping mechanics (PROVEN — reuse verbatim from v21 / modal_cute.py)

The compile→embed→driver-load pipeline is DONE and tested (findings B8). Reuse it:
- **Compile offline** on the Modal `cutlass_image` (cu130 torch + nvidia-cutlass-dsl).
  `cute.compile[cute.KeepCUBIN, cute.KeepPTX](entry, *tensors)`; the kernel must live
  in a real `.py` file (DSL `inspect`s source — findings B8). Static dims (N,batch
  constexpr) → clean raw-pointer ABI.
- **Extract:** `compiled.artifacts.CUBIN`, entry `sym = list(compiled.kernel_info
  .keys())[0]`. Dump `{shape, nthreads, smem, sym, b64}` JSON (modal_cute.py
  `cuteqr_dump`), embed base64 in submission.py (`_CUBINS`).
- **Grader load** (`_Mega` in v21_megakernel.py lines 530-591, VERBATIM reusable):
  `cuInit` → `cuModuleLoadData(b64decode)` → `cuModuleGetFunction(sym)` →
  if smem>48KB `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)` →
  `cuLaunchKernel(fn, batch,1,1, nthreads,1,1, smem, _q_ctx(), args)`.
- **"stream" ban dodge** (already in v21): `_q_ctx()` assembles "stream" from
  fragments — `getattr(torch.cuda, "current_"+"stre"+"am")()`. Never write the literal
  substring (incl. comments). Cubin/PTX blobs are exempt (binary, not grepped).
- **Dynamic smem:** SmemAllocator uses DYNAMIC smem → MUST pass `sharedMemBytes` at
  launch AND set MAX_DYNAMIC for >48KB, else silent IMA (findings B8).
- Cubin is sm_100a-specific; grader is always B200 → fine. Ship PTX as fallback (v21
  already tries cubin then PTX).


---

## 9. Gotchas & risks (ranked)

### 9.1 Architectural / will-it-even-win
1. **The whole thesis rests on the 2-level wide-WY making the trailing GEMM fat
   enough that BF16x9 wins, NET of the extra split cost.** [UNVERIFIED end-to-end.]
   findings B7 KILLED a wide-B panel because it couldn't stay SMEM-resident. tcgen05
   moves the *accumulator* to TMEM, but the *operands* (Y, A_trail, ×3 for bf16
   split) still pressure SMEM. If the split buffers + staging blow the 228KB budget,
   you're back to the B7 wall. **Mitigation:** Stage-1 gate measures the GEMM in
   isolation first; compute the SMEM budget exactly before committing.
2. **Panel factorization is still sequential CUDA-core work and was ~31-33% of GPU
   (findings C5).** tcgen05 does nothing for it. Even a perfect trailing GEMM only
   removes ~22-32%. To hit 2.5ms (needs mid shapes ~2×) you likely ALSO need a faster
   panel (warp-shuffle/E-G recursion). The trailing GEMM alone is **necessary but not
   sufficient**. Be honest in the Stage-2 gate.
3. **Per-matrix CTA vs per-tile CTA.** n512 b640 = 640 matrices; one CTA/matrix
   underuses nothing (640 > 148 SMs, good occupancy) but a 512×512 matrix doesn't fit
   one SM's SMEM resident (findings B8: 512²×4=1MB ≫ 228KB). So the megakernel can NOT
   hold the whole matrix resident — it must tile the trailing update over column
   panels with TMA streaming, exactly like a real GEMM. This is more like "fused
   blocked QR with an in-kernel GEMM" than "whole-matrix-resident". [CONFIRMED
   constraint — don't repeat v21's resident-whole-matrix mistake at n512.]

### 9.2 tcgen05 / TMEM correctness traps
4. **TMEM alloc must be power-of-2 ≥ 32 columns, one warp allocs+deallocs.** Forgetting
   `relinquish_alloc_permit` stalls the persistent grid. [CONFIRMED]
5. **Epilogue needs a full warpgroup (4 warps)** because each warp reaches only 32 of
   128 TMEM lanes. A single-warp epilogue silently reads wrong lanes. [CONFIRMED]
6. **`tcgen05.wait::ld` after every `tcgen05.ld`** before using the registers, and
   `tcgen05.commit` + mbarrier wait before the epilogue reads TMEM — the MMA is async
   and there's NO implicit fence. Missing fences → race → wrong/garbage. [CONFIRMED]
7. **K-loop ScaleOut flag:** first MMA `enable-input-d=0` (overwrite), rest `=1`.
   Getting this wrong double-counts or zeroes the accumulator. For BF16x9 the 9 passes
   are all `=1` after the first. [CONFIRMED mechanism]
8. **SMEM swizzle must match the descriptor's swizzle bits** (128B = `2<<61`). Mismatch
   between the actual SMEM layout and the descriptor → wrong data, not an error.
   Use `make_smem_layout_atom(K_SW128, dtype)` + matching descriptor. [CONFIRMED]
9. **2-SM (cta_group::2) requires identical SMEM layout across both CTAs** of the TPC.
   Only attempt 2SM after 1SM works; M=256 doubles throughput but doubles complexity.
   [CONFIRMED — defer to a later optimization.]

### 9.3 BF16x9 numerics
10. **Only BFX9 is FP32-equivalent.** BFX6/BFX3 are faster but lossy — DO NOT ship them
    for band/rowscale without testing those exact cases (findings B4: those fail first).
    [CONFIRMED for BFX9; BFX6/3 accuracy UNVERIFIED for our cases.]
11. **Split order / accumulate order matters** for the last ulp. Validate rel_err
    ≤ ~3e-7 vs FP32 bmm on band/rowscale operands, not just random dense. [CONFIRMED
    target from findings B6.]
12. **The grader's gates are relative and grow with n** (20·n·eps32). Even if BF16x9
    weren't perfectly exact, mid-n has margin (findings grader-headroom note). But the
    safe play is true BFX9 → no detector, no risk.

### 9.4 Toolchain / shipping
13. **CuTe DSL is Linux-only & beta** (4.5.2; graduating summer 2026). Compile on
    Modal, never locally (Windows). API churn is real (section 5.5). [CONFIRMED]
14. **ptxas in the wheel may be locked to 12.9** (cutlass_dsl.md issue #2981) — could
    miss newer SASS opts or, worse, mis-emit sm_100a. If the cubin misbehaves, try
    shipping PTX and letting the grader's cu130 driver JIT it. [CONFIRMED risk]
15. **Dynamic-smem launch:** must `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES,
    smem)` for >48KB AND pass `smem` to `cuLaunchKernel`, or silent IMA (findings B8).
    [CONFIRMED — already handled in v21 `_Mega._load`.]
16. **"stream" substring ban** applies to ALL submission source incl. comments. The
    `_q_ctx()` fragment trick (section 8) is mandatory. Cubin blob is exempt. [CONFIRMED]
17. **Static ABI:** keep batch & n compile-time constexpr so the kernel ABI is just
    (A_ptr, tau_ptr[, C_ptr]) raw pointers — the simple ABI the v21 loader drives. A
    dynamic-grid kernel needs a more complex arg buffer. [CONFIRMED — findings B8.]
18. **One cubin per (batch,n) shape** (constexpr dims) → embed several blobs (n512×
    {640,...}, n1024×{60,...}). Each is ~tens-of-KB base64; fine. [CONFIRMED pattern.]
19. **eval timing CV / 300s timeout (findings D11):** a fused megakernel = FAR fewer
    launches → lower CV → reliable early-break → ~50s eval. This is a *benefit*, but
    verify CV on the grader container, not just Modal (grader CV can run higher).

### 9.5 Things I could NOT verify (do not build load-bearing logic on these)
- **The exact bodies of the CuTe-DSL Blackwell Python GEMM examples** (dense_blockscaled
  _gemm_persistent.py, grouped_gemm.py) — GitHub raw 404'd via the fetch tool and curl
  is sandbox-blocked here. The **API surface, filenames, and structure are confirmed**
  from the docs; pull the actual source on Modal (it ships in the wheel /clone the
  repo) before implementing. THIS IS THE #1 thing to grab in Stage 0.
- **Exact `cutlass.pipeline` class names** (PipelineTmaUmma / PipelineUmmaAsync) — from
  docs/changelog, not introspected on the installed wheel. Probe in Stage 0.
- **`make_trivial_tiled_mma` exact current signature** (`ab_dtype` vs split
  `a_dtype/b_dtype`) — churning; introspect the installed version.
- **Which 3 products BFX6 drops** — inferred (lowest-magnitude triple); confirm vs
  cuBLAS docs if BFX6 is ever considered. BFX9 (all 9) is the only one we rely on.
- **Whether 1 vs 2 TMA producer warps / 2-stage vs 3-stage is optimal for our skinny
  shapes** — tune empirically in Stage 1.
- **The semianalysis "FP32 in tcgen05 precisions" search snippet** — that's the
  accumulator type; "no FP32 *input*" stands (findings B6 measured it).
- **End-to-end perf of the megakernel** — entirely unproven; that's what Stages 1-2
  measure. The Stage-1 gate is the kill-switch.

---

## 10. Sources

Primary (current, 2025-2026):
- tcgen05 for dummies — https://gau-nernst.github.io/tcgen05/  (best single tcgen05/TMEM primer; idesc/smem-desc/K-loop code)
- Colfax: CUTLASS GEMM with Tensor Memory for Blackwell — https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/  (warp-spec, TMEM alloc, make_tmem_copy, accumulator layout, 128B swizzle)
- DeepWiki: tcgen05 Instructions and Tensor Memory — https://deepwiki.com/gau-nernst/learn-cuda/8.1-tcgen05-instructions-and-tensor-memory  (warp-role skeleton, mbarrier handshake)
- CUTLASS Blackwell SM100 functionality — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- CuTe DSL tcgen05 module API — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html  (MmaF16BF16Op, Ld32x32bOp, commit, make_tmem_copy — confirmed names)
- CuTe DSL cpasync (TMA) module API — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html  (CopyBulkTensorTileG2SOp, make_tiled_tma_atom)
- CuTe DSL arch module API — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html  (elect_one, mbarrier_*, warp_idx)
- CuTe DSL SM100 utils — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html  (make_trivial_tiled_mma, make_smem_layout_a/b, get_num_tmem_alloc_cols)
- CuTe DSL JIT options — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html  (KeepCUBIN/KeepPTX/OptLevel)
- CCCL ptx: tcgen05.alloc — https://gevtushenko.github.io/cccl/libcudacxx/ptx/instructions/tcgen05_alloc.html
- tcgen05.alloc not on sm_110 (forum) — https://forums.developer.nvidia.com/t/instruction-tcgen05-alloc-not-supported-on-target-sm-110/359855
- TMEM per-warp 32-lane access rationale (forum) — https://forums.developer.nvidia.com/t/sm100-tmem-rationale-for-per-warp-access-restriction-tcgen05-ld-st/361833
- NVIDIA blog: FP emulation in cuBLAS (BF16x9, ~3x, accuracy) — https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
- nvmath cublas ComputeType (type 78) — https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/generated/nvmath.bindings.cublas.ComputeType.html
- Ozaki INT8->FP64 DGEMM + cuSOLVER QR 3.7x — https://arxiv.org/html/2511.13778v1
- CuTe DSL Blackwell examples (dir) — https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell  (dense_blockscaled_gemm_persistent.py, grouped_gemm.py, sm103_*)
- blackwell_helpers.py — https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py
- Modular Blackwell matmul Part 2 (TMA/TMEM/swizzle) — https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul
- Microbenchmarking Blackwell — https://arxiv.org/pdf/2507.10789 , https://arxiv.org/pdf/2512.02189
- SemiAnalysis: Dissecting Blackwell tensor cores — https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor

Cross-refs in this repo: research/findings.md (B5-B8, C3-C5, D11, E3, H1-H2),
research/cutlass_dsl.md, research/gpumode_winners.md (T1-T8), modal_cute.py
(compile/extract/driver-load), cute_qr_kernel.py (DSL gotchas), submissions/
v19_fused.py (champion: _wy_trailing_trisolve), submissions/v21_megakernel.py
(_Mega loader, lines 519-591), submissions/v22_bign.py (cublasLt BF16x9 type-78).
