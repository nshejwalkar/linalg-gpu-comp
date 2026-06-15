# CuTe-DSL Blackwell Dense GEMM — Concrete Patterns Reference

> **Purpose.** Copy-pasteable Python DSL idioms for a B200 tcgen05/TMEM persistent GEMM kernel.
> Synthesized from primary sources June 2026. Status tags: [CONFIRMED] = from primary source;
> [LIKELY] = strong inference; [UNVERIFIED] = not bit-verified. Read together with
> `tcgen05_tmem.md` (the engineering spec) and `tcgen05_stage0_api.md` (probed API).
>
> **Key breakthrough from the opus agent (2026-06-15):** the canonical recipe uses
> `make_fragment_A/B(sA/sB)` directly on the SMEM tensor — NOT `partition_A/B`. The
> `recast_ptr` trick moves the swizzle into the pointer so `cute.gemm` sees an affine layout.
> See §1 for the end-to-end recipe.

---

## 0. Imports (canonical)

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils  # NOT cutlass.utils.blackwell

from cutlass.cute.nvgpu.tcgen05 import (
    MmaF16BF16Op, CtaGroup, OperandSource, OperandMajorMode,
    Field, SmemLayoutAtomKind, Repetition, Pack,
    Ld32x32bOp, Ld16x256bOp, Ld16x128bOp, Ld16x64bOp, Ld16x32bx2Op,
    St32x32bOp,
    make_tmem_copy, make_smem_layout_atom, make_umma_smem_desc, commit,
)
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SOp, CopyBulkTensorTileS2GOp,
    make_tiled_tma_atom, tma_partition, prefetch_descriptor,
    update_tma_descriptor, fence_tma_desc_acquire, fence_tma_desc_release,
)
from cutlass.pipeline import (
    PipelineTmaUmma, PipelineUmmaAsync, PipelineTmaAsync,
    PipelineState, NamedBarrier, CooperativeGroup,
)
from cutlass.cute import (
    KeepCUBIN, KeepPTX, OptLevel,
)
import cutlass.cute.arch as arch
```

[CONFIRMED — from `tcgen05_stage0_api.md`, probed on installed wheel 4.5.2]

---

## 1. Canonical CuTe-DSL Blackwell Dense GEMM Recipe (end-to-end)

This is the recipe that COMPILES (confirmed by opus agent `opus_probe_recast`).
The critical insight: `make_fragment_A/B` takes the raw SMEM tensor; `recast_ptr`
moves the swizzle into the pointer before `cute.gemm`.

### 1.1 Step-by-step with exact API names

```python
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.utils as utils
from cutlass.cute.nvgpu.tcgen05 import (
    MmaF16BF16Op, CtaGroup, OperandMajorMode, OperandSource, Field,
)
from cutlass import BFloat16, Float32

# ── (A) Create the TiledMMA ─────────────────────────────────────────
# NEW API (4.5.2 wheel): a_dtype/b_dtype separately.  ab_dtype is DEPRECATED.
tiled_mma = utils.make_trivial_tiled_mma(
    a_dtype=BFloat16,
    b_dtype=BFloat16,
    a_leading_mode=OperandMajorMode.K,   # K-major = row-major A
    b_leading_mode=OperandMajorMode.K,   # K-major = col-major B (transposed)
    acc_dtype=Float32,
    cta_group=CtaGroup.ONE,              # single-SM; use TWO for 2-SM
    mma_tiler_mn=(128, 256),             # (M, N) tile for MMA
    # a_source defaults to SMEM (smem_desc); use OperandSource.TMEM for _TS atoms
)

# ── (B) Allocate staged SMEM tensors with swizzled layouts ──────────
NUM_STAGES = 2
mma_tiler_mnk = (128, 256, 64)          # (M, N, K) — K must match your tile
sA_layout = utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, BFloat16, NUM_STAGES)
sB_layout = utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, BFloat16, NUM_STAGES)
# sA_layout is rank-4: ((M, K), swizzle_atom, 1, stage)
# Allocate SMEM via SmemAllocator (or @cute.jit smem_size param):
smem_A = cute.make_tensor(smem_ptr_A, sA_layout)   # composed (swizzled) layout
smem_B = cute.make_tensor(smem_ptr_B, sB_layout)

# ── (C) The recast_ptr trick: move swizzle → pointer ────────────────
# Required before make_fragment_A/B so cute.gemm sees an affine layout.
# sA_layout has .inner (swizzle part) and .outer (affine part).
pA = cute.recast_ptr(smem_A.iterator, sA_layout.inner, BFloat16)
sA_affine = cute.make_tensor(pA, sA_layout.outer)  # now affine, swizzle in ptr

pB = cute.recast_ptr(smem_B.iterator, sB_layout.inner, BFloat16)
sB_affine = cute.make_tensor(pB, sB_layout.outer)

# ── (D) Create SMEM-descriptor operand fragments ────────────────────
# Slice to stage 0 (last dim of rank-4 layout):
sA_stage0 = sA_affine[None, None, None, 0]
sB_stage0 = sB_affine[None, None, None, 0]

tCrA = tiled_mma.make_fragment_A(sA_stage0)   # returns smem_desc fragment
tCrB = tiled_mma.make_fragment_B(sB_stage0)   # shape: (MMA, MMA_K, 1, 1) ish

# ── (E) Allocate TMEM and create accumulator ────────────────────────
# TMEM alloc — must be called by a single warp (not just a single thread)
num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)  # or supply manually
# In-kernel: arch.alloc_tmem writes the address into an SMEM slot
arch.alloc_tmem(num_tmem_cols, smem_tmem_ptr_slot, is_two_cta=None, arch='sm_100')
arch.relinquish_tmem_alloc_permit()    # release alloc lock for other CTAs
arch.sync_threads()
tmem_ptr = arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=smem_tmem_ptr_slot)

# Build fake layout to get correct shape, then repoint to real TMEM ptr:
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])   # (M, N) shaped
tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)            # TMEM tensor (layout only)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)       # real TMEM accumulator
# tCtAcc layout: ptr<f32,tmem> o ((128,256),1,1):((65536,1),0,0)
# — stride 65536 (= 1<<16) between rows = TMEM lane encoding

# ── (F) K-loop: issue MMA instructions ──────────────────────────────
# First K-block: overwrite accumulator (ACCUMULATE=False)
tiled_mma.set(Field.ACCUMULATE, False)
for k_block in range(num_k_blocks):
    stage = k_block % NUM_STAGES
    sA_this = sA_affine[None, None, None, stage]
    sB_this = sB_affine[None, None, None, stage]
    tCrA_this = tiled_mma.make_fragment_A(sA_this)
    tCrB_this = tiled_mma.make_fragment_B(sB_this)

    # Only one warp, one thread issues the MMA (warp=MMA_WARP, elect_one)
    with arch.elect_one():
        cute.gemm(tiled_mma, tCtAcc, tCrA_this, tCrB_this, tCtAcc)
    tiled_mma.set(Field.ACCUMULATE, True)   # accumulate from 2nd k-block on

# ── (G) Signal MMA done → epilogue ──────────────────────────────────
# tcgen05.commit to mbarrier (CuTe DSL wrapper):
tcgen05.commit(mma_done_mbar_ptr, cta_group=CtaGroup.ONE)

# ── (H) Epilogue: TMEM → registers (full warpgroup = 4 warps) ───────
# Wait for MMA completion on the mma_done mbarrier, then:
tiled_t2r = make_tmem_copy(Ld32x32bOp(Repetition.x8), tCtAcc)
thr_t2r = tiled_t2r.get_slice(arch.thread_idx())   # thread_idx across 4 warps
tDtAcc = thr_t2r.partition_S(tCtAcc)
tDrAcc = thr_t2r.make_fragment(tDtAcc)
cute.copy(tiled_t2r, tDtAcc, tDrAcc)
arch.fence_view_async_tmem_load()   # MANDATORY: wait before using registers

# ── (I) Dealloc TMEM at kernel exit ─────────────────────────────────
arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=None, arch='sm_100')
```

[CONFIRMED via opus_probe_recast + dense_gemm.py canonical recipe. Exact slice syntax
`sA[None,None,None,0]` for rank-4 confirmed in `opus_progress.md`.]

### 1.2 What make_smem_layout_a/b actually returns

The output is a **rank-4 ComposedLayout**: shape `(M_smem, K_smem, 1, NUM_STAGES)` with
a swizzle composed onto the inner dimensions. Concretely for BF16, M=128, K=64, stages=2:
- Inner (swizzle): `S<1,4,3>` (128B atom, matches the MMA descriptor's swizzle=128B)
- Outer (affine): `((M, K), 1, 1, stage)`
- Stride between stages = total bytes of one A stage / sizeof(BFloat16)

When you do `recast_ptr(sA.iterator, sA_layout.inner, BFloat16)`, the swizzle is baked
into the pointer arithmetic, and `sA_layout.outer` is the plain affine layout passed to
`cute.gemm`. This is why `make_fragment_A(sA_affine_stage0)` works where `make_fragment_A(sA)`
(the composed layout) fails with "Expected affine layout". [CONFIRMED — opus_probe_recast]

### 1.3 make_trivial_tiled_mma — confirmed signature (4.5.2)

```python
# NEW (current) signature:
utils.make_trivial_tiled_mma(
    a_dtype=BFloat16,
    b_dtype=BFloat16,
    a_leading_mode=OperandMajorMode.K,
    b_leading_mode=OperandMajorMode.K,
    acc_dtype=Float32,
    cta_group=CtaGroup.ONE,
    mma_tiler_mn=(128, 256),
    a_source=OperandSource.SMEM,   # optional, default is SMEM
)

# DEPRECATED (old) signature still accepted:
utils.make_trivial_tiled_mma(
    ab_dtype=BFloat16,             # same type for both A and B
    a_leading_mode=OperandMajorMode.K,
    b_leading_mode=OperandMajorMode.K,
    acc_dtype=Float32,
    cta_group=CtaGroup.ONE,
    mma_tiler_mn=(128, 256),
)
```

[CONFIRMED — blackwell_helpers.py source fetch + stage0 probe. The wheel dispatches on
argument names; use the new API.]

### 1.4 cute.gemm call signature

```python
cute.gemm(tiled_mma, acc, fragA, fragB, acc)
# ↑ D=C form: D = A*B + C; pass same tensor for acc and C to accumulate in-place.
# fragA = tiled_mma.make_fragment_A(sA_affine_stage0)   — SMEM descriptor fragment
# fragB = tiled_mma.make_fragment_B(sB_affine_stage0)   — SMEM descriptor fragment
# acc   = tCtAcc  (TMEM accumulator tensor)
```

[CONFIRMED — opus_progress.md canonical recipe section]

---

## 2. tcgen05 MMA + TMEM — PTX-level reference

### 2.1 Instruction syntax (PTX)

```ptx
# 1-SM MMA (BF16 in, FP32 accumulator)
tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, enable-input-d;

# 2-SM MMA
tcgen05.mma.cta_group::2.kind::f16 [d-tmem], a-desc, b-desc, idesc, enable-input-d;

# enable-input-d = predicate register
#   False/0 → D = A*B         (overwrite, first K-block)
#   True/1  → D = A*B + D     (accumulate, subsequent K-blocks)
```

[CONFIRMED — gau-nernst tcgen05 tutorial]

### 2.2 Instruction descriptor (idesc, 32-bit)

```c
constexpr uint32_t idesc =
    (1U << 4U)                         // d (accumulator) dtype = FP32
  | (1U << 7U)                         // a operand type = BF16
  | (1U << 10U)                        // b operand type = BF16
  | ((uint32_t)BLOCK_N >> 3U << 17U)   // MMA_N encoding
  | ((uint32_t)BLOCK_M >> 4U << 24U);  // MMA_M encoding
```

For TF32 inputs: change bits [7:9] and [10:12] accordingly. In CuTe DSL these are
constructed automatically from the MMA atom type. [CONFIRMED — gau-nernst]

### 2.3 SMEM matrix descriptor (64-bit, one for A, one for B)

The descriptor encodes the SMEM base address, leading-dim byte offset (LBO), and
stride-dim byte offset (SBO), plus swizzle mode in bits [61:63]:

```c
// K-major layout, 128B swizzle (swizzle=2 = 128B):
const int SBO = 8 * 128;   // 8 core matrices × 128 bytes each
uint64_t desc = desc_encode(smem_base_addr)
              | (desc_encode(SBO) << 32ULL)
              | (1ULL << 46ULL)         // bit 46 = 1 (required)
              | (2ULL << 61ULL);        // bits [61:63] = 2 → 128B swizzle

// No swizzle version (K-major, no swizzle):
const int LBO = height * 16;
const int SBO_ns = 8 * 16;
uint64_t desc = desc_encode(smem_base_addr)
              | (desc_encode(LBO) << 16ULL)
              | (desc_encode(SBO_ns) << 32ULL)
              | (1ULL << 46ULL);
```

In CuTe DSL, `make_umma_smem_desc(src, layout, major, next_src)` builds this
descriptor from a tensor. However, you typically avoid calling this directly —
`make_fragment_A(sA_affine)` calls it internally. [CONFIRMED — gau-nernst]

### 2.4 TMEM allocation / deallocation

```python
# Python DSL (arch module):
arch.alloc_tmem(
    num_columns,            # must be power-of-2, >= 32
    smem_ptr_to_write_address,  # SMEM slot where TMEM base addr is written
    is_two_cta=None,        # None = sm_100 default; True for 2-SM
    arch='sm_100',
)
arch.relinquish_tmem_alloc_permit()   # release per-SM alloc lock
arch.sync_threads()                   # broadcast TMEM addr to all threads
tmem_ptr = arch.retrieve_tmem_ptr(
    element_type=Float32,
    alignment=16,
    ptr_to_buffer_holding_addr=smem_ptr_slot,
)
# ...at kernel exit:
arch.dealloc_tmem(tmem_ptr, num_columns, arch='sm_100')
```

```ptx
# PTX equivalents:
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_addr], nCols;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, nCols;
```

**Constraints:** [CONFIRMED]
- `nCols` must be power-of-2 and >= 32 (minimum alloc = 32 columns)
- Alloc and dealloc must be issued by **the same warp** (not just same thread)
- The warp that allocates must call `relinquish_alloc_permit` before other CTAs can allocate
- Total TMEM capacity = 512 columns; one 128×256 FP32 accumulator = 256 cols (half)

High-level wrapper available:
```python
tmem_allocator = utils.TmemAllocator()  # (or TmemBufferPool)
tmem_allocator.allocate(num_cols)
tmem_allocator.relinquish_alloc_permit()
tmem_allocator.retrieve_ptr(dtype, alignment, smem_slot)
tmem_allocator.free(tmem_ptr, num_cols)
```
[CONFIRMED — stage0_api.md]

### 2.5 TMEM → register load (epilogue)

```python
# CuTe DSL — full warpgroup (4 warps required, each warp covers 32 of 128 lanes):
tiled_t2r = make_tmem_copy(Ld32x32bOp(Repetition.x8), tCtAcc)
# x8 = 8 FP32 values per thread per call
thr_t2r = tiled_t2r.get_slice(thread_idx)
src = thr_t2r.partition_S(tCtAcc)   # TMEM source partitioned to this thread
dst = thr_t2r.make_fragment(src)     # register destination
cute.copy(tiled_t2r, src, dst)
arch.fence_view_async_tmem_load()    # MANDATORY before using dst in registers
```

```ptx
# PTX raw:
tcgen05.ld.sync.aligned.32x32b.x8.b32 {r0,r1,...,r7}, [tmem_addr];
tcgen05.wait::ld.sync.aligned;       # fence after ld, before register use
```

TMEM address encoding: `tmem_addr = (lane << 16) | col`
- lane = warp_id * 32 + lane_in_warp (0-127)
- col = column index (0-511)

[CONFIRMED — gau-nernst, deepwiki, Colfax, stage0_api.md]

### 2.6 ScaleOut / ACCUMULATE field

```python
# CuTe DSL:
tiled_mma.set(Field.ACCUMULATE, False)  # D = A*B     (overwrite, first k-block)
tiled_mma.set(Field.ACCUMULATE, True)   # D = A*B + D (accumulate, subsequent)

# Field.NEGATE_A / NEGATE_B — compute -A*B or A*(-B) in hardware:
tiled_mma.set(Field.NEGATE_A, True)   # → D = -(A*B) + D = D - A*B
# This fuses the QR subtract epilogue H -= Y@W for FREE — no separate subtraction.
```

[CONFIRMED — stage0_api.md probe, `mma.py:531`]

### 2.7 Supported shapes and dtypes

```
tcgen05.mma shapes (1-SM, cta_group::1):
  M = 128      (N rows, fixed for 1SM)
  N = 8, 16, ..., 256  (multiples of 8)
  K = 16 (BF16/FP16), 8 (TF32), 32 (FP8)  per instruction

tcgen05.mma shapes (2-SM, cta_group::2):
  M = 256  (across 2 SMs in a TPC)
  N, K same as above

Input dtypes: BF16, FP16, TF32, FP8(e4m3/e5m2), FP6, FP4, INT8, INT4
Accumulator: FP32, INT32 (always)
NO FP32 input dtype — confirmed by tcgen05_tmem.md §1.3
```

[CONFIRMED — gau-nernst, Colfax, arXiv 2512.02189, tcgen05_tmem.md]

---

## 3. TMA (cp.async.bulk.tensor) + Warp-Specialized Pipelines

### 3.1 make_tiled_tma_atom — full signature

```python
from cutlass.cute.nvgpu.cpasync import (
    make_tiled_tma_atom, CopyBulkTensorTileG2SOp, CopyBulkTensorTileS2GOp,
    tma_partition, prefetch_descriptor, update_tma_descriptor,
    fence_tma_desc_acquire, fence_tma_desc_release,
)

# Load (GMEM → SMEM):
tma_atom_A, tma_tensor_A = make_tiled_tma_atom(
    op=CopyBulkTensorTileG2SOp(cta_group=CtaGroup.ONE),
    gmem_tensor=A_gmem,       # cute.Tensor over global memory
    smem_layout_=sA_layout,   # the staged SMEM layout (rank-4, swizzled)
    cta_tiler=(128, 64),      # (M_tile, K_tile) — must match mma_tiler_mnk[:1] and K
    num_multicast=1,          # 1 for no multicast; cluster size for 2-SM
)
# Returns: (CopyAtom, gmem-tensor-with-TMA-descriptor-embedded)

# Store (SMEM → GMEM):
tma_atom_C, tma_tensor_C = make_tiled_tma_atom(
    op=CopyBulkTensorTileS2GOp(),
    gmem_tensor=C_gmem,
    smem_layout_=sC_layout,
    cta_tiler=(128, 256),
)
```

[CONFIRMED — docs.nvidia.com/cutlass cpasync API, stage0_api.md]

### 3.2 TMA load in-kernel (producer warp)

```python
# One elected thread issues TMA:
with arch.elect_one():
    # Signal expected bytes for the mbarrier:
    arch.mbarrier_arrive_and_expect_tx(tile_ready_mbar[stage], bytes_A + bytes_B)
    # Issue the async copies:
    sA_dst, tma_src_A = tma_partition(tma_atom_A, cta_coord=(0,), cta_layout=Layout(1),
                                       smem_tensor=smem_A[stage], gmem_tensor=tma_tensor_A)
    cute.copy(tma_atom_A, tma_src_A[k_coord], sA_dst)

    sB_dst, tma_src_B = tma_partition(tma_atom_B, ...)
    cute.copy(tma_atom_B, tma_src_B[k_coord], sB_dst)
```

[CONFIRMED — stage0_api.md, Modular blog (TMA single-thread issue pattern)]

### 3.3 PipelineTmaUmma + PipelineUmmaAsync (the two producer-consumer chains)

The two mbarrier chains in §3.2 of `tcgen05_tmem.md` are wrapped by these classes:

```python
# Chain 1: TMA → SMEM → MMA  (TMA produces, UMMA consumes)
pipe_tma_umma = PipelineTmaUmma.create(
    num_stages=NUM_STAGES,
    producer_group=CooperativeGroup(Agent.Thread, size=1),  # TMA warp
    consumer_group=CooperativeGroup(Agent.Thread, size=32), # MMA warp
    tx_count=bytes_per_stage,   # bytes TMA will transfer per stage
    barrier_storage=smem_barrier_ptr,
    cta_layout_vmnk=None,       # None for single-SM
)

# Chain 2: UMMA → TMEM → Epilogue  (UMMA produces, epilogue consumes)
pipe_umma_async = PipelineUmmaAsync.create(
    num_stages=1,               # only 1 accumulator slot usually
    producer_group=CooperativeGroup(Agent.Thread, size=32), # MMA warp
    consumer_group=CooperativeGroup(Agent.Thread, size=128),# epilogue warpgroup
    barrier_storage=smem_mma_done_ptr,
    cta_group=CtaGroup.ONE,
)

# Producer (TMA warp):
producer = pipe_tma_umma.make_producer()
handle = producer.acquire_and_advance()       # wait stage buffer free
# ... issue TMA ...
handle.commit()                               # signal tile-ready

# Consumer/Producer (MMA warp):
consumer_tma = pipe_tma_umma.make_consumer()
producer_umma = pipe_umma_async.make_producer()
handle = consumer_tma.wait_and_advance()      # wait tile-ready
# ... issue cute.gemm ...
handle.release()                              # signal buffer-free
producer_umma.acquire_and_advance().commit()  # signal mma-done (via tcgen05.commit internally)

# Consumer (epilogue warpgroup):
consumer_epi = pipe_umma_async.make_consumer()
handle = consumer_epi.wait_and_advance()      # wait mma-done
# ... tcgen05.ld TMEM → regs → store ...
handle.release()
```

[CONFIRMED class names; method signatures from docs. Usage pattern is LIKELY — confirm
exact argument types against installed wheel. The phase-bit management is internal.]

### 3.4 mbarrier primitives (low-level, if not using Pipeline wrappers)

```python
# Initialize (one thread, once):
arch.mbarrier_init(mbar_ptr, count=1)          # count = arrival count expected

# Producer: announce expected bytes and arrive:
arch.mbarrier_arrive_and_expect_tx(mbar_ptr, bytes)  # TMA producer
arch.mbarrier_arrive(mbar_ptr)                        # non-TMA producer

# Consumer: wait for all arrivals:
arch.mbarrier_wait(mbar_ptr, phase=0)          # blocking
ok = arch.mbarrier_try_wait(mbar_ptr, phase)   # non-blocking → Boolean

# Phase flips 0→1→0 each time the barrier completes one full cycle.
```

[CONFIRMED — arch API docs, stage0_api.md]

### 3.5 Descriptor update for batched GEMM

```python
# For batched QR: update TMA descriptor base address per matrix
arch.fence_tma_desc_release()
update_tma_descriptor(tma_atom, new_gmem_tensor, tma_desc_ptr)
arch.fence_tma_desc_acquire(tma_desc_ptr)
prefetch_descriptor(tma_atom)
```

[CONFIRMED — cpasync API docs]

---

## 4. BF16x9 / Ozaki Exact-FP32

### 4.1 The 3-split (hi/mid/lo) — conceptual

For any FP32 value `x` (24-bit mantissa), split into 3 BF16 values (8-bit mantissa each):
```python
# Python/DSL pseudocode:
x_hi  = bf16(x)               # round-to-nearest-bf16
r1    = x - float32(x_hi)     # exact residual
x_mid = bf16(r1)
r2    = r1 - float32(x_mid)
x_lo  = bf16(r2)
# Invariant: x_hi + x_mid + x_lo == x exactly (3 × 8 = 24 mantissa bits)
```

CuTe DSL has `arch.cvt_f32x2_bf16x2(vec2)` for packed FP32→BF16 conversion.
[CONFIRMED recipe is standard Ozaki/BF16x9; cvt_f32x2_bf16x2 confirmed in arch API]

### 4.2 The 9-product GEMM

Apply the 3-split to both operands A → (A0,A1,A2) and B → (B0,B1,B2), then compute:
```
result = Σ_{i,j ∈ {0,1,2}} MMA(A_i, B_j)   → 9 BF16 GEMMs into 1 FP32 accumulator
```

Using tcgen05 with a single FP32 TMEM accumulator:
```python
# Descending magnitude order: (A0,B0), (A0,B1), (A1,B0), (A0,B2), (A2,B0), (A1,B1), ...
tiled_mma.set(Field.ACCUMULATE, False)   # pass 0: overwrite
for i, j in [(0,0), (0,1), (1,0), (0,2), (2,0), (1,1), (1,2), (2,1), (2,2)]:
    cute.gemm(tiled_mma, tCtAcc, frags_A[i], frags_B[j], tCtAcc)
    tiled_mma.set(Field.ACCUMULATE, True)   # passes 1-8: accumulate
```

**Alternative (NegateA trick for subtract-fuse):**
```python
# For the trailing update H -= Y @ W, use Field.NEGATE_A on the last (or any) pass:
# Set NEGATE_A=True before the MMA call, ACCUMULATE=True → accumulator gets D - A*B
# This fuses the subtract epilogue INTO the MMA loop — no extra epilogue subtraction needed.
tiled_mma.set(Field.NEGATE_A, True)
tiled_mma.set(Field.ACCUMULATE, True)
cute.gemm(tiled_mma, tCtAcc, frags_A[i], frags_B[j], tCtAcc)
```

[CONFIRMED mechanism; the exact 9-pass ordering above is LIKELY best-practice magnitude-descending;
the NegateA fusion is CONFIRMED from Field enum definition in stage0_api.md]

### 4.3 Performance expectations (from primary sources)

| Config | Perf | Source |
|--------|------|--------|
| BF16x9 vs native FP32 on GB200, large shapes | up to 3× | NVIDIA blog |
| BF16x9, ecTrans (real app) | 2.4× | NVIDIA blog |
| B200 FP16→FP32 MMA throughput | 482.4 TFLOPS | arXiv 2512.02189 |
| B200 BF16 MMA throughput (native) | 1,926 TFLOPS | arXiv 2512.02189 |
| BF16x9 theoretical max vs native BF16 | ~(1926/9)×fuse ~ 214 TFLOPS | estimate |
| cublasLt type-78 on 640×512×512 GEMM | 2.08× vs FP32 | findings B6 |
| BF16x9 breakeven K | ~128 (K=32: 0.43×, K=256: 1.39×) | findings B7 |

[CONFIRMED numbers from cited sources; BF16x9 theoretical is derived estimate]

### 4.4 cublasLt type-78 (Stage-1 baseline)

```python
# ctypes path (cuda.bindings has no cublas submodule):
import ctypes
libcublaslt = ctypes.CDLL("libcublasLt.so.13")
# CUBLAS_COMPUTE_32F_EMULATED_16BFX9 = 78
# Use with: computeType=78, scaleType=CUDA_R_32F, alpha/beta=FP32
# beta=1.0 fuses the subtract (C = alpha*A*B + beta*C with alpha=-1)
```

Status: still present in cu130 on the grader. Deprecated in cuBLAS 13.3 but not yet removed.
[CONFIRMED — tcgen05_tmem.md §6.4; findings B6/v18/v22]

### 4.5 In-kernel BF16x9 SMEM budget consideration

For tcgen05 BF16x9, operands need 3× SMEM vs single-precision:
- A tile (M=128, K=64, BF16): 128×64×2 = 16 KB per stage
- With 3 splits × 2 stages = 96 KB for A alone
- Plus B and barriers — budget tightly against 228 KB SMEM limit

[CONFIRMED concern — tcgen05_tmem.md §7.3; LIKELY need to measure exact budget]

---

## 5. Triton Gluon tcgen05 Path

### 5.1 Imports

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    fence_async_shared,
)
```

[CONFIRMED — extracted verbatim from triton/python/tutorials/gluon/06-tcgen05.py]

### 5.2 TMEM allocation (Gluon)

```python
tmem_layout: gl.constexpr = TensorMemoryLayout(
    (BLOCK_M, BLOCK_N),          # tile shape
    col_stride=32 // dtype.primitive_bitwidth,
)
acc_tmem = allocate_tensor_memory(
    element_ty=gl.float32,
    shape=[BLOCK_M, BLOCK_N],
    layout=tmem_layout,
)
```

[CONFIRMED — verbatim from 06-tcgen05.py]

### 5.3 MMA issuance (Gluon)

```python
# Simple case (single tile):
tcgen05_mma(a_smem, b_smem, acc_tmem)           # use_acc defaults to False (overwrite)
tcgen05_commit(mma_bar)                          # tcgen05.commit to mbarrier

# With mbarrier inline (no separate commit needed):
tcgen05_mma(a_smem, b_smem, acc_tmem, mbarriers=[mma_bar], mbarrier_preds=[True])

# K-loop accumulation:
use_acc = False
for k in range(0, K, BLOCK_K):
    # ... TMA load a_smem, b_smem ...
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=use_acc)
    tcgen05_commit(mma_bar)
    mbarrier.wait(mma_bar, phase=phase)
    use_acc = True
    phase ^= 1
```

[CONFIRMED — verbatim from blocked_matmul_kernel in 06-tcgen05.py]

### 5.4 TMA (Gluon)

```python
# Prepare descriptor from tensor (host side):
a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

# In-kernel: expect bytes + async load:
mbarrier.expect(tma_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
tma.async_load(a_desc, [off_m, k], tma_bar, a_smem)
mbarrier.wait(tma_bar, phase=phase)

# Store:
tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
tma.store_wait(pendings=0)
```

[CONFIRMED — verbatim from 06-tcgen05.py]

### 5.5 Pipelined pattern (Gluon, from blocked_matmul_pipelined_kernel)

The pipelined kernel uses double-buffered SMEM + alternating mbarrier indices:
```python
# Double-buffer index management:
@gluon.jit
def get_and_increment(counter):
    return counter % 2, counter // 2 & 1, counter + 1
# Returns: (buffer_index, phase_bit, new_counter)

# Two separate pairs of TMA and MMA mbarriers (load_bars, mma_bars), each with 2 slots:
load_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
mma_bars  = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
for i in gl.static_range(2):
    mbarrier.init(load_bars.index(i), count=1)
    mbarrier.init(mma_bars.index(i),  count=1)
```

[CONFIRMED — verbatim from 06-tcgen05.py blocked_matmul_pipelined_kernel]

### 5.6 Gluon vs CuTe-DSL trade-offs

| Aspect | Triton Gluon | CuTe-DSL |
|--------|-------------|---------|
| Shipping | Import at runtime (Triton must be present) | cubin embed, driver-load |
| TMEM control | `allocate_tensor_memory` (high-level) | `arch.alloc_tmem` (raw) |
| BF16x9 | No built-in support; must layer manually | `Field.NEGATE_A`, cvt helpers |
| Warp-spec pipelines | Manual mbarrier management | PipelineTmaUmma/PipelineUmmaAsync |
| SMEM layout | `NVMMASharedLayout.get_default_for(...)` | `make_smem_layout_a/b` |
| `recast_ptr` needed | No (Gluon handles swizzle internally) | YES — critical |
| Hang risk | Initial test hung GPU (stage1_smoke.py) | Compile errors, not hangs |
| Status on grader | Triton available; TMEM extension untested | cubin path proven (B8) |

[CONFIRMED — from tcgen05_stage0_api.md §Stage-1 status, gluon tutorial]

---

## 6. Blackwell Microbenchmarking — Perf Model Numbers

Source: arXiv 2512.02189 "Microbenchmarking NVIDIA's Blackwell Architecture" [CONFIRMED]

### 6.1 TMEM bandwidth

- **16 TB/s read bandwidth** per SM [CONFIRMED — arXiv 2512.02189]
- Saves ~12 TB/s of data movement per SM compared to Hopper (no TMEM intermediate writes to global)
- Optimal tile size for TMEM efficiency: 64×64 elements at FP8 = 4KB; tiles smaller than 32×32 underutilize the 1024-bit memory interface

### 6.2 tcgen05 single-instruction latency (SI-LAT)

| Operation | Tile Shape | Latency (cycles) |
|-----------|-----------|------------------|
| tcgen05.mma FP16 | m64n64k16 | **11.0** |
| tcgen05.mma FP16 | m128n128k16 | **11.3** |
| tcgen05.mma FP16 | m256n256k16 | **11.4** |
| Hopper wgmma FP16 | m64n64k16 | 32.0 |

Blackwell achieves **2.9–11.2× lower latency** than Hopper. [CONFIRMED — arXiv 2512.02189]

### 6.3 Peak throughput (B200)

| Precision | Throughput |
|-----------|-----------|
| FP16 → FP32 accumulator | **482.4 TFLOPS** |
| FP16 → FP16 accumulator | 964.8 TFLOPS |
| BF16 → FP32 accumulator | **~482 TFLOPS** (same datapath as FP16) |
| FP8 → FP32 | 1,912.8 TFLOPS |
| BF16 (native, FP16 acc) | 1,926.4 TFLOPS |

**Critical:** "FP32 accumulation halves throughput" — the accumulator write datapath, not the multiply units, is the bottleneck. BF16x9 over 9 MMA calls each at 482 TFLOPS effective → theoretical max ≈ 54 TFLOPS (9 passes). But each pass reuses the TMEM accumulator without SMEM roundtrip, so overlap is key. [CONFIRMED raw numbers; efficiency analysis is ours]

### 6.4 CTA-pair (2-SM) gains

Training speedup decomposes as:
- SM count increase: 1.09×
- CTA pairing: **1.27×**
- TMEM: **1.26×**

For a standalone GEMM, 2-SM gives near-2× throughput (weak scaling confirmed perfect).
[CONFIRMED — arXiv 2512.02189]

### 6.5 SMEM bandwidth limit

MMA in SS mode (both A, B from SMEM) is bandwidth-constrained at N<128: ~128 B/cycle
SMEM bandwidth per SM. For our shapes (N=256, K=64) this should not be a bottleneck
since the innermost MMA K-tile is 16 BF16 = 32 bytes per A-row. [CONFIRMED limit; our shape LIKELY ok]

### 6.6 Perf model for BF16x9 trailing GEMM

For a trailing update of shape (M=128, N=256, K=128, batch=640):
```
Total FP16-equivalent MACs per batch element = 2 × 128 × 256 × 128 = 8.4M
At 482 TFLOPS / SM (FP32 acc) = 482e12 MACs/s
Per-SM compute time for one tile (9 BF16x9 passes, K=128) ≈ 9 × 2×128×256×128 / 482e12
≈ 9 × 8.4e6 / 482e12 ≈ 156 ns per SM

With 148 SMs, processing 640 matrices serially per SM: 640/148 ≈ 4.3 matrices/SM
Total compute time estimate: 4.3 × (number of K-tiles) × 156 ns
```

This is a rough lower bound. Pipeline depth and memory-bound phases dominate in practice.
[Derived from CONFIRMED throughput numbers; end-to-end timing is UNVERIFIED until Stage 1]

---

## 7. Warp-Specialized Persistent Kernel Skeleton (Python DSL)

### 7.1 Complete skeleton

```python
@cute.kernel
def blackwell_gemm_persistent(
    A_ptr, B_ptr, C_ptr,
    M: cute.Const, N: cute.Const, K: cute.Const, batch: cute.Const,
    tma_atom_A, tma_tensor_A,
    tma_atom_B, tma_tensor_B,
):
    # ── Shared memory ───────────────────────────────────────────
    smem_A = cute.alloc_shared(BFloat16, smem_A_layout)     # rank-4, staged
    smem_B = cute.alloc_shared(BFloat16, smem_B_layout)
    smem_mbar_tma   = cute.alloc_shared(cute.Int64, [NUM_STAGES])
    smem_mbar_umma  = cute.alloc_shared(cute.Int64, [1])
    smem_tmem_slot  = cute.alloc_shared(cute.UInt32, [1])

    warp_id = arch.warp_idx()
    thread_id = arch.thread_idx()
    block_id = arch.block_idx()
    num_blocks = arch.grid_dim()

    # ── One-time setup ──────────────────────────────────────────
    if warp_id == 0:
        for s in range(NUM_STAGES):
            arch.mbarrier_init(smem_mbar_tma[s], count=1)
        arch.mbarrier_init(smem_umma_mbar, count=1)

    if warp_id == 1:
        with arch.elect_one():
            arch.alloc_tmem(num_tmem_cols, smem_tmem_slot, arch='sm_100')
            arch.relinquish_tmem_alloc_permit()

    arch.sync_threads()
    tmem_ptr = arch.retrieve_tmem_ptr(Float32, 16, smem_tmem_slot)

    # Build accumulator tensor pointing to TMEM:
    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

    # ── Persistent tile loop ────────────────────────────────────
    tile_id = block_id
    while tile_id < num_tiles:
        m_tile, n_tile, batch_tile = get_tile_coords(tile_id)

        if warp_id < NUM_TMA_WARPS:                          # PRODUCER warp(s)
            phase = 0
            for k in range(0, K, BLOCK_K):
                stage = (k // BLOCK_K) % NUM_STAGES
                with arch.elect_one():
                    arch.mbarrier_arrive_and_expect_tx(smem_mbar_tma[stage], bytes_AB)
                    cute.copy(tma_atom_A, tma_tensor_A[m_tile, k], smem_A[:,:,:,stage])
                    cute.copy(tma_atom_B, tma_tensor_B[n_tile, k], smem_B[:,:,:,stage])

        elif warp_id == MMA_WARP:                            # MMA warp (1 thread issues)
            tiled_mma.set(Field.ACCUMULATE, False)
            phase = 0
            for k in range(0, K, BLOCK_K):
                stage = (k // BLOCK_K) % NUM_STAGES
                arch.mbarrier_wait(smem_mbar_tma[stage], phase)
                with arch.elect_one():
                    sA_this = get_affine_smem_stage(smem_A, stage)
                    sB_this = get_affine_smem_stage(smem_B, stage)
                    tCrA = tiled_mma.make_fragment_A(sA_this)
                    tCrB = tiled_mma.make_fragment_B(sB_this)
                    cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
                    tiled_mma.set(Field.ACCUMULATE, True)
                    if (k // BLOCK_K + 1) % NUM_STAGES == 0:
                        phase ^= 1
                # Signal buffer free for TMA to reuse:
                arch.mbarrier_arrive(smem_mbar_buf_free[stage])
            # Signal UMMA done to epilogue:
            tcgen05.commit(smem_mbar_umma, cta_group=CtaGroup.ONE)

        else:                                                # EPILOGUE warpgroup (warps 4-7)
            arch.mbarrier_wait(smem_mbar_umma, phase=0)
            tiled_t2r = make_tmem_copy(Ld32x32bOp(Repetition.x8), tCtAcc)
            thr_t2r = tiled_t2r.get_slice(thread_id)
            src = thr_t2r.partition_S(tCtAcc)
            dst = thr_t2r.make_fragment(src)
            cute.copy(tiled_t2r, src, dst)
            arch.fence_view_async_tmem_load()
            # Subtract-fuse epilogue: C -= dst  (or use Field.NEGATE_A above)
            store_to_global_with_subtract(C_ptr, dst, m_tile, n_tile)

        tile_id += num_blocks

    # ── Cleanup ─────────────────────────────────────────────────
    if warp_id == 1:
        with arch.elect_one():
            arch.dealloc_tmem(tmem_ptr, num_tmem_cols, arch='sm_100')
```

[Structure CONFIRMED against tcgen05_tmem.md §3 + Colfax + gau-nernst. Exact DSL syntax
for `cute.alloc_shared` and the elect_one context manager form are LIKELY — probe the wheel.
The skeleton captures all the necessary synchronization points.]

### 7.2 Key gotchas checklist (for debugging)

1. **recast_ptr is mandatory.** `cute.gemm` rejects composed layouts. Apply before `make_fragment_A/B`.
2. **TMEM alloc = single warp** (not single thread). Use `with arch.elect_one()` inside a warp-selected branch.
3. **`relinquish_tmem_alloc_permit` before `sync_threads`** — otherwise other CTAs can't schedule this SM.
4. **Epilogue = full warpgroup (4 warps)**. Each covers 32 of 128 TMEM lanes. Single warp = wrong data.
5. **`fence_view_async_tmem_load()` after every TMEM load** before using registers.
6. **`tcgen05.commit` before epilogue waits** — MMA is async, no implicit fence.
7. **K-loop: ACCUMULATE=False on first block, True on all subsequent** (including BF16x9's 9 passes).
8. **SMEM swizzle must match descriptor bits** — use `make_smem_layout_a/b` + `recast_ptr`, not manual layouts.
9. **mbarrier phase flips 0→1→0** each full cycle — track phase in the K-loop.
10. **TMEM alloc is power-of-2 ≥ 32 columns** — round up.

---

## 8. API Quick-Reference Card

### 8.1 The 6 key helper calls in order

```python
# 1. Create MMA atom
tiled_mma = utils.make_trivial_tiled_mma(a_dtype, b_dtype, a_leading, b_leading, acc_dtype, cta_group, (M,N))

# 2. Get swizzled SMEM layouts
sA_layout = utils.make_smem_layout_a(tiled_mma, (M,N,K), a_dtype, num_stages)
sB_layout = utils.make_smem_layout_b(tiled_mma, (M,N,K), b_dtype, num_stages)

# 3. Move swizzle to pointer
pA = cute.recast_ptr(smem_A.iterator, sA_layout.inner, a_dtype)
sA_affine = cute.make_tensor(pA, sA_layout.outer)

# 4. Create MMA operand fragments (SMEM descriptors)
tCrA = tiled_mma.make_fragment_A(sA_affine_stage0)
tCrB = tiled_mma.make_fragment_B(sB_affine_stage0)

# 5. Create TMEM accumulator
acc_shape = tiled_mma.partition_shape_C((M, N))
tCtAcc = cute.make_tensor(tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)

# 6. Issue MMA
cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
```

### 8.2 Field enum values (for MMA control)

```python
Field.ACCUMULATE   # True = D += A*B; False = D = A*B (overwrite)
Field.NEGATE_A     # True = D += -(A*B)   ← use for subtract-fuse trailing update
Field.NEGATE_B     # True = D += A*(-B)
Field.SFA          # scale factor for A (block-scaled ops only)
Field.SFB          # scale factor for B (block-scaled ops only)
```

### 8.3 SmemLayoutAtomKind values

```python
K_INTER=6, K_SW32=7, K_SW64=8, K_SW128=9       # K-major variants
MN_INTER=1, MN_SW32=2, MN_SW64=3, MN_SW128=4   # MN-major variants
MN_SW128_32B=5                                   # MN-major 128B/32B variant
```

Use `K_SW128` for standard BF16/FP16 K-major inputs. [CONFIRMED — stage0_api.md]

### 8.4 TMEM load ops and when to use them

```python
Ld32x32bOp(Repetition.x8)   # 8 FP32/thread, 32 lanes, 32 bits — general M=128,N≤256
Ld16x256bOp(...)             # 256b wide, 16 lanes per warp — wide N
Ld16x128bOp(...)             # 128b wide, 16 lanes — medium N
Ld16x64bOp(...)              # 64b wide, 16 lanes — narrow N
```

Use `utils.get_tmem_load_op(cta_tile_shape, layout_d, elem_ty_d, elem_ty_acc, epi_tile, use_2cta)` to
auto-select. [CONFIRMED — utils API docs]

### 8.5 Compile and extract cubin (PROVEN)

```python
from cutlass.cute import KeepCUBIN, KeepPTX, OptLevel
compiled = cute.compile[OptLevel(3), KeepCUBIN, KeepPTX](entry_fn, *sample_tensors)
cubin = compiled.artifacts.CUBIN        # bytes
ptx   = compiled.artifacts.PTX
sym   = list(compiled.kernel_info.keys())[0]   # GPU entry symbol name
```

[CONFIRMED — findings B8, tcgen05_tmem.md §5.8]

---

## 9. Sources

| Source | URL | What's confirmed |
|--------|-----|-----------------|
| gau-nernst tcgen05 tutorial | https://gau-nernst.github.io/tcgen05/ | idesc, smem-desc, K-loop, alloc/dealloc PTX |
| Colfax TMEM GEMM tutorial | https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/ | TiledMMA, TmemAllocator, ScaleOut, 4-warp epilogue |
| Colfax TBC (2-SM) | https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/ | cluster layout, TMA multicast masks, SMEM bit24 trick |
| deepwiki tcgen05 | https://deepwiki.com/gau-nernst/learn-cuda/8.1-tcgen05-instructions-and-tensor-memory | warp roles, lane encoding, mbarrier phases |
| arXiv 2512.02189 | https://arxiv.org/html/2512.02189v2 | TMEM 16 TB/s, tcgen05 11.0-11.4 cycle SI-LAT, 482 TFLOPS FP16→FP32 |
| SemiAnalysis Blackwell | https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor | single-thread issue, M=128 near 100% peak, SMEM BW limit |
| Triton Gluon 06-tcgen05.py | https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/06-tcgen05.py | COMPLETE source: TensorMemoryLayout, tcgen05_mma, double-buffer pipeline |
| CUTLASS utils_sm100 API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html | make_trivial_tiled_mma, make_smem_layout_a/b, get_num_tmem_alloc_cols |
| CUTLASS tcgen05 API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html | MmaF16BF16Op, Field, SmemLayoutAtomKind, Ld32x32bOp, make_tmem_copy |
| CUTLASS cpasync API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html | make_tiled_tma_atom, tma_partition, update_tma_descriptor |
| CUTLASS arch API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html | alloc_tmem, mbarrier_*, elect_one, cvt_f32x2_bf16x2 |
| CUTLASS pipeline API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html | PipelineTmaUmma, PipelineUmmaAsync, PipelineState, NamedBarrier |
| NVIDIA BF16x9 blog | https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/ | up to 3× vs FP32, 2.4× ecTrans, accuracy equivalent |
| blackwell_helpers.py | https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py | make_trivial_tiled_mma supports both ab_dtype and a_dtype/b_dtype |
| Modular Blackwell blog | https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul | Swizzle<3,4,3>, LBO/SBO, TMA+tcgen05 = 155→288 TFLOPS |
| CUTLASS examples search | https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py | Exists; GitHub raw 404 (use gh or clone on Modal) |
| Stage0 probe | tcgen05/stage0_probe.py + tcgen05_stage0_api.md | All module contents probed on installed 4.5.2 wheel |
| opus progress | tcgen05/opus_progress.md | recast_ptr pattern, make_fragment_A/B vs partition_A/B |

---

## UNVERIFIED flags (do NOT build on without checking)

- `cute.alloc_shared` DSL syntax for SMEM allocation — probe the wheel (might be a decorator param)
- `arch.fence_view_async_tmem_load` exact DSL name — confirmed in stage0_api.md as `fence_view_async_tmem_load/store`; exact call syntax untested
- `PipelineTmaUmma.create` exact `cta_layout_vmnk` and `tx_count` types — probe before use
- The exact BF16x9 9-pass magnitude ordering — confirmed the mechanism; the specific ordering above (A0B0 first) is conventional but not bit-verified for our shapes
- CUTLASS Python examples source bodies (fp16_gemm_0.py, dense_blockscaled_gemm_persistent.py) — GitHub raw 404; pull from Modal clone
