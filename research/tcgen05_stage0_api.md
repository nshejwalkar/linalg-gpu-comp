# Stage 0 RESULTS — pinned CuTe-DSL tcgen05/TMEM API (CONFIRMED on Modal B200)

Resolves the "unverified" flags in `tcgen05_tmem.md §9.5`. Probed against the INSTALLED wheel
`nvidia-cutlass-dsl 4.5.2` on the B200 grader-mirror image. Source: `tcgen05/stage0_probe.py`.
(The Stage-1 Gluon smoke test that ran AFTER this hung the GPU — see note at bottom.)

## `cutlass.cute.nvgpu.tcgen05` — the MMA + TMEM-copy atoms
- **MMA op (what we need for BF16x9):** `MmaF16BF16Op(ab_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode)`. Also `MmaTF32Op`, `MmaFP8Op`, `MmaF8F6F4Op`, block-scaled variants.
- `CtaGroup`: `ONE`, `TWO` (2-SM/CTA-pair UMMA). `OperandSource`: `SMEM`, `TMEM`. `OperandMajorMode`: `K`, `MN`.
- ⭐ **`Field`: `ACCUMULATE`, `NEGATE_A`, `NEGATE_B`, `SFA`, `SFB`** — `NEGATE_A`/`NEGATE_B` let the MMA compute `-A@B` / `A@(-B)` in-hardware → **fuses our trailing subtract `H -= Y@W` for free** (set NEGATE + ACCUMULATE).
- **TMEM↔reg copy atoms:** `Ld32x32bOp`, `Ld16x256bOp`, `Ld16x128bOp`, `Ld16x64bOp`, `Ld16x32bx2Op`, `St32x32bOp`, `St16x*` (+ `LdRed*` reduce variants), with `Repetition` (x1..x128) and `Pack`/`Unpack`.
- **Helpers:** `make_tmem_copy(atom, tmem_tensor)`, `make_s2t_copy`, `make_umma_smem_desc(src, layout, major,...)`, `make_smem_layout_atom(kind, dtype)`, `commit(mbar_ptr, ..., cta_group)`, `get_tmem_copy_properties`, `find_tmem_tensor_col_offset`.
- `SmemLayoutAtomKind`: `K_INTER/K_SW32/K_SW64/K_SW128`, `MN_INTER/MN_SW32/MN_SW64/MN_SW128/MN_SW128_32B` (swizzles).

## `cutlass.cute.arch` — TMEM alloc, mbarriers, BF16x9 math, warp-spec
- **TMEM:** `alloc_tmem(num_columns, smem_ptr_to_write_address, is_two_cta=None, arch='sm_100')`, `dealloc_tmem(ptr, num_columns,...)`, `relinquish_tmem_alloc_permit(...)`, `retrieve_tmem_ptr`, `get_max_tmem_alloc_cols`/`get_min_tmem_alloc_cols`. Constants: `SM100_TMEM_CAPACITY_COLUMNS`, `SM100_TMEM_MIN_ALLOC_COLUMNS`, `TMEM_MAX/MIN_ALLOC_COLUMNS_MAP`.
- **mbarriers (sync):** `mbarrier_init(mbar,cnt)`, `mbarrier_arrive`, `mbarrier_arrive_and_expect_tx(mbar,bytes)`, `mbarrier_wait(mbar,phase)`, `mbarrier_try_wait`, `mbarrier_conditional_try_wait`, `mbarrier_init_fence`.
- ⭐ **BF16x9 / Ozaki math:** `cvt_f32_bf16`, `cvt_f32x2_bf16x2` (split FP32→BF16 hi/lo), and `add_packed_f32x2`, `mul_packed_f32x2`, `fma_packed_f32x2` (recombine the 9 products), `sub_packed_f32x2`. This is the in-register split/accumulate path.
- **Warp-spec / regs:** `elect_one()`, `warp_idx()`, `lane_idx()`, `setmaxregister_increase/decrease`, `warpgroup_reg_alloc/dealloc`, `make_warp_uniform`, `shuffle_sync*`, `warp_reduction_sum/max`, `vote_*`.
- **Fences:** `fence_view_async_tmem_load/store`, `fence_view_async_shared`, `fence_proxy`, `fence_acq_rel_*`.

## `cutlass.cute.nvgpu.cpasync` — TMA
- `CopyBulkTensorTileG2SOp(cta_group)`, `CopyBulkTensorTileS2GOp()`, multicast + reduce variants.
- `make_tiled_tma_atom(op, gmem_tensor, smem_layout, cta_tiler, num_multicast=1)`, `tma_partition(...)`, `prefetch_descriptor`, `create_tma_multicast_mask`.

## `cutlass.pipeline` — warp-spec pipelines (real class names)
- `PipelineTmaUmma` (sm100), `PipelineUmmaAsync` (sm100), `PipelineTmaAsync`, `PipelineAsync` (sm90).
- API: `.create(...)`, `make_producer`/`make_consumer`/`make_participants`, `producer_acquire`/`producer_commit`/`producer_try_acquire`/`producer_tail`, `consumer_wait`/`consumer_release`/`consumer_try_wait`.
- `PipelineState`, `make_pipeline_state`, `NamedBarrier`, `CooperativeGroup`.

## `cutlass.utils` — allocators + schedulers + Blackwell helpers
- **`TmemAllocator`** (`.allocate`, `.free`, `.retrieve_ptr`, `.wait_for_alloc`, `.reserve`, `.relinquish_alloc_permit`, `.check_valid_num_columns`), **`SmemAllocator`**, `TmemBufferPool`.
- `make_trivial_tiled_mma(*args, **kwargs)` (signature is `*args` — inspect a working example for the real call), `get_num_tmem_alloc_cols`, `get_tmem_load_op`, `compute_epilogue_tile_shape`, `get_smem_store_op`, `make_smem_layout_a/b/epi`.
- `cutlass.utils.blackwell_helpers`: `cluster_shape_to_tma_atom_A/B`, `get_smem_layout_atom_ab/epi`, `make_trivial_tiled_mma`, `tile_to_mma_shape`. **NOTE: `cutlass.utils.blackwell` does NOT exist** — it's `blackwell_helpers`.
- Schedulers: `StaticPersistentTileScheduler`, `PersistentTileSchedulerParams`, etc.

## ⚠️ Stage-1 status + the hang
- The Stage-1 Gluon smoke test (`tcgen05/stage1_smoke.py`, "128x256x64 BF16->FP32 single tile") **HUNG the GPU** (ran 1h36m, killed). A bad tcgen05 launch hangs the B200 → the Modal job never returns.
- **Build rule for the relaunch:** run every Stage-1 Modal smoke test **FOREGROUND under `timeout 120`** (never an unbounded background bash). The earlier freeze was the agent backgrounding an unbounded hung run.
- Route decision still open (Gluon vs CuTe-DSL): Stage 0 confirms the CuTe-DSL API is fully present and rich (above). The Gluon route's first attempt hung; re-evaluate with bounded timeouts.
