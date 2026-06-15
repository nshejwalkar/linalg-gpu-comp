# QR Competition — context for Claude

> **Resuming after an interruption (e.g. Claude's 5h limit)? Read [RESUME.md](RESUME.md) first** —
> it has the exact current state + next actions. `auto_bench.py` continues deterministic
> benchmarking/tuning with NO Claude needed (survives rate limits).


GPU MODE `qr` leaderboard (`gpu-mode/reference-kernels`, `problems/linalg/qr_py`).
Goal: a fast **batched Householder QR** returning `(H, tau)` in `torch.geqrf` /
LAPACK SGEQRF compact format, running on a **B200**.

## The task

- Input: `A`, a `batch x n x n` CUDA tensor, `torch.float32`.
- Output: `(H, tau)` exactly as `torch.geqrf(A)` would produce:
  - `H` `(batch,n,n)`: `R` in the upper triangle, Householder vectors below the diagonal (unit diagonal implicit).
  - `tau` `(batch,n)`: reflector coefficients.
- Checker materializes `Q = torch.linalg.householder_product(H, tau)`, `R = triu(H)`.
- **Hard gates** (relative, measured in FP64, no atol):
  - factor residual `R - Qᵀ@A`: `rtol = 20·n·eps32`
  - orthogonality `QᵀQ - I`: `rtol = 100·n·eps32`
  - (triangularity is implied by the factor residual against `triu(H)`)
- **Ranking**: geometric mean of runtime across the 7 benchmark shapes.
  Side prizes: fastest / most elegant / strangest.

## Benchmark shapes (the ranked set — from `reference_repo/task.yml`)

| batch | n | cond | note |
|------:|---:|----:|------|
| 20  | 32   | 1 | tiny — overhead-bound |
| 40  | 176  | 1 | |
| 40  | 352  | 1 | |
| 640 | 512  | 2 | **headline case, "especially important"** |
| 60  | 1024 | 2 | |
| 8   | 2048 | 1 | |
| 2   | 4096 | 1 | small batch, huge matrix |

**Geomean weights every shape equally in log-space.** Being 10× slower than the
baseline on the n=32 case costs exactly as much as 10× slower on n=4096. The
three small cases and overhead/launch latency are first-class concerns, not
afterthoughts. `torch.geqrf` (cuSOLVER) is the baseline to beat per-case.

## Directory layout

```
qr_competition/
  CLAUDE.md            ← this file
  modal_qr.py          ← Modal launcher: private B200 iteration (test + bench)
  submission.py        ← ACTIVE candidate (currently = blocked_wy). Edit/replace this.
  reference_repo/      ← the real competition harness (do not edit)
    reference.py         generate_input / check_implementation / ref_kernel=torch.geqrf
    eval.py              official test/benchmark/leaderboard runner (POPCORN_FD protocol)
    task.py  task.yml    types + real test/benchmark specs
    utils.py             clear_l2_cache, set_seed, DeterministicContext
    submission.py        starter placeholder (geqrf); ignore — popcorn swaps in ours
  archive/             ← prior work (Sonnet's first pass)
    submission_blocked_wy.py   pure-torch blocked WY Householder (CPU-validated, known-good)
    submission_triton.py       Triton panel-factorization variant (UNVALIDATED, see gotchas)
    test_local.py              old CPU correctness suite (hardcoded /home/claude paths)
    test_against_reference.py  old CPU ref-checker run (hardcoded paths)
    STRATEGY.md                Trefethen/Schreiber–Van Loan derivation notes (background)
```

## Research notes
- **[research/findings.md](research/findings.md) — the trial-and-error lab notebook.**
  Structured (Observation → Evidence → Takeaway, with CONFIRMED/REJECTED/OPEN tags). Consult
  this FIRST before re-trying something; it records what works and what's already failed.

Deep-dive background in `research/`:
- [research/qr_algorithms.md](research/qr_algorithms.md) — why the output contract forces
  Householder/WY (rules out CholeskyQR), shape-regime strategy, recursive panel.
- [research/b200_hardware.md](research/b200_hardware.md) — precision throughput table,
  TF32 = biggest free lever, compute- vs launch-bound regimes, shared-mem fit rule.
- [research/profiling.md](research/profiling.md) — ncu/nsys unusable on Modal; use
  torch.profiler; CUDA graphs as the launch-overhead fix.

## Windows + Modal gotchas (important)
- **Always set `PYTHONUTF8=1`** before `modal run` (PowerShell: `$env:PYTHONUTF8=1`).
  Modal's CLI prints Unicode (`✓`); the Windows cp1252 console otherwise crashes with
  `'charmap' codec can't encode`.
- `modal_qr.py` ships harness + submission **file contents as function args** instead of
  mounting the local dir. Mounting (even `copy=True`) trips *"<file> was modified during
  build process"* on Windows because Defender/the search indexer touches files during the
  multi-second image build. Passing strings sidesteps the watcher.

## Two loops: iterate privately, submit officially

### 1. Iterate on Modal (our account, our B200, NO submission slot used)
Run from the `modal` conda env, inside this folder:

```bash
conda activate modal              # has modal 1.5.0; torch/triton run in the remote image
modal run modal_qr.py                                 # correctness + benchmark vs geqrf
modal run modal_qr.py --mode test                     # correctness only (fast)
modal run modal_qr.py --mode bench                    # benchmark only
modal run modal_qr.py --mode baseline                 # just time torch.geqrf
modal run modal_qr.py --submission archive/submission_blocked_wy.py
modal run modal_qr.py --mode bench --tf32             # allow TF32 matmuls (speed/accuracy knob)
```

- `--submission` is a path relative to this folder; defaults to `submission.py`.
- The container image is torch (cu128, Blackwell-capable) + triton, built once and cached.
  Local file edits re-sync on every `modal run`.
- The harness imports the REAL `reference.py` (same generate_input + checker the grader
  uses) and times with CUDA events + `clear_l2_cache`, mirroring `eval.py`. It prints
  per-shape ms for our kernel and `torch.geqrf`, plus the geomean speedup.
- Modal profile in use: `neelandjay` (`~/.modal.toml`). If `gpu="B200"` is unavailable,
  fall back to `H100`/`H200` in `modal_qr.py` for logic checks (timings won't be ranked-accurate).

### 2. Submit officially with popcorn-cli (uses a real slot — ask before leaderboard mode)
```bash
cp <winner>.py submission.py            # if not already the active file
popcorn-cli register discord            # interactive, one-time (needs Discord)
popcorn-cli submit --gpu B200 --leaderboard qr --mode test        submission.py
popcorn-cli submit --gpu B200 --leaderboard qr --mode benchmark   submission.py
popcorn-cli submit --gpu B200 --leaderboard qr --mode leaderboard submission.py  # ranked — ASK FIRST
```
popcorn-cli numbers are the source of truth. Use Modal to get there cheaply.

## Baseline to beat — torch.geqrf, ranked on B200 (geomean ≈ 45 ms)

| shape | geqrf | note |
|---|---:|---|
| b20 n32   | 323 µs  | overhead-bound |
| b40 n176  | 22.1 ms | geqrf already slow |
| b40 n352  | 50.4 ms | |
| b640 n512 | **1070 ms** | headline case — geqrf barely batches |
| b60 n1024 | 239 ms  | |
| b8 n2048  | 76.5 ms | |
| b2 n4096  | 51.9 ms | |

**Key insight:** cuSOLVER's batched geqrf under-parallelizes across the batch.
Large-batch cases (b640 n512 = 1070ms ≈ 1.7ms/matrix serialized; the b40 cases)
are catastrophically slow; small-batch/large-n is comparatively fine. Our blocked-WY
+ batched `bmm` design (all batch elements' GEMMs concurrent on tensor cores) directly
attacks the slow cases. **The big-batch shapes are where the leaderboard is won.**
NOTE: all 7 ranked benchmark shapes are **dense** (no `case` field). Structured types
(band/upper/rankdef/clustered/nearcollinear/rowscale/nearrank) appear ONLY in the
correctness `tests` and are pass/fail, NOT timed.

## Results log — blocked_wy (current submission.py), FP32, B200 via Modal

19/19 correctness ✓. Per-shape vs geqrf (Modal bench, mirrors eval.py):

| shape | geqrf | blocked_wy | speedup | winner |
|---|---:|---:|---:|---|
| b20 n32   | 0.33 ms | 5.48 ms | 0.06× | geqrf (16×) |
| b40 n176  | 21.8 ms | 33.1 ms | 0.66× | geqrf |
| b40 n352  | 51.2 ms | 71.3 ms | 0.72× | geqrf |
| b640 n512 | 1073 ms | 114 ms  | **9.45×** | **us** |
| b60 n1024 | 240 ms  | 212 ms  | 1.14× | us |
| b8 n2048  | 77 ms   | 406 ms  | 0.19× | geqrf (5×) |
| b2 n4096  | 52 ms   | 828 ms  | 0.06× | geqrf (16×) |

**Geomean 0.448× (behind overall).** Clear split: we win **large-batch** (the Python
block loop is hidden by batch-parallel `bmm`), geqrf wins **small-batch/large-n + tiny-n**
(our sequential Python panel loop dominates with no batch to hide it).

**Profiled (torch.profiler) → blocked_wy is CPU-dispatch / launch-bound, not FLOP-bound.**
- b640 n512: ~288 ms/iter CPU dispatch vs ~67 ms/iter GPU (wall 114 ms); ~10k kernel
  launches/iter (`copy_` 4468×, `mul` 2975×, + sub_/pow/div/masked_fill_/norms — the
  per-element panel Householder ops). `bmm` is only **15.8%** of GPU time.
- b2 n4096: ~2.26 s/iter CPU dispatch, 36,668 `copy_` calls/iter. Python-loop bound.
- ⇒ The lever is **kernel COUNT**, not precision/FLOPs. Fix via (a) CUDA graphs to remove
  CPU dispatch, (b) a fused panel/whole-QR kernel to remove the GPU-side tiny-op bloat.

**TF32 tested → REJECTED.** `allow_tf32=True` breaks `band` (27.9) and `rowscale` (26.1)
correctness AND is slower on every shape (b640 n512: 114→159 ms). Inference: GEMMs are
NOT the bottleneck — blocked_wy is **launch-bound + panel-bound**. Precision is a dead end;
optimize launches + panel.

**Decided next steps (priority order):**
1. **Shape-based dispatch** — pick better of {geqrf, ours} from `data.shape`. Floor
   geomean ≈ **1.40× → ahead of baseline**, trivial `if`. Do this first.
2. **CUDA graphs** on the large-batch path — kill per-block Python/launch overhead
   (the actual bottleneck, per the TF32 result). Top perf lever now.
3. **Fused small-n kernel** (n≲230, one block/matrix) — turn n32/n176 from 1.0× to >1×.
4. Better panel (Triton/recursive) for large-n only if we choose to fight geqrf there
   (likely not worth it — just dispatch to geqrf for n2048/n4096).
5. Precision (TF32/BF16): only ever selectively, never global, never band/rowscale.

## Grader environment & numerical headroom (from a real popcorn-cli test run)

Official grader: `NVIDIA B200`, **Torch 2.12.0+cu130**, Linux, 1 GPU. `modal_qr.py`'s
image is pinned to cu130 to match. The `torch.geqrf` baseline passes 19/19 tests with
large margin: gates are `scaled_factor_residual < 20` and `scaled_orth_residual < 100`,
and geqrf sits at `scaled_factor ≈ 0.002–0.09`, `scaled_orth ≈ 0.1–1.1`. **Implication:
low-precision (TF32/BF16) internal strategies have lots of room** — inflating residuals
20–100× would likely still pass, especially at large n. Worth pursuing.

## Algorithm (current submission.py = blocked WY Householder)

Blocked WY-form (Schreiber–Van Loan): for each column block of width `_BLOCK`,
1. **panel** — `b` sequential Householder steps on the panel (O(b·n²), GEMV-bound);
2. **WY build** — `G = YᵀY` via one batched GEMM, then `T` via a `b`-step recurrence;
3. **trailing update** — `A -= Y @ (Tᵀ @ (Yᵀ@A))`, three batched GEMMs (tensor-core bound).

Output is `(H, tau)` in LAPACK convention; CPU-validated 26/26 (synthetic) + 10/10
(real checker, n≤512). `_BLOCK=64`, untuned.

## Known gotchas / open issues

- **⛔ The "work on another stream" rejection is a NAIVE SUBSTRING CHECK, not a real stream ban.**
  The grader API does `if "stream" in submission_code.lower(): reject(...)` (gpu-mode/kernelbot
  `api_utils.py`). It greps the source for the literal substring **"stream"** — in comments,
  docstrings, variable names, ANYTHING (incl. "upstream"/"downstream"). It does NOT analyze actual
  CUDA stream use. ⇒ **Just never write the substring "stream" in a submission.** Triton kernels and
  `torch.compile` are FINE (v5–v9 were all rejected only because the word "stream" appeared in their
  code/comments; v9 passed once scrubbed). The one thing genuinely unusable is **CUDA graphs**,
  because `torch.cuda.Stream` literally contains "stream" (and graphs also race the harness's
  canary check). Champion path: **v9 (Triton panel, ~2.84×)** once "stream"-free.
  NOTE: there's ALSO a runtime canary/shadow check (pygpubench manager.cpp) that detects kernels
  which truly run off the harness's `torch.cuda.current_stream()` — but normal Triton/torch inherit
  it, so that's not a concern for us.


- **TF32 / precision is a real lever.** The factor tolerance `20·n·eps32` is relative
  and grows with `n` (~1e-2 at n=4096), so TF32 trailing GEMMs *may* pass and would be a
  big speedup — but clustered/nearcollinear/nearrank cases fail first. Always note which
  TF32 setting a benchmark ran under. Returned factors must be FP32 regardless.
- **Triton variant (`archive/submission_triton.py`) is unvalidated and has real bugs:**
  - Loads the whole `(M_POW2, B)` panel into registers, one program per batch element —
    does **not** scale; the first block of n=4096 is a 4096×64 tile → register spill/fail.
    Only viable for small panel heights (roughly n ≤ 256).
  - `tl.arange(0, B)` requires `B` to be a power of two. With `_BLOCK=64`, n=176 yields a
    width-48 final block → errors. Any non-pow2 `_BLOCK` (e.g. 96) breaks every block.
    Fix: a `B_POW2` constexpr with masking, like the existing `M_POW2`.
  - `UPDATE_ONLY_PANEL` kwarg is dead code. The in-place `tl.store` race is a non-issue
    (one program per batch element, disjoint memory).
- **The real opponent is cuSOLVER (`torch.geqrf`).** Likely outcome: custom blocked-WY+bmm
  wins on large-batch/medium-n (the 512×640 headline) and loses on small-batch/huge-n
  (2048, 4096). Adaptive dispatch (geqrf for some shapes, custom for others) may be
  necessary, not optional — but costs "elegant"-prize points.
- **eval.py warmup** only runs the first (n=32) shape, so first-touch Triton compile for
  other shapes lands in the timed loop; amortized over ~200 repeats, so it's noise.

## Workflow

1. Edit `submission.py` (or add a new file and point `--submission` at it).
2. `modal run modal_qr.py --mode test` → confirm correctness on all task.yml tests.
3. `modal run modal_qr.py --mode bench` → per-shape ms + geomean vs geqrf.
4. Tune (e.g. `_BLOCK ∈ {32,64,96,128,256}`), record numbers per shape, keep winners.
5. When a candidate clearly beats geqrf on the geomean, validate with popcorn-cli
   `--mode test`, then `--mode benchmark`. Only run `--mode leaderboard` after asking.
