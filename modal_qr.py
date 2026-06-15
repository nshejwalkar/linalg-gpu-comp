"""
Modal launcher for private B200 iteration on the GPU MODE `qr` kernel.

Fast, private iteration loop on our own Modal account. Runs the real competition
harness (reference.py generate_input / check_implementation) against a chosen
submission file and reports:

  * correctness  — all task.yml `tests` through check_implementation (the hard gate)
  * benchmark    — per-shape time for our submission AND torch.geqrf, plus the
                   geometric-mean speedup (the leaderboard ranking metric)
  * profile      — torch.profiler per-kernel CUDA times (ncu/nsys aren't usable on
                   managed Modal; see research/profiling.md)

Does NOT consume a competition slot. popcorn-cli is the source of truth for official
numbers; timing here mirrors reference_repo/eval.py (CUDA events + clear_l2_cache).

Usage (from the `modal` conda env, inside qr_competition/):
  $env:PYTHONUTF8=1                                        # Windows: avoid cp1252 crash
  modal run modal_qr.py                                    # test + bench, submission.py
  modal run modal_qr.py --mode test
  modal run modal_qr.py --mode bench
  modal run modal_qr.py --mode baseline                    # time torch.geqrf only
  modal run modal_qr.py --mode profile                     # torch.profiler kernel table
  modal run modal_qr.py --submission archive/submission_blocked_wy.py
  modal run modal_qr.py --mode bench --tf32                # allow TF32 matmuls

NOTE: we pass file *contents* as function args rather than mounting the local dir.
On Windows, Modal's live directory-watch (and copy=True) trips "<file> was modified
during build process" when Defender/the search indexer touches a file during the
multi-second image build. Passing strings sidesteps the watcher entirely.
"""

import os
import modal

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Image: torch + triton for Blackwell (cu130 — matches the real grader, observed
# as torch 2.12.0+cu130 from a popcorn-cli run). Built once and cached; no local
# files baked in, so it stays stable across edits.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)

app = modal.App("qr-bench", image=image)

# Ranked benchmark shapes (task.yml `benchmarks`) — all dense.
BENCH_SHAPES = [
    {"batch": 20,  "n": 32,   "cond": 1, "seed": 43214},
    {"batch": 40,  "n": 176,  "cond": 1, "seed": 423011},
    {"batch": 40,  "n": 352,  "cond": 1, "seed": 123456},
    {"batch": 640, "n": 512,  "cond": 2, "seed": 1029},     # << headline case
    {"batch": 60,  "n": 1024, "cond": 2, "seed": 75342},
    {"batch": 8,   "n": 2048, "cond": 1, "seed": 224466},
    {"batch": 2,   "n": 4096, "cond": 1, "seed": 32412},
]

REMOTE = "/root/qr"
# Files shipped to the container (contents passed as args, written remotely).
HARNESS_FILES = [
    "reference_repo/reference.py",
    "reference_repo/task.py",
    "reference_repo/utils.py",
    "reference_repo/task.yml",
]


# ── Helpers that run INSIDE the container ────────────────────────────────────

def _materialize(files: dict):
    """Write the shipped file contents into /root/qr and put reference_repo on path."""
    import sys
    for rel, content in files.items():
        path = os.path.join(REMOTE, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    sys.path.insert(0, os.path.join(REMOTE, "reference_repo"))  # task/reference/utils


_SUB_MOD = None  # populated by _load_kernel; profile mode inspects it for Triton kernels


def _load_kernel():
    """Import custom_kernel from the active submission written at /root/qr."""
    global _SUB_MOD
    import importlib.util
    path = os.path.join(REMOTE, "submission_active.py")
    spec = importlib.util.spec_from_file_location("active_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SUB_MOD = mod
    return mod.custom_kernel


def _bench_fn(fn, A, repeats=50, warmup=8):
    """Time fn(A) with CUDA events + L2 flush, mirroring eval.py. Returns (mean_ms, best_ms)."""
    import torch
    from utils import clear_l2_cache

    for _ in range(warmup):          # absorbs Triton/cuBLAS compile + autotune
        fn(A)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        clear_l2_cache()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(A)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))  # ms
    times.sort()
    return sum(times) / len(times), times[0]


# ── Remote entrypoint ────────────────────────────────────────────────────────

@app.function(gpu="B200", timeout=1800)
def run(files: dict, mode: str = "all", tf32: bool = False):
    import math
    import torch

    _materialize(files)
    import yaml
    from reference import generate_input, check_implementation

    print(f"torch {torch.__version__} | device: {torch.cuda.get_device_name(0)}")
    assert torch.cuda.is_available(), "no CUDA in container"
    if "B200" not in torch.cuda.get_device_name(0):
        print("  WARNING: GPU is not a B200 — numbers won't match the leaderboard.")

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    print(f"mode: {mode} | tf32: {tf32}\n")

    kernel = _load_kernel()

    # ── Correctness: every task.yml `tests` case through the real checker ──────
    if mode in ("all", "test"):
        specs = yaml.safe_load(files["reference_repo/task.yml"])
        print("=" * 78)
        print("CORRECTNESS  (task.yml tests — hard gate)")
        print("=" * 78)
        npass = 0
        for t in specs["tests"]:
            case = t.get("case", "dense")
            A = generate_input(t["batch"], t["n"], t["cond"], t["seed"], case)
            try:
                out = kernel(A.clone())
                ok, msg = check_implementation(A, out)
            except Exception as exc:
                ok, msg = False, f"EXCEPTION: {exc}"
            npass += ok
            tag = "PASS" if ok else "FAIL"
            short = msg.split(";")[0] if ok else msg
            print(f"  {tag}  b={t['batch']:>3} n={t['n']:>4} cond={t['cond']} "
                  f"{case:<13} {short}")
        print(f"\n  {npass}/{len(specs['tests'])} passed\n")

    # ── Eval-fit probe: replicate eval.py's ranked loop (recheck every iter, 30s
    #    summed-kernel cap, err/mean<0.001 early-break) and report WHERE the wall goes.
    if mode == "evalfit":
        import math, time
        from utils import clear_l2_cache
        print("=" * 96)
        print("EVAL-FIT PROBE  (ranked loop: recheck EVERY iter; break at err/mean<0.001 | 30s summed | 120s)")
        print("=" * 96)
        print(f"  {'shape':<13}{'inputs':>7}{'iters':>7}{'break':>13}{'kern_s':>8}{'recheck_s':>10}{'wall_s':>8}{'CV%':>7}")
        TARGET = 256 * 1024 * 1024
        grand = 0.0
        for s in BENCH_SHAPES:
            b, n = s["batch"], s["n"]
            cnt = max(1, min(50, TARGET // (b * n * n * 4)))
            dl = [generate_input(b, n, s["cond"], s["seed"] + k, "dense") for k in range(cnt)]
            ref = [d.clone() for d in dl]
            for d in dl:
                kernel(d)
            torch.cuda.synchronize()
            durs = []
            recheck_s = 0.0
            start = time.perf_counter()
            brk = "1000cap"
            for i in range(1000):
                clear_l2_cache(); torch.cuda.synchronize()
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                outs = [kernel(d) for d in dl]
                e1.record(); torch.cuda.synchronize()
                durs.append(e0.elapsed_time(e1) * 1e6 / len(dl))   # ns/input
                r0 = time.perf_counter()
                for rd, o in zip(ref, outs):
                    check_implementation(rd, o)
                recheck_s += time.perf_counter() - r0
                total = time.perf_counter() - start
                if i > 1 and total > 0.1:
                    mean = sum(durs) / len(durs)
                    std = math.sqrt(sum((x - mean) ** 2 for x in durs) / (len(durs) - 1))
                    if std / math.sqrt(len(durs)) / mean < 0.001:
                        brk = "err<0.001"; break
                    if mean * len(durs) > 30e9:
                        brk = "30s-summed"; break
                    if total > 120:
                        brk = "120s-wall"; break
            wall = time.perf_counter() - start
            mean = sum(durs) / len(durs)
            std = math.sqrt(sum((x - mean) ** 2 for x in durs) / max(len(durs) - 1, 1))
            kern_s = mean * len(durs) * len(dl) / 1e9
            grand += wall
            print(f"  b{b}n{n:<9}{cnt:>7}{len(durs):>7}{brk:>13}{kern_s:>8.1f}{recheck_s:>10.1f}"
                  f"{wall:>8.1f}{std/mean*100:>7.1f}")
        print(f"\n  RANKED-phase wall total: {grand:.1f}s  (real eval adds test+benchmark phases + ~2s compile)")
        return

    # ── Compile-time probe: isolate kernel compile (first call) from run (steady) ──
    if mode == "compiletime":
        import time
        print("=" * 78)
        print("COMPILE-TIME PROBE  (first call = compile+run; steady = cached run)")
        print("=" * 78)
        for s in [{"batch": 640, "n": 512, "cond": 2, "seed": 1029},
                  {"batch": 60, "n": 1024, "cond": 2, "seed": 75342}]:
            A = generate_input(s["batch"], s["n"], s["cond"], s["seed"], "dense")
            torch.cuda.synchronize(); t0 = time.time()
            kernel(A)                                   # FIRST: triggers JIT compile
            torch.cuda.synchronize(); first = time.time() - t0
            for _ in range(3):
                kernel(A)
            torch.cuda.synchronize(); t2 = time.time()
            for _ in range(5):
                kernel(A)
            torch.cuda.synchronize(); run = (time.time() - t2) / 5
            g0 = time.time(); torch.geqrf(A); torch.cuda.synchronize(); geqrf_first = time.time() - g0
            print(f"  n={s['n']:>4} b={s['batch']:>3}: first(compile+run)={first:7.2f}s | "
                  f"steady run={run*1000:7.1f}ms | COMPILE≈{first - run:6.2f}s | "
                  f"geqrf 1st={geqrf_first*1000:.1f}ms")
        return

    # ── Profile: torch.profiler (Kineto/CUPTI) — works on Modal w/o ncu perms ──
    if mode == "profile":
        # No-root profiling stack (see research/profiling_and_nodes.md): per-kernel
        # CUDA self-time + FLOPs (panel-vs-bmm split), do_bench median, and Triton
        # kernel resource attrs (regs/spills/smem/occupancy). Default shapes are the
        # ones our kernel actually accelerates so the split is meaningful.
        from torch.profiler import profile, ProfilerActivity
        prof_shapes = [
            {"batch": 640, "n": 512,  "cond": 2, "seed": 1029},
            {"batch": 60,  "n": 1024, "cond": 2, "seed": 75342},
        ]
        for s in prof_shapes:
            A = generate_input(s["batch"], s["n"], s["cond"], s["seed"], "dense")
            for _ in range(6):                       # warm up: compile + autotune
                kernel(A)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True) as prof:
                for _ in range(10):
                    kernel(A)
                torch.cuda.synchronize()
            print("=" * 78)
            print(f"PROFILE  b={s['batch']} n={s['n']}  (10 iters; self CUDA time + FLOPs)")
            print("=" * 78)
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18))
            try:
                import triton.testing as tt
                ms = tt.do_bench(lambda: kernel(A), warmup=200, rep=400, return_mode="median")
                print(f"  do_bench median: {ms:.3f} ms")
            except Exception as ex:
                print(f"  do_bench failed: {ex}")
            print()

        # Triton kernel resource usage (free, no-root): regs / spills / smem / occupancy.
        try:
            import triton
            print("=" * 78); print("TRITON KERNEL RESOURCES (regs/spills/smem -> occupancy)")
            print("=" * 78)
            found = False
            for name, obj in vars(_SUB_MOD).items():
                if isinstance(obj, triton.runtime.JITFunction):
                    found = True
                    cks = []
                    cache = getattr(obj, "cache", {})
                    for d in (cache.values() if isinstance(cache, dict) else []):
                        cks.extend(d.values() if isinstance(d, dict) else [])
                    print(f"  {name}: {len(cks)} compiled variant(s)")
                    for ck in cks:
                        nr = getattr(ck, "n_regs", None)
                        ns = getattr(ck, "n_spills", None)
                        md = getattr(ck, "metadata", None)
                        sm = getattr(md, "shared", None)
                        nw = getattr(md, "num_warps", None)
                        occ = ""
                        try:
                            occ_reg = 65536 // (nr * 32 * nw)
                            occ_smem = (228 * 1024) // sm if sm else 64
                            occ = f"  ~{min(occ_reg, occ_smem, 64 // nw)} blk/SM"
                        except Exception:
                            pass
                        flag = "  <-- SPILLS!" if ns else ""
                        print(f"    regs/thd={nr} spills={ns} smem={sm}B warps={nw}{occ}{flag}")
            if not found:
                print("  (no Triton JITFunctions in submission — pure torch)")
        except Exception as ex:
            print(f"  triton attr dump failed: {ex}")
        return

    # ── Benchmark: our kernel vs torch.geqrf, geomean speedup ─────────────────
    if mode in ("all", "bench", "baseline"):
        print("=" * 78)
        print("BENCHMARK  (per-shape mean ms; geomean = ranking metric)")
        print("=" * 78)
        print(f"  {'shape':<28}{'geqrf ms':>11}{'ours ms':>11}{'speedup':>10}")
        ratios = []
        for s in BENCH_SHAPES:
            A = generate_input(s["batch"], s["n"], s["cond"], s["seed"], "dense")
            g_mean, _ = _bench_fn(torch.geqrf, A)
            label = f"b={s['batch']} n={s['n']}"
            if mode == "baseline":
                print(f"  {label:<28}{g_mean:>11.3f}")
                continue
            o_mean, _ = _bench_fn(kernel, A)
            sp = g_mean / o_mean if o_mean > 0 else float("nan")
            ratios.append(sp)
            print(f"  {label:<28}{g_mean:>11.3f}{o_mean:>11.3f}{sp:>9.2f}x")
        if ratios:
            geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            print(f"\n  GEOMEAN speedup vs torch.geqrf: {geo:.3f}x  "
                  f"({'ahead' if geo > 1 else 'behind'})\n")


@app.local_entrypoint()
def main(submission: str = "submission.py", mode: str = "all", tf32: bool = False):
    def _read(rel):
        with open(os.path.join(LOCAL_DIR, rel), "r", encoding="utf-8") as f:
            return f.read()

    files = {rel: _read(rel) for rel in HARNESS_FILES}
    files["submission_active.py"] = _read(submission)
    print(f"shipping {submission} + harness to B200 (mode={mode}, tf32={tf32})")
    run.remote(files=files, mode=mode, tf32=tf32)
