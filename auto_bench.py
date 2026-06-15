#!/usr/bin/env python
"""
Standalone deterministic benchmark driver — NEEDS NO CLAUDE.

Runs entirely on your machine + Modal, so it keeps making progress even while
Claude is rate-limited (the 5-hour window). It benchmarks candidate submissions
on a B200, parses correctness + geomean, logs/ranks them, and (optionally) sweeps
block sizes. It does the DETERMINISTIC part of the job (validate/tune/track) that
doesn't need model reasoning. It does NOT write new kernels and does NOT submit to
the leaderboard (do that manually with popcorn-cli once you've reviewed results).

USAGE (from this folder, with the modal env):
    conda activate modal
    python auto_bench.py                      # benchmark every submissions/*.py not yet logged
    python auto_bench.py v9_triton.py v8_compile.py   # specific files
    python auto_bench.py --force              # re-benchmark all
    python auto_bench.py --sweep v1.py        # _BLOCK in {32,64,96,128,256} variants of v1.py
    python auto_bench.py --rank               # just print the current ranking from the log

Schedule it (Windows) to run unattended, e.g. start in 10 min:
    schtasks /create /tn qr_autobench /sc once /st <HH:MM> ^
      /tr "cmd /c conda activate modal && cd C:\\Users\\Neel\\modal\\qr_competition && python auto_bench.py"
Results accumulate in auto_bench_log.json + auto_bench_log.md regardless.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUBDIR = os.path.join(HERE, "submissions")
LOG_JSON = os.path.join(HERE, "auto_bench_log.json")
LOG_MD = os.path.join(HERE, "auto_bench_log.md")
TIMEOUT_S = 1800  # per benchmark run


def _load_log():
    if os.path.exists(LOG_JSON):
        with open(LOG_JSON) as f:
            return json.load(f)
    return []


def _save_log(rows):
    with open(LOG_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    # human-readable mirror, ranked by geomean (correct runs first)
    ranked = sorted(
        rows,
        key=lambda r: (-(r["passed"] == 19), -(r["geomean"] or 0)),
    )
    lines = ["# auto_bench log (ranked; needs 19/19 to count)\n",
             "| file | passed | geomean speedup | when |", "|---|---|---|---|"]
    for r in ranked:
        g = f'{r["geomean"]:.3f}x' if r["geomean"] is not None else "—"
        lines.append(f'| {r["file"]} | {r["passed"]}/19 | {g} | {r["ts"]} |')
    with open(LOG_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _bench(rel):
    """Run the Modal harness for submissions/<rel>, parse correctness + geomean."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = ["modal", "run", "modal_qr.py", "--submission", f"submissions/{rel}", "--mode", "all"]
    print(f"[auto_bench] running {rel} ...", flush=True)
    try:
        p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True,
                           text=True, timeout=TIMEOUT_S)
        out = p.stdout + "\n" + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT", -1
    passed = re.search(r"(\d+)/19 passed", out)
    geo = re.search(r"GEOMEAN speedup vs torch\.geqrf: ([\d.]+)x", out)
    shapes = re.findall(r"b=\d+ n=\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)x", out)
    row = {
        "file": rel,
        "passed": int(passed.group(1)) if passed else 0,
        "geomean": float(geo.group(1)) if geo else None,
        "rc": rc,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[auto_bench] {rel}: {row['passed']}/19  geomean="
          f"{row['geomean']}x  rc={rc}", flush=True)
    return row


def _make_sweep(rel):
    """Write _BLOCK variants of submissions/<rel> into submissions/, return names."""
    src = os.path.join(SUBDIR, rel)
    with open(src) as f:
        code = f.read()
    base = rel[:-3]
    names = []
    for blk in (32, 64, 96, 128, 256):
        variant = re.sub(r"_BLOCK\s*=\s*\d+", f"_BLOCK = {blk}", code, count=1)
        name = f"{base}_blk{blk}.py"
        with open(os.path.join(SUBDIR, name), "w") as f:
            f.write(variant)
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="specific submission files (basename under submissions/)")
    ap.add_argument("--force", action="store_true", help="re-benchmark even if already logged")
    ap.add_argument("--sweep", metavar="FILE", help="generate _BLOCK variants of FILE and benchmark")
    ap.add_argument("--rank", action="store_true", help="just print the ranking from the log")
    args = ap.parse_args()

    rows = _load_log()
    done = {r["file"] for r in rows}

    if args.rank:
        for r in sorted(rows, key=lambda r: (-(r["passed"] == 19), -(r["geomean"] or 0))):
            print(f'{r["file"]:28} {r["passed"]}/19  '
                  f'{(str(round(r["geomean"],3))+"x") if r["geomean"] else "—":>9}')
        return

    if args.sweep:
        targets = _make_sweep(args.sweep)
    elif args.files:
        targets = args.files
    else:
        targets = sorted(os.path.basename(p) for p in
                         __import__("glob").glob(os.path.join(SUBDIR, "*.py")))

    for rel in targets:
        if rel in done and not args.force:
            print(f"[auto_bench] skip {rel} (already logged; --force to redo)")
            continue
        row = _bench(rel)
        rows = [r for r in rows if r["file"] != rel] + [row]
        _save_log(rows)  # checkpoint after every run (crash-safe)

    print(f"\n[auto_bench] done. Ranking (see {os.path.basename(LOG_MD)}):")
    for r in sorted(rows, key=lambda r: (-(r["passed"] == 19), -(r["geomean"] or 0)))[:10]:
        g = f'{r["geomean"]:.3f}x' if r["geomean"] is not None else "—"
        print(f'  {r["file"]:28} {r["passed"]}/19  {g:>9}')


if __name__ == "__main__":
    main()
