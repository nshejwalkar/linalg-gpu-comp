"""
Render progress.svg from bench_history.json — geomean speedup vs torch.geqrf
over our submission versions. Pure Python (no deps); run from qr_competition/:

    python plot_progress.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "bench_history.json")
OUT = os.path.join(HERE, "progress.svg")

W, H = 1000, 420
ML, MR, MT, MB = 70, 30, 30, 60          # margins
PW, PH = W - ML - MR, H - MT - MB        # plot area

BG = "#0d0d0d"
FG = "#e0e0e0"
GRID = "#262626"
DOT = "#34d399"      # green
LINE = "#f59e0b"     # orange
BASE = "#666666"


def main():
    with open(DATA) as f:
        hist = json.load(f)

    ys = [h["geomean_speedup"] for h in hist]
    labels = [h["version"] for h in hist]
    n = len(hist)
    ymax = max(max(ys), 1.0) * 1.15
    ymin = 0.0

    def px(i):
        return ML + (PW * i / max(n - 1, 1))

    def py(v):
        return MT + PH * (1 - (v - ymin) / (ymax - ymin))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="monospace">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<text x="{ML}" y="20" fill="{FG}" font-size="15">QR leaderboard: geomean speedup vs torch.geqrf</text>',
    ]

    # y gridlines + labels
    nticks = 5
    for t in range(nticks + 1):
        v = ymax * t / nticks
        y = py(v)
        parts.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ML-8}" y="{y+4:.1f}" fill="{FG}" font-size="12" text-anchor="end">{v:.2f}x</text>')

    # baseline at 1.0x
    yb = py(1.0)
    parts.append(f'<line x1="{ML}" y1="{yb:.1f}" x2="{ML+PW}" y2="{yb:.1f}" stroke="{BASE}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{ML+PW}" y="{yb-6:.1f}" fill="{BASE}" font-size="11" text-anchor="end">geqrf baseline (1.0x)</text>')

    # connecting line
    pts = " ".join(f"{px(i):.1f},{py(ys[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{LINE}" stroke-width="2"/>')

    # dots + version labels + value labels
    for i in range(n):
        x, y = px(i), py(ys[i])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{DOT}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-12:.1f}" fill="{DOT}" font-size="11" text-anchor="middle">{ys[i]:.2f}x</text>')
        parts.append(f'<text x="{x:.1f}" y="{MT+PH+18:.1f}" fill="{FG}" font-size="11" text-anchor="middle">{labels[i]}</text>')

    parts.append("</svg>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {OUT}  ({n} points, max {max(ys):.2f}x)")


if __name__ == "__main__":
    main()
