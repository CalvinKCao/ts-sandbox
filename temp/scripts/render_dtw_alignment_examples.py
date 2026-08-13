#!/usr/bin/env python3
"""Render seeded example DTW alignments for binary fixed-control and MMPD packs."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = Path(
    "/home/cao/.codex/visualizations/2026/08/10/"
    "019fed43-fab5-7130-8efa-6b17549be388/dtw-warp-alignments.html"
)
SEED = 20260812
SPECS = {
    "traffic": {
        "Binary": REPO / "temp/fixed_control_stats_packs/staged_dpmpp_samples_traffic.npz",
        "MMPD": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_traffic.npz",
    },
    "exchange_rate": {
        "Binary": REPO / "temp/fixed_control_stats_packs/staged_dpmpp_samples_exchange_rate.npz",
        "MMPD": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_exchange_rate.npz",
    },
    "PeMS": {
        "Binary": REPO / "temp/fixed_control_stats_packs/staged_dpmpp_samples_PeMS.npz",
        "MMPD": REPO / "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_PeMS.npz",
    },
}


def dtw_path(target: np.ndarray, pred: np.ndarray, radius: int = 3) -> tuple[float, list[tuple[int, int]]]:
    """Unnormalized L1 DTW and its band-constrained alignment."""
    n = len(target)
    costs = np.full((n + 1, n + 1), np.inf, dtype=float)
    costs[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(n, i + radius) + 1):
            costs[i, j] = abs(target[i - 1] - pred[j - 1]) + min(
                costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1]
            )

    i = j = n
    path: list[tuple[int, int]] = []
    while i or j:
        if not i or not j:
            raise RuntimeError("DTW path left the feasible band")
        path.append((i - 1, j - 1))
        choices = (costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1])
        move = int(np.argmin(choices))
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return float(costs[n, n]), path[::-1]


def pxy(values: np.ndarray, x0: float, y0: float, width: float, height: float, lo: float, hi: float) -> str:
    if hi - lo < 1e-12:
        hi = lo + 1.0
    return " ".join(
        f"{x0 + width * i / max(len(values) - 1, 1):.2f},{y0 + height * (1 - (v - lo) / (hi - lo)):.2f}"
        for i, v in enumerate(values)
    )


def example_svg(target: np.ndarray, pred: np.ndarray, path: list[tuple[int, int]], loss: float, label: str) -> str:
    """A two-lane curve plot with light alignment threads behind the series."""
    outer_w, outer_h = 138, 133
    x, width = 21, 107
    top_y, lane_h, bottom_y = 25, 35, 79
    lo, hi = float(min(target.min(), pred.min())), float(max(target.max(), pred.max()))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    top = lambda i, v: (x + width * i / (len(target) - 1), top_y + lane_h * (1 - (v - lo) / (hi - lo)))
    bottom = lambda i, v: (x + width * i / (len(pred) - 1), bottom_y + lane_h * (1 - (v - lo) / (hi - lo)))
    threads = "".join(
        f'<line class="thread" x1="{top(i, target[i])[0]:.2f}" y1="{top(i, target[i])[1]:.2f}" '
        f'x2="{bottom(j, pred[j])[0]:.2f}" y2="{bottom(j, pred[j])[1]:.2f}"/>'
        for i, j in path
    )
    return f'''<svg class="example" viewBox="0 0 {outer_w} {outer_h}" role="img" aria-label="{escape(label)}, DTW loss {loss:.3f}">
  <text class="mini-label" x="4" y="11">{escape(label)}</text>
  <text class="mini-loss" x="4" y="126">DTW {loss:.3f}</text>
  <text class="lane-label" x="4" y="{top_y + lane_h / 2:.1f}">GT</text>
  <text class="lane-label" x="4" y="{bottom_y + lane_h / 2:.1f}">P</text>
  <line class="baseline" x1="{x}" y1="{top_y + lane_h / 2}" x2="{x + width}" y2="{top_y + lane_h / 2}"/>
  <line class="baseline" x1="{x}" y1="{bottom_y + lane_h / 2}" x2="{x + width}" y2="{bottom_y + lane_h / 2}"/>
  {threads}
  <polyline class="gt" points="{pxy(target, x, top_y, width, lane_h, lo, hi)}"/>
  <polyline class="pred" points="{pxy(pred, x, bottom_y, width, lane_h, lo, hi)}"/>
</svg>'''


def make_examples(y_true: np.ndarray, samples: np.ndarray, dataset: str, model: str, length: int) -> tuple[str, float]:
    # Draw a whole saved trajectory, then a random subwindow; both paths are de-meaned
    # exactly as in the reported DTW metric.
    name_code = sum(ord(c) for c in f"{dataset}:{model}:{length}")
    rng = np.random.default_rng(SEED + name_code)
    panels = []
    losses = []
    n_windows, n_variates, n_samples, horizon = samples.shape
    for _ in range(5):
        wi = int(rng.integers(n_windows))
        vi = int(rng.integers(n_variates))
        si = int(rng.integers(n_samples))
        start = int(rng.integers(horizon - length + 1))
        target = y_true[wi, vi, start:start + length].astype(float)
        pred = samples[wi, vi, si, start:start + length].astype(float)
        target -= target.mean()
        pred -= pred.mean()
        loss, path = dtw_path(target, pred)
        losses.append(loss)
        label = f"w{wi} · v{vi} · s{si} · t{start}"
        panels.append(example_svg(target, pred, path, loss, label))
    return "".join(panels), float(np.mean(losses))


def main() -> None:
    sections = []
    for dataset, models in SPECS.items():
        model_blocks = []
        for model, path in models.items():
            with np.load(path, allow_pickle=False) as pack:
                y_true = np.asarray(pack["y_true"])
                samples = np.asarray(pack["samples"])
            rows = []
            for length in (8, 16):
                examples, mean_loss = make_examples(y_true, samples, dataset, model, length)
                rows.append(f'''<div class="length-row">
  <div class="length-heading">L={length} <span>mean {mean_loss:.3f}</span></div>
  <div class="example-grid">{examples}</div>
</div>''')
            model_blocks.append(f'''<section class="model-block" aria-label="{dataset} {model}">
  <h3>{escape(model)}</h3>
  {''.join(rows)}
</section>''')
        sections.append(f'''<section class="dataset-block">
  <h2>{escape(dataset)}</h2>
  {''.join(model_blocks)}
</section>''')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f'''<div id="dtw-warp-alignments" class="dtw-viz">
<style>
  #dtw-warp-alignments {{ color: var(--foreground); font-size: var(--font-size-base); }}
  #dtw-warp-alignments .dataset-block {{ border-top: 1px solid var(--border); margin-top: 1rem; padding-top: .65rem; }}
  #dtw-warp-alignments .dataset-block:first-child {{ border-top: 0; margin-top: 0; padding-top: 0; }}
  #dtw-warp-alignments h2, #dtw-warp-alignments h3 {{ margin: 0; font-weight: 500; }}
  #dtw-warp-alignments h2 {{ margin-bottom: .45rem; }}
  #dtw-warp-alignments h3 {{ margin: .75rem 0 .25rem; }}
  #dtw-warp-alignments .length-row {{ display: grid; grid-template-columns: 4.7rem minmax(0, 1fr); gap: .35rem; align-items: center; margin: .2rem 0; }}
  #dtw-warp-alignments .length-heading {{ font-weight: 500; }}
  #dtw-warp-alignments .length-heading span {{ display: block; color: var(--muted-foreground); font-size: .86em; font-weight: 400; }}
  #dtw-warp-alignments .example-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .15rem; }}
  #dtw-warp-alignments .example {{ display: block; width: 100%; height: auto; }}
  #dtw-warp-alignments .thread {{ stroke: var(--muted-foreground); stroke-opacity: .26; stroke-width: .65; }}
  #dtw-warp-alignments .baseline {{ stroke: var(--border); stroke-width: .65; }}
  #dtw-warp-alignments .gt {{ fill: none; stroke: var(--viz-series-1); stroke-width: 1.55; stroke-linejoin: round; stroke-linecap: round; }}
  #dtw-warp-alignments .pred {{ fill: none; stroke: var(--viz-series-2); stroke-width: 1.55; stroke-linejoin: round; stroke-linecap: round; }}
  #dtw-warp-alignments text {{ fill: var(--foreground); font-family: inherit; }}
  #dtw-warp-alignments .mini-label {{ font-size: 7px; }}
  #dtw-warp-alignments .mini-loss {{ font-size: 8px; font-weight: 500; }}
  #dtw-warp-alignments .lane-label {{ fill: var(--muted-foreground); font-size: 7px; }}
  @media (max-width: 520px) {{
    #dtw-warp-alignments .length-row {{ grid-template-columns: 1fr; gap: .1rem; margin: .55rem 0; }}
    #dtw-warp-alignments .length-heading span {{ display: inline; margin-left: .35rem; }}
    #dtw-warp-alignments .example-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  }}
</style>
{''.join(sections)}
</div>
''')
    print(OUT)


if __name__ == "__main__":
    main()
