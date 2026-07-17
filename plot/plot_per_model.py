"""Forest plot of the per-model own-company favoritism estimates.

Broken out of ``explore_analysis.py`` section 3. Reads the finalized per-model
estimates from ``outputs/final-results/per_model_analysis.json`` (written by
``per_model_analysis.py``), whose theta is already in the paper's favorability
units Y in [0, 1], so no rescaling is applied.

Usage:
    python plot_per_model.py
    python plot_per_model.py --json outputs/final-results/per_model_analysis.json \\
        --out outputs/hypothesis_testing_per_model.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Allow running as ``python plot/plot_per_model.py`` from the repo root: the
# script's own directory is on sys.path, but config.py lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import company_display, model_display


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["results"] if isinstance(data, dict) and "results" in data else data


def plot(results: list[dict]) -> Figure:
    # Sort ascending so the strongest favoritism sits at the top.
    results = sorted(results, key=lambda r: r["theta_hat"])
    ys = list(range(len(results)))

    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(results) + 1))
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=0)
    for y, r in zip(ys, results):
        t = r["theta_hat"]
        lo, hi = r.get("ci_lower"), r.get("ci_upper")
        err = (
            [[t - lo], [hi - t]]
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi)
            else None
        )
        ax.errorbar(
            t,
            y,
            xerr=err,
            fmt="o",
            color="black",
            markersize=5,
            capsize=3,
            elinewidth=1,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            f"{company_display(r['company'])} - {model_display(r['model'])}"
            for r in results
        ]
    )
    ax.set_xlabel(r"Partiality $\hat{\theta}_m$")
    ax.margins(y=0.04)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.14 * (x1 - x0))
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        type=Path,
        default=Path("outputs/final-results/per_model_analysis.json"),
    )
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/hypothesis_testing_per_model.png")
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    results = load_results(args.json)
    fig = plot(results)

    pdf_path = Path("outputs/pdf") / args.out.with_suffix(".pdf").name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.out, pdf_path):
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
