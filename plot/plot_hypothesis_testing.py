"""Forest plot of the own-company favoritism estimates.
Usage:
    python plot_hypothesis_testing.py
    python plot_hypothesis_testing.py --json outputs/final-results/main_analysis.json \\
        --out outputs/hypothesis_testing.png --alpha 0.05

Reads the finalized estimates from ``outputs/final-results/`` (see
``final_results.py``), whose theta is already in the paper's favorability units
Y in [0, 1]. A legacy flat-list JSON in score/10 units (e.g. the old
``outputs/hypothesis_testing.json``) is still accepted and rescaled by 10/4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Allow running as ``python plot/plot_hypothesis_testing.py`` from the repo root:
# the script's own directory is on sys.path, but config.py lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import company_display

# hypothesis_testing.py normalizes the response as score/10 (DEFAULT_SCORE_MAX),
# while the paper defines the partiality score as (score - 1)/4 on the 1..5
# Likert mean. theta is a contrast, so the shift cancels and the two scales
# differ by the constant factor 10/4. The final-results JSON is already in Y
# units; this factor is only applied to legacy score/10 inputs.
THETA_SCALE = 10.0 / 4.0


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "results" in data:
        # final-results format: theta already in Y units, no rescaling.
        return data["results"]
    # Legacy flat list in score/10 units -> rescale to Y units.
    for r in data:
        r["theta_hat"] *= THETA_SCALE
        for key in ("ci_lower", "ci_upper"):
            if r.get(key) is not None:
                r[key] *= THETA_SCALE
    return data


def plot(results: list[dict]) -> Figure:
    # Sort ascending so the strongest favoritism sits at the top.
    results = sorted(results, key=lambda r: r["theta_hat"])
    ys = list(range(len(results)))

    have_ci = all(r.get("ci_lower") is not None for r in results)

    # Permutation p-values are floored at 1/(B+1); the smallest observed
    # value is therefore an upper bound, so report it with "<" instead of "=".
    pvals = [r.get("p_holm", r["p_value"]) for r in results]
    p_min = min(pvals)

    fig, ax = plt.subplots(figsize=(6, 0.32 * len(results) + 1))
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=0)

    for y, r in zip(ys, results):
        t = r["theta_hat"]
        err = [[t - r["ci_lower"]], [r["ci_upper"] - t]] if have_ci else None
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
        p = r.get("p_holm", r["p_value"])
        rel = "<" if p <= p_min else "="
        # Use plain decimals when two digits suffice; otherwise scientific.
        p_str = f"{p:.2f}" if round(p, 2) >= 0.01 else f"{p:.1e}"
        ax.annotate(
            f"p{rel}{p_str}",
            (r["ci_upper"] if have_ci else t, y),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=7,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([company_display(r["company"]) for r in results])
    ax.set_xlabel(r"Partiality $\hat{\theta}_c$")
    ax.margins(y=0.12)
    # Widen the right side so the p-value labels stay inside the frame.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.12 * (x1 - x0))
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json", type=Path, default=Path("outputs/final-results/main_analysis.json")
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/hypothesis_testing.png"))
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
