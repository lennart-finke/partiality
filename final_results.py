"""Finalize effect estimates (theta, two-sided p-values, two-sided CIs) for
the main own-company partiality test and the two sensitivity analyses, and
emit LaTeX tables analogous to ``tab:company_effects`` in ``paper/main.tex``.

This is a *thin driver* on top of the pre-registered modules
``hypothesis_testing.py`` and ``sensitivity_analyses.py`` -- those are left
untouched. Here we only

  * fix the run configuration (permutation counts, two-sided test, CIs),
  * rescale theta to the paper's Y units, and
  * write JSON + LaTeX to ``outputs/final-results/``.

Scaling.  ``hypothesis_testing``/``sensitivity_analyses`` normalize the
response as ``score / 10`` (DEFAULT_SCORE_MAX), whereas the paper defines the
favorability score as ``Y = (score - 1) / 4`` on the 1..5 Likert mean, so
Y in [0, 1] and theta in [-1, 1]. theta is a contrast, so the -1 shift cancels
and the two scales differ by the constant factor 10/4 = 2.5. We rescale here so
the reported theta lives in Y units (theta = 1 <=> maximally favorable to own
company and maximally unfavorable to others).

Permutations.  The main analysis uses s = 1,000,000; the sensitivity analyses
use s = 100,000 (fewer, per the spec). All tests are TWO-SIDED, and CIs invert
the two-sided test.

Usage:
    python final_results.py
    python final_results.py --main-permutations 1000000 \
        --sensitivity-permutations 100000 --seed 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hypothesis_testing import (
    confidence_interval,
    freedman_lane,
    load_csv,
)
from config import company_display
from sensitivity_analyses import run_cutoff_sensitivity, run_judge_bias

# score/10 -> (score-1)/4 ; theta is a contrast so the shift cancels.
THETA_SCALE = 10.0 / 4.0

OUT_DIR = Path("outputs/final-results")


# ---------------------------------------------------------------------------
# Rescale + serialize
# ---------------------------------------------------------------------------


def _rescale(results: list) -> list[dict]:
    """Return result dicts with theta_hat and CI bounds in Y units."""
    out = []
    for r in results:
        d = asdict(r)
        d["theta_hat"] *= THETA_SCALE
        for k in ("ci_lower", "ci_upper"):
            if d.get(k) is not None:
                d[k] *= THETA_SCALE
        out.append(d)
    return out


def _write_json(name: str, results: list[dict], meta: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": results}, fh, indent=2)
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# LaTeX table generation (analogous to tab:company_effects in main.tex)
# ---------------------------------------------------------------------------


def _fmt_theta(x: float | None) -> str:
    if x is None:
        return "--"
    return f"${x:+.3f}$"


def _sci(x: float) -> str:
    """LaTeX math-mode body: 2 decimals for x>=0.01, else 'm \\cdot 10^{e}'
    (trailing '.0' on the mantissa trimmed, e.g. '7 \\cdot 10^{-6}')."""
    import math

    if x <= 0:
        return "0"
    if x >= 0.01:
        return f"{x:.2f}"
    e = math.floor(math.log10(x))
    m = f"{x / 10.0**e:.1f}".removesuffix(".0")
    return f"{m} \\cdot 10^{{{e}}}"


def _fmt_p(p: float, n_perm: int) -> str:
    """Two-sided p, with the permutation floor 1/(s+1) reported as '< 10^-e'."""
    import math

    floor = 1.0 / (n_perm + 1)
    e = round(math.log10(n_perm))
    if p <= 1.5 * floor:
        return f"$< 10^{{-{e}}}$"
    return f"${_sci(p)}$"


def _fmt_holm(p: float, p_raw: float, n_perm: int) -> str:
    """Holm-corrected p; bold when significant at 0.05. When the raw p sits at
    the permutation floor, the Holm value m/(s+1) is itself an upper bound, so
    report it with '<' (e.g. '< 7 \\cdot 10^{-6}' for 7 companies at s=1e6)."""
    floor = 1.0 / (n_perm + 1)
    if p_raw <= 1.5 * floor:
        return f"$\\bf{{< {_sci(p)}}}$"
    if p >= 0.9995:
        return "$1$"
    return f"$\\bf{{{_sci(p)}}}$" if p < 0.05 else f"${_sci(p)}$"


def _company_table(results: list[dict], n_perm: int, caption: str, label: str) -> str:
    rows = []
    for r in sorted(results, key=lambda d: d["theta_hat"], reverse=True):
        rows.append(
            f"{company_display(r['company']):9s} & {_fmt_theta(r['ci_lower'])} & "
            f"{_fmt_theta(r['theta_hat'])} & {_fmt_theta(r['ci_upper'])} & "
            f"{_fmt_p(r['p_value'], n_perm)} & "
            f"{_fmt_holm(r['p_holm'], r['p_value'], n_perm)} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "\\textbf{Company} & \\multicolumn{3}{c}{\\bf{Partiality "
        "$\\mathbf \\theta_c$, 95\\% CI}} & \\textbf{$p$} & "
        "\\textbf{$p_{\\text{Holm}}$} \\\\\n"
        "\\cmidrule(lr){2-4}\n"
        " & Lower & \\textbf{$\\hat{\\theta}_c$} & Upper & & \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _write_table(name: str, tex: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(tex, encoding="utf-8")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def run_main(csv: Path, n_perm: int, seed: int) -> None:
    print(f"\n===== MAIN ANALYSIS (s={n_perm:,}, two-sided) =====")
    df = load_csv(csv)
    print(
        f"Loaded {len(df):,} rows, {df['company_id'].nunique()} companies, "
        f"{df['target_model_id'].nunique()} target models, "
        f"{df['story_id'].nunique()} stories, {df['judge_model_id'].nunique()} judges"
    )
    results, diag = freedman_lane(
        df, n_permutations=n_perm, seed=seed, alternative="two-sided"
    )
    for k, r in enumerate(results):
        r.ci_lower, r.ci_upper = confidence_interval(diag, k, alpha=0.05)
    scaled = _rescale(results)
    meta = {
        "analysis": "main",
        "n_permutations": n_perm,
        "seed": seed,
        "alternative": "two-sided",
        "ci_alpha": 0.05,
        "units": "Y = (score-1)/4 in [0,1]; theta in [-1,1] (rescaled by 10/4)",
        "block_cols": list(diag["block_cols"]),
    }
    _write_json("main_analysis.json", scaled, meta)
    _write_table(
        "main_table.tex",
        _company_table(
            scaled,
            n_perm,
            "Estimated model partiality $\\hat\\theta_c$ for each company (in "
            "favorability units $Y\\in[0,1]$), with non-simultaneous confidence "
            "intervals at $\\alpha=0.05$ and uncorrected as well as "
            "Holm--Bonferroni-corrected two-sided $p$-values.",
            "tab:company_effects",
        ),
    )


def run_sensitivities(csv: Path, n_perm: int, seed: int, cutoffs_path: Path) -> None:
    df = load_csv(csv)
    cutoffs = json.loads(cutoffs_path.read_text())
    cutoffs = {k: v for k, v in cutoffs.items() if not k.startswith("_")}

    print(f"\n===== KNOWLEDGE-CUTOFF SENSITIVITY (s={n_perm:,}, two-sided) =====")
    results, diag = run_cutoff_sensitivity(
        df,
        model_cutoffs=cutoffs,
        n_permutations=n_perm,
        seed=seed,
        alternative="two-sided",
        with_ci=True,
        alpha=0.05,
    )
    scaled = _rescale(results)
    meta = {
        "analysis": "sensitivity_cutoff",
        "n_permutations": n_perm,
        "seed": seed,
        "alternative": "two-sided",
        "ci_alpha": 0.05,
        "units": "Y = (score-1)/4 in [0,1]; theta in [-1,1] (rescaled by 10/4)",
        "block_cols": list(diag["block_cols"]),
        "note": "chi orthogonalized against the pre/post knowledge-cutoff indicator.",
    }
    _write_json("sensitivity_cutoff.json", scaled, meta)
    _write_table(
        "sensitivity_cutoff_table.tex",
        _company_table(
            scaled,
            n_perm,
            "Knowledge-cutoff sensitivity: estimated partiality $\\hat\\theta_c$ "
            "using the cutoff-orthogonalized contrast $\\chi^\\star$ (favorability "
            "units $Y\\in[0,1]$), with non-simultaneous 95\\% confidence intervals "
            "and uncorrected as well as Holm--Bonferroni-corrected two-sided "
            "$p$-values ($s=" + f"{n_perm:,}".replace(",", "{,}") + "$ permutations).",
            "tab:sensitivity_cutoff",
        ),
    )

    print(f"\n===== JUDGE-BIAS SENSITIVITY (s={n_perm:,}, two-sided) =====")
    results, diag = run_judge_bias(
        df,
        n_permutations=n_perm,
        seed=seed,
        alternative="two-sided",
        with_ci=True,
        alpha=0.05,
    )
    scaled = _rescale(results)
    meta = {
        "analysis": "sensitivity_judge",
        "n_permutations": n_perm,
        "seed": seed,
        "alternative": "two-sided",
        "ci_alpha": 0.05,
        "units": "Y = (score-1)/4 in [0,1]; theta in [-1,1] (rescaled by 10/4)",
        "block_cols": list(diag["block_cols"]),
        "note": "interaction taken over the judge model's company instead of the target's.",
    }
    _write_json("sensitivity_judge.json", scaled, meta)
    _write_table(
        "sensitivity_judge_table.tex",
        _company_table(
            scaled,
            n_perm,
            "Judge-bias sensitivity: estimated partiality $\\hat\\theta_{c(k)}$ of "
            "judge models toward their own company (favorability units "
            "$Y\\in[0,1]$), with non-simultaneous 95\\% confidence intervals and "
            "uncorrected as well as Holm--Bonferroni-corrected two-sided "
            "$p$-values ($s=" + f"{n_perm:,}".replace(",", "{,}") + "$ permutations).",
            "tab:sensitivity_judge",
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/scores/scores.csv"))
    ap.add_argument(
        "--model-cutoffs", type=Path, default=Path("data/model_cutoffs.json")
    )
    ap.add_argument("--main-permutations", type=int, default=1_000_000)
    ap.add_argument("--sensitivity-permutations", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-main", action="store_true")
    ap.add_argument("--skip-sensitivities", action="store_true")
    args = ap.parse_args()

    if not args.skip_main:
        run_main(args.csv, args.main_permutations, args.seed)
    if not args.skip_sensitivities:
        run_sensitivities(
            args.csv, args.sensitivity_permutations, args.seed, args.model_cutoffs
        )


if __name__ == "__main__":
    main()
