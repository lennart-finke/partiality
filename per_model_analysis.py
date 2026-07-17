"""Per-model own-company partiality (Freedman-Lane), broken out of
``explore_analysis.py`` section 3 so it can be finalized at full permutation
count independently of the rest of the exploratory script.

This reruns the main analysis but per target *model* instead of per company:
in the linear model the contrast becomes

    chi(i, j) = +1/2 if model i is from the company story j is about, else -1/2,

and theta_{c(i)} becomes theta_i. All models are fit jointly, exactly as in
the pre-registered per-company test. We use s = 100,000 permutations (fewer
than the main analysis) and a TWO-SIDED test; CIs invert the two-sided test.

The response is already the paper's favorability score Y = (score - 1) / 4 in
[0, 1] (read from the enriched per-sample data), so theta is in Y units and
needs no rescaling -- theta = 1 <=> maximally favorable to own company and
maximally unfavorable to others.

Usage:
    python per_model_analysis.py
    python per_model_analysis.py --n-permutations 100000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import company_display, model_display
from hypothesis_testing import confidence_interval, freedman_lane

FE = "target_model_id + story_id + judge_model_id + prompt_template_id"
OUT_DIR = Path("outputs/final-results")


def load_valid(parquet: Path) -> pd.DataFrame:
    """Enriched, valid rows with the FE columns and Y = (score-1)/4."""
    e = pd.read_parquet(parquet)
    e = e.rename(
        columns={
            "target_company": "company_id",
            "story_company": "story_company_id",
            "target_model": "target_model_id",
            "judge_model": "judge_model_id",
            "template_id": "prompt_template_id",
        }
    )
    valid = e[
        (e.language_score.between(1, 5))
        & (e.content_score.between(1, 5))
        & (e.completeness_score.between(1, 5))
    ].copy()
    for c in [
        "story_id",
        "judge_model_id",
        "target_model_id",
        "prompt_template_id",
        "company_id",
        "story_company_id",
    ]:
        valid[c] = valid[c].astype(str)
    # Company-level ids for the permutation stratum (matches hypothesis_testing).
    valid["target_company_id"] = valid["company_id"]
    valid["judge_company_id"] = valid["judge_model_id"].str.split("/").str[0]
    valid["y"] = ((valid["score"] - 1) / 4).clip(0, 1)
    return valid


def build_model_chi(valid: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict, dict]:
    """chi__<safe(m)> = I(target_model=m) * (+1/2 if story about company(m) else -1/2)."""
    model_company = valid.groupby("target_model_id").company_id.first()
    models = sorted(model_company.index)
    safe = {m: re.sub(r"\W", "_", m) for m in models}
    dfm = valid.copy()
    model_cols = []
    for m in models:
        chi = np.where(dfm.story_company_id == model_company[m], 0.5, -0.5)
        dfm[f"chi__{safe[m]}"] = np.where(dfm.target_model_id == m, chi, 0.0)
        model_cols.append(f"chi__{safe[m]}")
    return dfm, model_cols, dict(model_company), safe


def _fmt(x: float | None) -> str:
    return "--" if x is None or not np.isfinite(x) else f"${x:+.3f}$"


def make_table(rows: list[dict], n_perm: int) -> str:
    body = []
    for r in sorted(rows, key=lambda d: d["theta_hat"], reverse=True):
        body.append(
            f"{model_display(r['model'])} & {company_display(r['company'])} & "
            f"{_fmt(r['ci_lower'])} & {_fmt(r['theta_hat'])} & "
            f"{_fmt(r['ci_upper'])} \\\\"
        )
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Per-model own-company partiality $\\hat\\theta_i$ (favorability "
        "units $Y\\in[0,1]$), with non-simultaneous 95\\% confidence intervals "
        "($s=" + f"{n_perm:,}".replace(",", "{,}") + "$ permutations).}\n"
        "\\label{tab:per_model_effects}\n"
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\n"
        "\\textbf{Model} & \\textbf{Company} & "
        "\\multicolumn{3}{c}{\\bf{Partiality $\\mathbf \\theta_i$, 95\\% CI}} \\\\\n"
        "\\cmidrule(lr){3-5}\n"
        " & & Lower & \\textbf{$\\hat{\\theta}_i$} & Upper \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet", type=Path, default=Path("outputs/enriched_scores.parquet")
    )
    ap.add_argument("--n-permutations", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    valid = load_valid(args.parquet)
    print(
        f"valid rows: {len(valid):,}  companies: {valid.company_id.nunique()}  "
        f"models: {valid.target_model_id.nunique()}  "
        f"judges: {valid.judge_model_id.nunique()}"
    )

    dfm, model_cols, model_company, safe = build_model_chi(valid)
    model_n = valid.groupby("target_model_id").size().to_dict()
    inv_safe = {s: m for m, s in safe.items()}

    print(
        f"\n=== Per-model own-company theta, Freedman-Lane "
        f"(s={args.n_permutations:,}, two-sided, Y units) ==="
    )
    results, diag = freedman_lane(
        dfm,
        chi_cols=model_cols,
        n_permutations=args.n_permutations,
        seed=args.seed,
        alternative="two-sided",
    )
    by_safe = {r.company: r for r in results}
    ci_by_safe = {}
    for i, c in enumerate(diag["companies"]):
        ci_by_safe[c] = confidence_interval(diag, i, alpha=0.05)

    rows = []
    for s, r in by_safe.items():
        m = inv_safe[s]
        lo, hi = ci_by_safe.get(s, (np.nan, np.nan))
        rows.append(
            {
                "model": m,
                "company": model_company[m],
                "theta_hat": r.theta_hat,  # already Y units
                "ci_lower": lo,
                "ci_upper": hi,
                "p_value": r.p_value,
                "p_holm": r.p_holm,
                "n": int(model_n[m]),
            }
        )
    rows.sort(key=lambda d: d["theta_hat"], reverse=True)

    hdr = f"{'model':32s} {'company':10s} {'theta':>9s} {'p':>8s} {'p_holm':>8s}"
    print(hdr)
    for r in rows:
        print(
            f"{r['model']:32s} {r['company']:10s} {r['theta_hat']:+9.4f} "
            f"{r['p_value']:8.4f} {r['p_holm']:8.4f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "analysis": "per_model",
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "alternative": "two-sided",
        "ci_alpha": 0.05,
        "units": "Y = (score-1)/4 in [0,1]; theta in [-1,1] (already in Y units, no rescaling)",
        "block_cols": list(diag["block_cols"]),
    }
    json_path = OUT_DIR / "per_model_analysis.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": rows}, fh, indent=2)
    print(f"\nWrote {json_path}")

    tex_path = OUT_DIR / "per_model_table.tex"
    tex_path.write_text(make_table(rows, args.n_permutations), encoding="utf-8")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
