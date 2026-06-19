"""Sensitivity analyses for the own-company partiality test.

Two follow-ups to the primary ``hypothesis_testing.py`` experiment:

1. **Knowledge-cutoff adjustment.** The contrast ``chi(c, j)`` may be
   confounded with whether the story post-dates the target model's
   knowledge cutoff. We replace ``chi`` with its company-wise
   orthogonalization against the pre/post-cutoff indicator
   ``chi^(2)(i, j)``:

        chi*(., j) := chi - <chi(., j), chi^(2)(., j)>
                                / <chi^(2)(., j), chi^(2)(., j)>
                          * chi^(2)(., j)

   and re-run the Freedman-Lane test with ``chi*`` in place of ``chi``.

2. **Judge bias.** Symmetrically, judge models may favor their own
   company. We re-run the Freedman-Lane test with the interaction term
   ``theta_{c(k)} * chi(c(k), j)``, i.e. the indicator is taken over the
   *judge* model's company rather than the target's.

In both experiments p-values are Holm-Bonferroni corrected *within the
experiment*. No correction is applied across the three experiments
(primary + two sensitivities), per the spec.

Usage:
    python sensitivity_analyses.py --csv data/scores/scores.csv \
        --out outputs/ --model-cutoffs data/model_cutoffs.json
    python sensitivity_analyses.py --synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from hypothesis_testing import (
    DEFAULT_SCORE_MAX,
    TestResult,
    _prepare,
    build_chi_columns,
    confidence_interval,
    freedman_lane,
    load_csv,
)


# ---------------------------------------------------------------------------
# Knowledge-cutoff contrast: orthogonalize chi against the pre/post-cutoff
# indicator chi^(2).
# ---------------------------------------------------------------------------


def build_chi_star_columns(
    df: pd.DataFrame,
    story_dates: dict[str, str] | None = None,
    model_cutoffs: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return df with chi^star columns: chi orthogonalized against
    chi^(2) (the pre/post-knowledge-cutoff indicator) per company.

    chi^(2)(i, j) = +1 if story j's date > target_model i's cutoff,
                    -1 if <= cutoff. Mean-centered within each company
    so it carries no constant offset that would clash with the FEs.
    """
    df, chi_cols = build_chi_columns(df)
    if story_dates is None:
        story_dates = (
            df.dropna(subset=["story_date"])
            .drop_duplicates("story_id")
            .set_index("story_id")["story_date"]
            .to_dict()
        )
    model_cutoffs = model_cutoffs or {}
    sd = df["story_id"].map(story_dates)
    mc = df["target_model_id"].map(model_cutoffs)
    after = np.where(
        sd.notna() & mc.notna(),
        np.where(sd.astype(str) > mc.astype(str), 1.0, -1.0),
        0.0,
    )
    for col in chi_cols:
        c = col.removeprefix("chi__")
        own = (df["company_id"] == c).to_numpy(dtype=float)
        chi2_c = after * own
        chi2_c = chi2_c - chi2_c.mean()
        denom = float(chi2_c @ chi2_c)
        if denom < 1e-12:
            continue
        chi_c = df[col].to_numpy()
        proj = float(chi_c @ chi2_c) / denom
        df[col] = chi_c - proj * chi2_c
    return df, chi_cols


# ---------------------------------------------------------------------------
# Judge-bias contrast: chi over the *judge* company instead of the target.
# ---------------------------------------------------------------------------


def build_chi_judge_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Like ``hypothesis_testing.build_chi_columns`` but the indicator is
    taken over the judge model's company rather than the target's:

        chi_judge_c (i, j, k) = I(judge_company(k) = c)
                                  * (+1/2 if story_company(j) = c else -1/2).

    The column names mirror the primary analysis (``chi__<company>``) so
    that ``freedman_lane`` consumes them unchanged.
    """
    if "judge_company_id" not in df.columns:
        raise KeyError(
            "Expected 'judge_company_id' in df; load via "
            "hypothesis_testing.load_csv / _prepare."
        )
    companies = sorted(set(df["judge_company_id"]).union(df["story_company_id"]))
    stories = df[["story_id", "story_company_id"]].drop_duplicates()
    out = df.copy()
    chi_cols: list[str] = []
    for c in companies:
        n_about = int((stories["story_company_id"] == c).sum())
        n_other = int(len(stories) - n_about)
        if n_about == 0 or n_other == 0:
            continue
        # Only meaningful if we actually have a judge from company c.
        if not (out["judge_company_id"] == c).any():
            continue
        col = f"chi__{c}"
        chi_vec = np.where(out["story_company_id"] == c, 0.5, -0.5)
        out[col] = np.where(out["judge_company_id"] == c, chi_vec, 0.0)
        chi_cols.append(col)
    return out, chi_cols


# ---------------------------------------------------------------------------
# Driver: run one sensitivity experiment end-to-end.
# ---------------------------------------------------------------------------


def run_cutoff_sensitivity(
    df: pd.DataFrame,
    *,
    model_cutoffs: dict[str, str] | None = None,
    story_dates: dict[str, str] | None = None,
    n_permutations: int = 2000,
    seed: int = 0,
    alternative: str = "greater",
    with_ci: bool = False,
    alpha: float = 0.05,
) -> tuple[list[TestResult], dict]:
    """Freedman-Lane on the chi-star (knowledge-cutoff-orthogonalized)
    contrast. ``model_cutoffs`` is a dict ``target_model_id ->
    YYYY-MM-DD``. ``story_dates`` defaults to the dates already on the
    dataframe.
    """
    df2, chi_cols = build_chi_star_columns(
        df, story_dates=story_dates, model_cutoffs=model_cutoffs
    )
    return _run(
        df2,
        chi_cols,
        n_permutations=n_permutations,
        seed=seed,
        alternative=alternative,
        with_ci=with_ci,
        alpha=alpha,
    )


def run_judge_bias(
    df: pd.DataFrame,
    *,
    n_permutations: int = 2000,
    seed: int = 0,
    alternative: str = "greater",
    with_ci: bool = False,
    alpha: float = 0.05,
) -> tuple[list[TestResult], dict]:
    """Freedman-Lane with the judge-company interaction."""
    df2, chi_cols = build_chi_judge_columns(df)
    return _run(
        df2,
        chi_cols,
        n_permutations=n_permutations,
        seed=seed,
        alternative=alternative,
        with_ci=with_ci,
        alpha=alpha,
    )


def _run(
    df: pd.DataFrame,
    chi_cols: list[str],
    *,
    n_permutations: int,
    seed: int,
    alternative: str,
    with_ci: bool,
    alpha: float,
) -> tuple[list[TestResult], dict]:
    results, diag = freedman_lane(
        df,
        chi_cols=chi_cols,
        n_permutations=n_permutations,
        seed=seed,
        alternative=alternative,
    )
    if with_ci:
        for k, r in enumerate(results):
            lo, hi = confidence_interval(diag, k, alpha=alpha)
            r.ci_lower, r.ci_upper = lo, hi
    return results, diag


# ---------------------------------------------------------------------------
# Pretty-printing & I/O
# ---------------------------------------------------------------------------


def _print_results(name: str, results: list[TestResult], with_ci: bool) -> None:
    print(f"\n=== {name} ===")
    header = f"{'company':12s}  {'theta_hat':>10s}  {'p':>7s}  {'p_holm':>7s}"
    if with_ci:
        header += "    95% CI"
    print(header)
    for r in results:
        ci = (
            f"  [{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]"
            if r.ci_lower is not None
            else ""
        )
        print(
            f"{r.company:12s}  {r.theta_hat:+10.4f}  "
            f"{r.p_value:7.4f}  {r.p_holm:7.4f}{ci}"
        )


def _write(results: list[TestResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _synth_judges_per_company(
    n_companies: int = 4,
    n_models_per_company: int = 2,
    n_stories_per_company: int = 3,
    n_judges_per_company: int = 1,
    n_templates: int = 4,
    theta_target: dict[str, float] | None = None,
    theta_judge: dict[str, float] | None = None,
    sigma: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Like ``hypothesis_testing.synthesize`` but each company also ships
    judge models, so the judge-bias contrast is non-degenerate. Also
    supports a per-company judge-side own-company effect.
    """
    rng = np.random.default_rng(seed)
    companies = [f"C{c}" for c in range(n_companies)]
    target_models = {
        c: [f"{c}/m{m}" for m in range(n_models_per_company)] for c in companies
    }
    judge_models = {
        c: [f"{c}/j{m}" for m in range(n_judges_per_company)] for c in companies
    }
    stories = [
        (f"s_{c}_{i}", c) for c in companies for i in range(n_stories_per_company)
    ]
    templates = [f"t{t}" for t in range(n_templates)]
    alpha = {m: rng.normal(0, 0.05) for c in companies for m in target_models[c]}
    beta = {s: rng.normal(0, 0.05) for s, _ in stories}
    gamma = {j: rng.normal(0, 0.05) for c in companies for j in judge_models[c]}
    nu = {t: rng.normal(0, 0.05) for t in templates}
    theta_target = theta_target or {c: 0.0 for c in companies}
    theta_judge = theta_judge or {c: 0.0 for c in companies}
    mu = 0.5
    rows = []
    for c_t in companies:
        for m in target_models[c_t]:
            for s, c_s in stories:
                chi_t = 0.5 if c_s == c_t else -0.5
                for c_j in companies:
                    for j in judge_models[c_j]:
                        chi_j = 0.5 if c_s == c_j else -0.5
                        for t in templates:
                            base = mu + alpha[m] + beta[s] + gamma[j] + nu[t]
                            base += theta_target[c_t] * chi_t
                            base += theta_judge[c_j] * chi_j
                            u = rng.uniform()
                            noise = (
                                rng.beta(2, 8) - 0.2
                                if u < 0.5
                                else rng.beta(8, 2) - 0.8
                            )
                            y = float(np.clip(base + sigma * noise, 0.0, 1.0))
                            rows.append(
                                {
                                    "score": y * 10.0,
                                    "company_id": c_t,
                                    "story_id": s,
                                    "story_company_id": c_s,
                                    "judge_model_id": j,
                                    "target_model_id": m,
                                    "prompt_template_id": t,
                                    "repetition_id": 1,
                                    "story_date": f"2026-{(stories.index((s, c_s)) % 12)+1:02d}-01",
                                }
                            )
    return _prepare(pd.DataFrame(rows))


def _self_test() -> None:
    companies = [f"C{c}" for c in range(4)]
    cutoffs = {m: "2026-06-01" for c in companies for m in [f"{c}/m0", f"{c}/m1"]}

    print("=== Null (no target or judge bias) ===")
    df = _synth_judges_per_company(seed=11)
    r_c, _ = run_cutoff_sensitivity(
        df, model_cutoffs=cutoffs, n_permutations=400, seed=11
    )
    _print_results("knowledge-cutoff sensitivity", r_c, with_ci=False)
    r_j, _ = run_judge_bias(df, n_permutations=400, seed=11)
    _print_results("judge-bias sensitivity", r_j, with_ci=False)

    print("\n\n=== Spiked TARGET bias on C0 (theta=+0.20) ===")
    df = _synth_judges_per_company(
        theta_target={c: (0.20 if c == "C0" else 0.0) for c in companies},
        seed=12,
    )
    r_c, _ = run_cutoff_sensitivity(
        df, model_cutoffs=cutoffs, n_permutations=800, seed=12, with_ci=True
    )
    _print_results("knowledge-cutoff sensitivity", r_c, with_ci=True)
    r_j, _ = run_judge_bias(df, n_permutations=800, seed=12)
    _print_results("judge-bias sensitivity (should NOT flag C0)", r_j, with_ci=False)

    print("\n\n=== Spiked JUDGE bias on C0 (theta=+0.20) ===")
    df = _synth_judges_per_company(
        theta_judge={c: (0.20 if c == "C0" else 0.0) for c in companies},
        seed=13,
    )
    r_c, _ = run_cutoff_sensitivity(
        df, model_cutoffs=cutoffs, n_permutations=800, seed=13
    )
    _print_results(
        "knowledge-cutoff sensitivity (should NOT flag C0)", r_c, with_ci=False
    )
    r_j, _ = run_judge_bias(df, n_permutations=800, seed=13, with_ci=True)
    _print_results("judge-bias sensitivity", r_j, with_ci=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/scores/scores.csv"))
    ap.add_argument(
        "--model-cutoffs",
        type=Path,
        default=None,
        help="JSON mapping target_model_id -> YYYY-MM-DD knowledge cutoff. "
        "Required (or pre-populated dataset) for the cutoff sensitivity.",
    )
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="greater",
    )
    ap.add_argument("--score-max", type=float, default=DEFAULT_SCORE_MAX)
    ap.add_argument("--with-ci", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        _self_test()
        return

    df = load_csv(args.csv, score_max=args.score_max)
    print(
        f"Loaded {len(df):,} rows, "
        f"{df['target_model_id'].nunique()} target models, "
        f"{df['story_id'].nunique()} stories, "
        f"{df['judge_model_id'].nunique()} judges, "
        f"{df['prompt_template_id'].nunique()} templates"
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = json.loads(args.model_cutoffs.read_text()) if args.model_cutoffs else {}
    if not cutoffs:
        print(
            "WARNING: no --model-cutoffs JSON supplied; the "
            "knowledge-cutoff indicator will be zero everywhere and "
            "chi* reduces to chi."
        )
    results, _ = run_cutoff_sensitivity(
        df,
        model_cutoffs=cutoffs,
        n_permutations=args.n_permutations,
        seed=args.seed,
        alternative=args.alternative,
        with_ci=args.with_ci,
        alpha=args.alpha,
    )
    _print_results("knowledge-cutoff sensitivity", results, args.with_ci)
    _write(results, out_dir / "sensitivity_cutoff.json")

    results, _ = run_judge_bias(
        df,
        n_permutations=args.n_permutations,
        seed=args.seed,
        alternative=args.alternative,
        with_ci=args.with_ci,
        alpha=args.alpha,
    )
    _print_results("judge-bias sensitivity", results, args.with_ci)
    _write(results, out_dir / "sensitivity_judge.json")


if __name__ == "__main__":
    main()
main()
