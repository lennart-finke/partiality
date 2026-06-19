"""Power analysis for the own-company partiality test.

We sample the independent-variable structure from
``data/judge_jobs.json`` (produced by ``preflight_queries.py``), so the
design — which target models / stories / judges / templates / reps each
observation belongs to — matches the real experiment. Y is generated
under H1 with a configurable true theta on a chosen company, additive
fixed effects, bimodal error (mixture of two normals), and finally
quantized to 15 equally-spaced bins on [0, 1] (matching the judge's
3-Likert-on-1..5 → 15-level discrete scale).

For each (theta_true, simulation) we run the Freedman-Lane test from
``hypothesis_testing.py`` and record:

  * theta_hat (computed once on the original data inside FL),
  * permutation p-value (one-sided "greater"),
  * power = P(reject H0 at alpha=0.05) across sims.

Usage:
    python power_analysis.py --thetas 0 0.05 0.1 0.15 0.2 --sims 50 \
        --stories-per-company 3 --n-permutations 500 \
        --jobs data/judge_jobs.json --company OpenAI
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hypothesis_testing import (
    build_chi_columns,
    freedman_lane,
    _prepare,
)


# ---------------------------------------------------------------------------
# Design loading & subsampling
# ---------------------------------------------------------------------------


def load_design(
    jobs_path: Path,
    stories_per_company: int | None = None,
    templates_per_target: int | None = None,
    reps: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load the (i, j, k, t, r) design from judge_jobs.json.

    Optionally subsample to keep the simulation tractable.
    """
    data = json.loads(jobs_path.read_text())
    df = pd.DataFrame(
        [
            {
                "story_id": str(r["story_id"]),
                "story_company_id": r["story_company"],
                "company_id": r["target_company"],
                "target_model_id": r["target_model"],
                "prompt_template_id": str(r["template_id"]),
                "repetition_id": int(r["rep"]),
                "judge_model_id": r["judge_model"],
                "story_date": str(r.get("story_date") or "2026-01-01"),
            }
            for r in data
        ]
    )
    rng = np.random.default_rng(seed)

    if stories_per_company is not None:
        keep = []
        for c, g in df.drop_duplicates("story_id").groupby("story_company_id"):
            sids = g["story_id"].to_numpy()
            n = min(stories_per_company, sids.size)
            keep.extend(rng.choice(sids, size=n, replace=False).tolist())
        df = df[df["story_id"].isin(keep)].copy()

    if templates_per_target is not None:
        templates = df["prompt_template_id"].unique()
        chosen = rng.choice(
            templates, size=min(templates_per_target, templates.size), replace=False
        )
        df = df[df["prompt_template_id"].isin(chosen)].copy()

    if reps is not None:
        df = df[df["repetition_id"] <= reps].copy()

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Simulation under H1
# ---------------------------------------------------------------------------


@dataclass
class SimParams:
    """Realistic-ish hyperparameters for synthesising Y."""

    mu: float = 0.5
    sd_target_model: float = 0.05
    sd_story: float = 0.08
    sd_judge: float = 0.05
    sd_template: float = 0.06
    # Bimodal mixture: two normals on the [0, 1] latent scale.
    bimodal_p_high: float = 0.4
    bimodal_mu_low: float = -0.15
    bimodal_mu_high: float = 0.25
    bimodal_sd: float = 0.10
    n_bins: int = 15  # judge mean of 3 Likerts in {1..5}: 13 levels; use 15 for slack


def _bimodal_noise(rng: np.random.Generator, n: int, p: SimParams) -> np.ndarray:
    high = rng.uniform(size=n) < p.bimodal_p_high
    return np.where(
        high,
        rng.normal(p.bimodal_mu_high, p.bimodal_sd, n),
        rng.normal(p.bimodal_mu_low, p.bimodal_sd, n),
    )


def _quantize(y: np.ndarray, n_bins: int) -> np.ndarray:
    levels = np.linspace(0.0, 1.0, n_bins)
    idx = np.clip(np.round(y * (n_bins - 1)).astype(int), 0, n_bins - 1)
    return levels[idx]


def simulate_response(
    df: pd.DataFrame,
    theta_true: dict[str, float],
    params: SimParams,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return df with a fresh 'y' column drawn under H1 with the given
    true thetas. 'score' is set to y * (n_bins - 1) so the bins match
    a 0..(n_bins-1) integer scale (close to the judge's 3..15 scale)."""
    n = len(df)
    alpha = {
        m: rng.normal(0, params.sd_target_model) for m in df["target_model_id"].unique()
    }
    beta = {s: rng.normal(0, params.sd_story) for s in df["story_id"].unique()}
    gamma = {j: rng.normal(0, params.sd_judge) for j in df["judge_model_id"].unique()}
    nu = {
        t: rng.normal(0, params.sd_template) for t in df["prompt_template_id"].unique()
    }

    # chi(c, j) — same parametrization as build_chi_columns: +1/2, -1/2.
    stories = df[["story_id", "story_company_id"]].drop_duplicates()
    companies = sorted(stories["story_company_id"].unique())
    n0 = {c: int((stories["story_company_id"] == c).sum()) for c in companies}
    n1 = {c: len(stories) - n0[c] for c in companies}

    fe = (
        df["target_model_id"].map(alpha).to_numpy()
        + df["story_id"].map(beta).to_numpy()
        + df["judge_model_id"].map(gamma).to_numpy()
        + df["prompt_template_id"].map(nu).to_numpy()
    )

    eta = np.full(n, params.mu) + fe
    for c, th in theta_true.items():
        if c not in n0 or n0[c] == 0 or n1[c] == 0:
            continue
        chi = np.where(df["story_company_id"] == c, 0.5, -0.5)
        own = (df["company_id"] == c).to_numpy()
        eta = eta + th * own * chi

    y_latent = eta + _bimodal_noise(rng, n, params)
    y_obs = _quantize(y_latent, params.n_bins)

    out = df.copy()
    out["y"] = y_obs
    out["score"] = y_obs * (params.n_bins - 1)  # 0..(n_bins-1) integer-ish
    return out


# ---------------------------------------------------------------------------
# One simulation: run both estimators
# ---------------------------------------------------------------------------


def run_one(
    df: pd.DataFrame,
    company_of_interest: str,
    n_permutations: int,
    seed: int,
) -> dict:
    df_chi, chi_cols = build_chi_columns(df)
    target_col = f"chi__{company_of_interest}"
    if target_col not in chi_cols:
        raise ValueError(
            f"chi column for {company_of_interest!r} not present (no contrast)"
        )

    t0 = time.perf_counter()
    results, diag = freedman_lane(
        df_chi,
        chi_cols=chi_cols,
        n_permutations=n_permutations,
        seed=seed,
        alternative="greater",
    )
    elapsed = time.perf_counter() - t0
    idx = diag["companies"].index(company_of_interest)

    return {
        "theta_hat": float(diag["theta_hat"][idx]),
        "p_permutation_greater": float(results[idx].p_value),
        "time_permutation": elapsed,
    }


# ---------------------------------------------------------------------------
# Power loop
# ---------------------------------------------------------------------------


def power_analysis(
    design: pd.DataFrame,
    company: str,
    thetas: list[float],
    n_sims: int,
    n_permutations: int,
    params: SimParams,
    seed: int = 0,
) -> pd.DataFrame:
    rng_seq = np.random.SeedSequence(seed)
    rows = []
    for theta in thetas:
        theta_dict = {company: float(theta)}
        for s in range(n_sims):
            sim_seed = int(rng_seq.spawn(1)[0].generate_state(1)[0])
            df_sim = simulate_response(
                design, theta_dict, params, np.random.default_rng(sim_seed)
            )
            df_sim = _prepare(df_sim.assign(score=df_sim["score"]))
            res = run_one(df_sim, company, n_permutations, seed=sim_seed)
            res["theta_true"] = float(theta)
            res["sim"] = s
            rows.append(res)
            print(
                f"  theta={theta:+.3f} sim={s+1:>3d}/{n_sims}  "
                f"theta_hat={res['theta_hat']:+.4f}  "
                f"p_perm={res['p_permutation_greater']:.3f}  "
                f"({res['time_permutation']:.1f}s)"
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summarize(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    g = results.groupby("theta_true")
    out = g.agg(
        n_sims=("sim", "count"),
        mean_theta_hat=("theta_hat", "mean"),
        power=("p_permutation_greater", lambda x: float(np.mean(x < alpha))),
        mean_p=("p_permutation_greater", "mean"),
    ).reset_index()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, default=Path("data/judge_jobs.json"))
    ap.add_argument(
        "--company",
        type=str,
        required=True,
        help="Company to spike with theta_true (e.g. OpenAI)",
    )
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.05])
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--n-permutations", type=int, default=300)
    ap.add_argument("--stories-per-company", type=int, default=None)
    ap.add_argument("--templates-per-target", type=int, default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/power_analysis.csv"))
    args = ap.parse_args()

    design = load_design(
        args.jobs,
        stories_per_company=args.stories_per_company,
        templates_per_target=args.templates_per_target,
        reps=args.reps,
        seed=args.seed,
    )
    print(
        f"Design: {len(design):,} rows  "
        f"({design['target_model_id'].nunique()} target models, "
        f"{design['story_id'].nunique()} stories, "
        f"{design['judge_model_id'].nunique()} judges, "
        f"{design['prompt_template_id'].nunique()} templates, "
        f"max rep={design['repetition_id'].max()})"
    )

    params = SimParams(n_bins=args.n_bins)
    results = power_analysis(
        design,
        args.company,
        args.thetas,
        args.sims,
        args.n_permutations,
        params,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    summary = summarize(results)
    print("\n=== Summary ===")
    with pd.option_context(
        "display.float_format", "{:.4f}".format, "display.width", 160
    ):
        print(summary.to_string(index=False))
    print(f"\nPer-sim results: {args.out}")


if __name__ == "__main__":
    main()
