"""Permutation-based hypothesis test for own-company partiality of LLMs.

Implements the Freedman-Lane procedure described in the project spec.

Reduced model:
    Y' = mu + alpha_i + beta_j + gamma_k + nu_t + eps_{ijkt}
where i = target model, j = story, k = judge model, t = prompt template.

Full model (per company c):
    Y = ... + theta_c * I(company(i)=c) * chi(c, j) + eps
with chi(c, j) = +1/2 if story j is about company c, else -1/2.

H_0^{(c)} : theta_c = 0, tested via Freedman-Lane: residuals from the
reduced fit are permuted within blocks (target_model, judge_model,
template, rep) -- i.e. all indices fixed except j -- then the full
model is refit on the rebuilt response and theta_c re-estimated. The
empirical distribution of theta_c values across permutations is the
null. Reported p-values are Holm-Bonferroni corrected across companies.

Fixed-effect absorption (alpha, beta, gamma, nu, parametrized to sum to
zero) is delegated to `pyfixest.feols`. The Freedman-Lane permutation
and confidence-interval-by-inversion machinery sit on top of
pyfixest-demeaned arrays via the Frisch-Waugh-Lovell identity.

CSV input format matches `export_scores_csv.py`:
    score, company_id, story_id, story_company_id, judge_model_id,
    target_model_id, prompt_template_id, repetition_id, story_date

Usage:
    python hypothesis_testing.py --csv data/scores/scores.csv --out outputs/
    python hypothesis_testing.py --synthetic                 # self-test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyfixest as pf


DEFAULT_SCORE_MAX = 10.0  # judge rubric is 0..10
FE_COLS = ["target_model_id", "story_id", "judge_model_id", "prompt_template_id"]
FE_FORMULA = " + ".join(FE_COLS)


# ---------------------------------------------------------------------------
# Data loading & normalization
# ---------------------------------------------------------------------------


def load_csv(path: Path, score_max: float = DEFAULT_SCORE_MAX) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _prepare(df, score_max)


def _prepare(df: pd.DataFrame, score_max: float = DEFAULT_SCORE_MAX) -> pd.DataFrame:
    df = df.dropna(subset=["score"]).copy()
    df["y"] = (df["score"].astype(float) / score_max).clip(0.0, 1.0)
    for c in FE_COLS + ["company_id", "story_company_id"]:
        df[c] = df[c].astype(str)
    df["repetition_id"] = df["repetition_id"].fillna(1).astype(int)
    # Company-level identifiers used for the permutation stratum (the
    # company-coarsened scheme — see DEFAULT_BLOCK_COLS).
    df["judge_company_id"] = df["judge_model_id"].astype(str).str.split("/").str[0]
    df["target_company_id"] = df["target_model_id"].astype(str).str.split("/").str[0]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interaction columns chi(c, j) * I(target_company = c)
# ---------------------------------------------------------------------------


def build_chi_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of `df` with one column `chi__<company>` per company.

        chi(c, j) = +1/2  if story j is about company c
                    -1/2  otherwise,
    multiplied by I(target_company == c).
    """
    companies = sorted(set(df["company_id"]).union(df["story_company_id"]))
    stories = df[["story_id", "story_company_id"]].drop_duplicates()
    out = df.copy()
    chi_cols: list[str] = []
    for c in companies:
        n0 = int((stories["story_company_id"] == c).sum())
        n1 = int(len(stories) - n0)
        if n0 == 0 or n1 == 0:
            continue
        col = f"chi__{c}"
        chi_vec = np.where(out["story_company_id"] == c, 0.5, -0.5)
        out[col] = np.where(out["company_id"] == c, chi_vec, 0.0)
        chi_cols.append(col)
    return out, chi_cols


# ---------------------------------------------------------------------------
# pyfixest helpers
# ---------------------------------------------------------------------------


def _demean(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Return residuals of each column after absorbing the FEs.

    Equivalent to M_FE @ X, where M_FE projects orthogonally to the
    column span of all level indicators for the four fixed effects.
    """
    cols = []
    for col in columns:
        fit = pf.feols(f"{col} ~ 1 | {FE_FORMULA}", data=df)
        cols.append(np.asarray(fit.resid()).reshape(-1))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


# ---------------------------------------------------------------------------
# Freedman-Lane test
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    company: str
    theta_hat: float
    p_value: float
    p_holm: float
    ci_lower: float | None = None
    ci_upper: float | None = None


DEFAULT_BLOCK_COLS = (
    "target_company_id",
    "judge_company_id",
    "prompt_template_id",
)


def _block_groups(
    df: pd.DataFrame,
    block_cols: Sequence[str] = DEFAULT_BLOCK_COLS,
) -> list[np.ndarray]:
    """Indices grouped by the permutation stratum. Story labels are
    permuted within each block.

    We stratify on (target_company, judge_company, template) rather than
    on the full (target_model, judge_model, template, rep) cell. Pooling
    sibling models within a company gives non-degenerate blocks (the
    finer cell is dominated by singletons in this design) at the cost of
    a slightly stronger exchangeability assumption: residuals must be
    exchangeable across model pairs within the same company pair after
    FE absorption. The regression FEs remain at the model level, so
    theta_hat is unaffected -- only the null distribution changes.
    """
    codes = df.groupby(list(block_cols), sort=False).ngroup().to_numpy()
    return [np.where(codes == g)[0] for g in range(codes.max() + 1)]


def freedman_lane(
    df: pd.DataFrame,
    chi_cols: list[str] | None = None,
    n_permutations: int = 2000,
    seed: int = 0,
    alternative: str = "greater",
    block_cols: Sequence[str] = DEFAULT_BLOCK_COLS,
) -> tuple[list[TestResult], dict]:
    """Run Freedman-Lane on the dataframe. Returns per-company results
    plus a diagnostics dict with the per-permutation operator vectors
    needed to invert the test (for CIs).
    """
    if chi_cols is None:
        df, chi_cols = build_chi_columns(df)

    # Demean Y and each chi column against the four-way FE structure.
    y_tilde = _demean(df, ["y"]).reshape(-1)
    Z_tilde = _demean(df, chi_cols)

    # Drop chi columns with zero norm (a company with no stories on one
    # side of the contrast within the present data).
    norms = (Z_tilde * Z_tilde).sum(axis=0)
    keep = np.where(norms > 1e-12)[0]
    Z_tilde = Z_tilde[:, keep]
    kept_cols = [chi_cols[k] for k in keep]
    kept_companies = [c.removeprefix("chi__") for c in kept_cols]

    # FWL: theta_hat = (Z~'Z~)^-1 Z~' y_tilde.
    A = np.linalg.solve(Z_tilde.T @ Z_tilde, Z_tilde.T)  # (C, N)
    theta_hat = A @ y_tilde

    # Cross-check against pyfixest's joint fit (sanity, should match).
    full = pf.feols(f"y ~ {' + '.join(kept_cols)} | {FE_FORMULA}", data=df)
    pf_theta = full.coef().reindex(kept_cols).to_numpy()
    if not np.allclose(theta_hat, pf_theta, atol=1e-6):
        raise RuntimeError(
            f"FWL theta_hat {theta_hat} disagrees with pyfixest {pf_theta}"
        )

    # Build permutation operators.
    blocks = _block_groups(df, block_cols=block_cols)
    rng = np.random.default_rng(seed)
    n = len(df)
    null_theta = np.zeros((n_permutations, kept_cols.__len__()))
    null_shift = np.zeros((n_permutations, kept_cols.__len__()))
    base = np.arange(n)
    for b in range(n_permutations):
        p = base.copy()
        for idx in blocks:
            if idx.size > 1:
                p[idx] = idx[rng.permutation(idx.size)]
        # FL: permute residuals -> coefficient via FWL on permuted residuals.
        null_theta[b] = A @ y_tilde[p]
        # Shift operator: how the per-permutation theta changes when we
        # subtract theta0 * chi_c from y BEFORE running F-L (used for CI
        # inversion). The c-th company sees a shift of -theta0 times
        # (A @ P @ chi_c_tilde)[c]; we cache just the relevant diagonal.
        null_shift[b] = np.einsum("ki,ik->k", A, Z_tilde[p])

    # Per-company p-values.
    pvals = np.empty(len(kept_cols))
    for k in range(len(kept_cols)):
        t = theta_hat[k]
        null = null_theta[:, k]
        if alternative == "greater":
            pvals[k] = (1 + np.sum(null >= t)) / (n_permutations + 1)
        elif alternative == "less":
            pvals[k] = (1 + np.sum(null <= t)) / (n_permutations + 1)
        else:
            pvals[k] = (1 + np.sum(np.abs(null) >= abs(t))) / (n_permutations + 1)
    p_holm = holm_bonferroni(pvals)

    results = [
        TestResult(
            company=kept_companies[k],
            theta_hat=float(theta_hat[k]),
            p_value=float(pvals[k]),
            p_holm=float(p_holm[k]),
        )
        for k in range(len(kept_cols))
    ]
    diagnostics = {
        "companies": kept_companies,
        "theta_hat": theta_hat,
        "null_theta": null_theta,
        "null_shift": null_shift,
        "alternative": alternative,
        "n_permutations": n_permutations,
        "block_cols": tuple(block_cols),
        "pyfixest_fit": full,
    }
    return results, diagnostics


def holm_bonferroni(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    out = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        adj = (m - rank) * p[k]
        running = max(running, adj)
        out[k] = min(running, 1.0)
    return out


# ---------------------------------------------------------------------------
# Confidence intervals by test inversion
# ---------------------------------------------------------------------------


def confidence_interval(
    diagnostics: dict,
    company_idx: int,
    alpha: float = 0.05,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Invert the F-L test for one company on a fine grid.

    Subtracting theta0 * chi_c from Y shifts both the observed
    statistic and each permuted statistic linearly:
        T_obs(theta0)         = theta_hat - theta0
        T_perm(theta0, b)     = null_theta[b] - theta0 * null_shift[b]
    so we reuse the cached operators without any refits.
    """
    k = company_idx
    theta_hat = diagnostics["theta_hat"][k]
    null_theta = diagnostics["null_theta"][:, k]
    null_shift = diagnostics["null_shift"][:, k]
    alt = diagnostics["alternative"]
    B = null_theta.size

    def pval(theta0: float) -> float:
        t_obs = theta_hat - theta0
        t_null = null_theta - theta0 * null_shift
        if alt == "greater":
            return (1 + np.sum(t_null >= t_obs)) / (B + 1)
        if alt == "less":
            return (1 + np.sum(t_null <= t_obs)) / (B + 1)
        return (1 + np.sum(np.abs(t_null) >= abs(t_obs))) / (B + 1)

    if grid is None:
        span = max(1e-6, abs(theta_hat), 4 * float(np.std(null_theta)))
        grid = np.linspace(theta_hat - 6 * span, theta_hat + 6 * span, 401)

    accept = np.asarray([pval(t0) > alpha for t0 in grid])
    if not accept.any():
        return float("nan"), float("nan")
    keep = grid[accept]
    return float(keep.min()), float(keep.max())


# ---------------------------------------------------------------------------
# Synthetic data and self-test
# ---------------------------------------------------------------------------


def synthesize(
    n_models_per_company: int = 2,
    n_companies: int = 4,
    n_stories_per_company: int = 3,
    n_judges: int = 3,
    n_templates: int = 4,
    n_reps: int = 1,
    theta_true: dict[str, float] | None = None,
    sigma: float = 0.15,
    bimodal: bool = True,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    companies = [f"C{c}" for c in range(n_companies)]
    target_models = {
        c: [f"{c}/m{m}" for m in range(n_models_per_company)] for c in companies
    }
    stories = [
        (f"s_{c}_{i}", c) for c in companies for i in range(n_stories_per_company)
    ]
    judges = [f"judge/{k}" for k in range(n_judges)]
    templates = [f"t{t}" for t in range(n_templates)]
    alpha = {m: rng.normal(0, 0.05) for c in companies for m in target_models[c]}
    beta = {s: rng.normal(0, 0.05) for s, _ in stories}
    gamma = {j: rng.normal(0, 0.05) for j in judges}
    nu = {t: rng.normal(0, 0.05) for t in templates}
    mu = 0.5
    theta_true = theta_true or {c: 0.0 for c in companies}
    rows = []
    for c_t in companies:
        for m in target_models[c_t]:
            for s, c_s in stories:
                chi_val = 0.5 if c_s == c_t else -0.5
                for j in judges:
                    for t in templates:
                        for r in range(1, n_reps + 1):
                            base = mu + alpha[m] + beta[s] + gamma[j] + nu[t]
                            base += theta_true[c_t] * chi_val
                            if bimodal:
                                u = rng.uniform()
                                noise = (
                                    rng.beta(2, 8) - 0.2
                                    if u < 0.5
                                    else rng.beta(8, 2) - 0.8
                                )
                                y = base + sigma * noise
                            else:
                                y = base + rng.normal(0, sigma)
                            y = float(np.clip(y, 0.0, 1.0))
                            rows.append(
                                {
                                    "score": y * 10.0,
                                    "company_id": c_t,
                                    "story_id": s,
                                    "story_company_id": c_s,
                                    "judge_model_id": j,
                                    "target_model_id": m,
                                    "prompt_template_id": t,
                                    "repetition_id": r,
                                    "story_date": f"2026-{(stories.index((s, c_s)) % 12)+1:02d}-01",
                                }
                            )
    return _prepare(pd.DataFrame(rows)), theta_true


def _self_test() -> None:
    print("=== Null case (theta_true = 0) ===")
    df, _ = synthesize(seed=1)
    results, _ = freedman_lane(df, n_permutations=500, seed=1)
    for r in results:
        print(
            f"  {r.company:8s}  theta_hat={r.theta_hat:+.4f}  "
            f"p={r.p_value:.3f}  p_holm={r.p_holm:.3f}"
        )
    n_fp = sum(1 for r in results if r.p_holm < 0.05)
    print(f"  false positives at Holm 0.05: {n_fp}/{len(results)}")

    print("\n=== Spiked (C0 theta=+0.20) ===")
    theta = {f"C{c}": 0.0 for c in range(4)}
    theta["C0"] = 0.20
    df, _ = synthesize(theta_true=theta, seed=2)
    results, diag = freedman_lane(df, n_permutations=1000, seed=2)
    for r in results:
        mark = "  <- spiked" if r.company == "C0" else ""
        print(
            f"  {r.company:8s}  theta_hat={r.theta_hat:+.4f}  "
            f"p={r.p_value:.3f}  p_holm={r.p_holm:.3f}{mark}"
        )
    k = diag["companies"].index("C0")
    lo, hi = confidence_interval(diag, k, alpha=0.05)
    print(f"  95% CI for C0:  [{lo:+.3f}, {hi:+.3f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_results(results: list[TestResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/scores/scores.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/hypothesis_testing.json"))
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="greater",
        help="'greater' tests own-company favoritism (default).",
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
        f"{df['company_id'].nunique()} target companies, "
        f"{df['target_model_id'].nunique()} target models, "
        f"{df['story_id'].nunique()} stories, "
        f"{df['judge_model_id'].nunique()} judges, "
        f"{df['prompt_template_id'].nunique()} templates"
    )

    results, diag = freedman_lane(
        df,
        n_permutations=args.n_permutations,
        seed=args.seed,
        alternative=args.alternative,
    )
    if args.with_ci:
        for k, r in enumerate(results):
            lo, hi = confidence_interval(diag, k, alpha=args.alpha)
            r.ci_lower, r.ci_upper = lo, hi

    print(
        f"\n{'company':12s}  {'theta_hat':>10s}  {'p':>7s}  {'p_holm':>7s}"
        f"{'    95% CI' if args.with_ci else ''}"
    )
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
    _write_results(results, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
