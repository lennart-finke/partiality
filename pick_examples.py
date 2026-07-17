"""Pick non-cherry-picked partiality examples via linear-model residuals.

Fit the additive fixed-effects model (sum-to-zero parametrization)

    Y' = mu + alpha_i + beta_j + gamma_k + nu_t + eps_ijkt

where Y = (score - 1) / 4 in [0, 1] is favorability (HIGH = partial), i = target
model, j = news story, k = judge model, t = prompt template. Then for each of
{xAI, DeepSeek, Anthropic, OpenAI} report the sample with the largest residual
(most favorable beyond what the additive effects predict) among samples where
the target model AND the news story are about that same company.

    .venv/bin/python pick_examples.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import statsmodels.formula.api as smf

from config import company_display, flip_inverted_scores, model_display

SCORES_DIR = "data/scores"
OUT_PATH = "outputs/own_company_residual_examples.jsonl"
# Company labels as stored in the data (config uses "Deepseek").
COMPANIES = ["xAI", "Deepseek", "Anthropic", "OpenAI"]
# Restrict the example pool to each company's newest target model. The linear
# model is still fit on all data; only the max-residual selection is filtered.
NEWEST_MODEL = {
    "xAI": "x-ai/grok-4.3",
    "Deepseek": "deepseek/deepseek-v4-pro",
    "Anthropic": "anthropic/claude-opus-4.7",
    "OpenAI": "openai/gpt-5.5",
}
FIELDS = (
    "story_id",
    "story_headline",
    "story_company",
    "target_company",
    "target_model",
    "judge_model",
    "template_id",
    "rep",
    "score",
    "reasoning",
    "response_file",
)


def _load(path: str) -> dict | None:
    try:
        d = flip_inverted_scores(json.load(open(path)))
    except Exception:
        return None
    return {k: d.get(k) for k in FIELDS}


def load_scores() -> pd.DataFrame:
    paths = [e.path for e in os.scandir(SCORES_DIR) if e.name.endswith(".json")]
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = [r for r in ex.map(_load, paths) if r is not None]
    df = pd.DataFrame(rows)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["score"].between(1, 5)].copy()
    df["y"] = (df["score"] - 1) / 4
    for c in ("target_model", "story_id", "judge_model", "template_id"):
        df[c] = df[c].astype(str)
    return df


def fit_residuals(df: pd.DataFrame) -> pd.DataFrame:
    # Sum-to-zero (deviation) coding via patsy's Sum contrast; residuals are
    # the eps_ijkt of the fitted additive model.
    formula = (
        "y ~ C(target_model, Sum) + C(story_id, Sum)"
        " + C(judge_model, Sum) + C(template_id, Sum)"
    )
    fit = smf.ols(formula, data=df).fit()
    df = df.copy()
    df["resid"] = fit.resid
    return df


def main() -> None:
    df = load_scores()
    print(
        f"valid samples: {len(df):,}  models: {df.target_model.nunique()}  "
        f"stories: {df.story_id.nunique()}  judges: {df.judge_model.nunique()}  "
        f"templates: {df.template_id.nunique()}"
    )
    df = fit_residuals(df)

    own = df[df.target_company == df.story_company]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for company in COMPANIES:
            sub = own[
                (own.target_company == company)
                & (own.target_model == NEWEST_MODEL[company])
            ]
            if sub.empty:
                print(f"{company_display(company)}: no own-company samples")
                continue
            r = sub.loc[sub.resid.idxmax()]
            completion = json.load(open(r.response_file)).get("completion")
            rec = {
                "residual": float(r.resid),
                "company": company_display(company),
                "article_title": r.story_headline,
                "model": model_display(r.target_model),
                "response": completion,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"{rec['company']:10s} residual={rec['residual']:+.3f}  "
                f"{rec['model']}  -- {rec['article_title']}"
            )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
