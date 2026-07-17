# Model Partiality

This code for a [pre-registered](https://osf.io/et6hv/overview) experiment testing if models by seven selected companies are more favorable to their creator companies than to other companies. Read the paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7059338).

For the responses by the target and judge models, see [the accompanying dataset](https://huggingface.co/datasets/lennart-finke/partiality).

For the exact code used in the preregistration, see [this commit](https://github.com/lennart-finke/partiality/commit/f4364e7d985b23a6539961dcf4110c7ab3114bd2). Note the data files describing the queries used for the experiment are only available [on OSF](https://osf.io/et6hv/overview) as they are larger than GitHub's default file size limit.


## Execution Pipeline
Scripts should be run in the order below. Steps sharing a number (e.g. `1a`, `1b`) are independent and can run in parallel. `config.py` is a shared module imported throughout, not run directly.

Since reproduction of the whole experiment needs a few hundred dollars of inference compute, we provide a way to execute the pipeline starting from after the API calls, by using `load_from_hf.py` and then continuing from step 12.

1. `nyt.py` — Fetch NYT Archive API articles.
2. `hn.py` — Fetch daily Hacker News front-page stories. (concurrent with (1))
3. `company_coverage.py` — Count articles mentioning each company to shortlist companies.
4. `judge.py` — LLM-score each story for relevance, reputational risk, and controversy.
5. `top_stories.py` — Select and deduplicate top negative stories per company.
6. `open_in_firefox.py` — Open selected story URLs to save full-text HTML with a userscript or similar.
7. `process_full_texts.py` — Convert saved HTML into clean per-URL Markdown.
8. `preflight_queries.py` — Render target prompts and judge jobs into JSON.
9. `power_analysis.py` — Simulate statistical power of the test design. Not needed to reproduce our results.
10. `query_targets.py` — Query target models with rendered prompts via OpenRouter.
11. `query_judges.py` — Judge models score each target response for favorability.
12. `export_scores_csv.py` — Join judge scores into the analysis CSV.
13. `hypothesis_testing.py` — Main per-company permutation (Freedman–Lane) partiality test.
14. `sensitivity_analyses.py` — Knowledge-cutoff and judge-bias sensitivity tests. (concurrent with (13))
15. `per_model_analysis.py` — Per-model (not per-company) partiality estimates. (concurrent with (13))
16. `qualitative_analysis.py` — Keyword-scan transcripts for awareness, disclosure, refusal, fabrication. (concurrent with (13))
17. `pick_examples.py` — Select example transcripts for Figure 1. (concurrent with (13))
18. `final_results.py` — Finalize theta, p-values, CIs and emit LaTeX tables (re-runs the core tests).
19. `plot/plot_hypothesis_testing.py` — Forest plot of per-company partiality estimates.
20. `plot/plot_per_model.py` — Forest plot of per-model partiality estimates. (concurrent with (19))
21. `plot/heatmap_company.py` — Company-by-company mean favorability heatmap. (concurrent with (19))
22. `plot/heatmap.py` — Model-by-story blocked favorability heatmap. (concurrent with (19))

## Modifications from the Preregistration
We list changes from the pre-registered code base. The non-self explanatory changes are also described in the paper.
- We added plotting scripts in `plot/`.
- We added the analyses `per_model_analysis.py`, `pick_examples.py`, `qualitative_analysis.py` as described in the paper.
- We added `load_from_hf.py` and a uv project (`pyproject.toml` / `uv.lock`) to reproduce from the published dataset without re-running collection. `load_from_hf.py` unpacks the dataset into the local file layout the scripts already expect; the analysis scripts themselves are unchanged.
- We added per-model knowledge-cutoff dates as a curated input at `data/model_cutoffs.json` (consumed by the cutoff sensitivity in `sensitivity_analyses.py` and `final_results.py`); sources are documented in the file's `_meta` block.
- Templates 24 and 25 had inverted rubrics, we corrected this in `export_scores_csv.py` and `config.py`.
- In `hypothesis_testing.py`, we always use a two-sided hypothesis, as described in the preregistration main text.
- In `sensitivity_analyses.py`, the logic to map judge models to their creator company was broken. We fixed this.

