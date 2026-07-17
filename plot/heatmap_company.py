"""Company-level blocked heatmap: each company block collapsed to a single
square. Rows = target company, cols = story company. Cell = weighted-average
favorability score over all models, news stories, prompt templates, judges,
and reps (i.e. the simple mean of `score` across every sample in the block).
Diagonal = a company's models on stories about itself."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

v = pd.read_csv("data/scores/scores.csv").rename(
    columns={
        "story_company_id": "story_company",
        "company_id": "target_company",
    }
)
v = v[v.score.between(1, 5)]
# Exclude the 4 mid-regeneration stories (see explore_heatmap.py).
INFLUX = {"47509984", "47888557", "42413195", "44504709"}
v = v[~v.story_id.astype(str).isin(INFLUX)]

# Companies sorted by pre-registered theta_hat, strongest favoritism first
# (matches plot_hypothesis_testing.py).
_base = ["Alibaba", "Anthropic", "Deepseek", "Google", "Meta", "OpenAI", "xAI"]
try:
    _th = {
        r["company"]: r["theta_hat"]
        for r in json.load(open("outputs/hypothesis_testing.json"))
    }
    CO = sorted(_base, key=lambda c: _th.get(c, 0.0), reverse=True)
except (OSError, json.JSONDecodeError):
    CO = _base

# Sample-weighted mean = simple mean of score over every sample in the block,
# rescaled from [1, 5] to Y in [0, 1] (neutral 3 -> 0.5), as in explore_heatmap.py.
M = (
    v.pivot_table("score", "target_company", "story_company", "mean")
    .reindex(index=CO, columns=CO)
    .to_numpy()
)
M = (M - 1.0) / 4.0
N = (
    v.pivot_table("score", "target_company", "story_company", "count")
    .reindex(index=CO, columns=CO)
    .to_numpy()
)

grand = (float(v.score.mean()) - 1.0) / 4.0
fig, ax = plt.subplots(figsize=(8.5, 7))
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
im = ax.imshow(M, cmap="RdYlGn", norm=norm)

for i in range(len(CO)):
    for j in range(len(CO)):
        ax.text(
            j,
            i,
            f"{M[i, j]:.2f}\n(n={int(N[i, j])})",
            ha="center",
            va="center",
            fontsize=11,
        )

ax.set_xticks(range(len(CO)))
ax.set_xticklabels(CO, rotation=30, ha="right", fontsize=13)
ax.set_yticks(range(len(CO)))
ax.set_yticklabels(CO, fontsize=13)
ax.set_xlabel("News story company", fontsize=15)
ax.set_ylabel("Target model company", fontsize=15)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("Mean Favorability $Y$", fontsize=13)
cb.ax.tick_params(labelsize=11)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig("outputs/heatmap_company.png", dpi=150, bbox_inches="tight")
os.makedirs("outputs/pdf", exist_ok=True)
plt.savefig("outputs/pdf/heatmap_company.pdf", bbox_inches="tight")

own = np.array([M[i, i] for i in range(len(CO))])
offmean = np.array(
    [np.nanmean([M[i, j] for j in range(len(CO)) if j != i]) for i in range(len(CO))]
)
print("Wrote outputs/heatmap_company.png  | grand mean", round(grand, 3))
print("own-company minus mean-over-other-story-companies (Y units):")
for c, d in sorted(zip(CO, own - offmean), key=lambda x: -x[1]):
    print(f"  {c:10s} {d:+.3f}")
