"""Blocked heatmap: target models (blocks of 3 by company) x news stories
(variable blocks by story company). Cell = raw favorability score (1-5)
averaged over all prompt templates, judges, and reps. RdYlGn colormap
centered at the neutral anchor (3)."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def company_order():
    """Companies sorted by pre-registered theta_hat, strongest favoritism
    first (matches plot_hypothesis_testing.py's ordering)."""
    base = ["Alibaba", "Anthropic", "Deepseek", "Google", "Meta", "OpenAI", "xAI"]
    try:
        th = {
            r["company"]: r["theta_hat"]
            for r in json.load(open("outputs/hypothesis_testing.json"))
        }
        return sorted(base, key=lambda c: th.get(c, 0.0), reverse=True)
    except (OSError, json.JSONDecodeError):
        return base


# Canonical pre-registered scores (all 210 stories). Rename to the
# column names used below. Score is the mean of the three 1-5 sub-scores.
v = pd.read_csv("data/scores/scores.csv").rename(
    columns={
        "story_company_id": "story_company",
        "target_model_id": "target_model",
        "company_id": "target_company",
    }
)
v = v[v.score.between(1, 5)]  # drop the handful of error sentinels (0/10)
# Drop 4 stories that are mid-regeneration (response files staged-deleted in the
# working tree): they survive only as a couple of stale spotcheck rows on a
# single model, so their columns would be ~all-empty. See git status.
INFLUX = {"47509984", "47888557", "42413195", "44504709"}
v = v[~v.story_id.astype(str).isin(INFLUX)]

# Consistent company order on both axes -> own-company blocks sit on the diagonal.
CO = company_order()

# Row order: models grouped by company (blocks of 3), sorted within.
models = v.groupby("target_company")["target_model"].unique().reindex(CO).explode()
row_models = list(models)
row_co = list(models.index)

# Col order: stories grouped by story company (variable blocks),
# chronological within each block.
dates = v.groupby("story_id")["story_date"].first()
col_stories, col_co = [], []
for c in CO:
    s = sorted(
        v.loc[v.story_company == c, "story_id"].unique(),
        key=lambda sid: (str(dates[sid]), str(sid)),
    )
    col_stories += s
    col_co += [c] * len(s)

# Mean score per (model, story) over templates/judges/reps,
# rescaled from [1, 5] to Y in [0, 1] (neutral 3 -> 0.5).
cell = v.pivot_table("score", "target_model", "story_id", "mean")
M = cell.reindex(index=row_models, columns=col_stories).to_numpy()
M = (M - 1) / 4


# Block boundaries.
def bounds(labels):
    b = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            b.append(i)
    b.append(len(labels))
    return b


rb, cb = bounds(row_co), bounds(col_co)

# NaN spacer rows/columns between blocks -> small white gaps. Cells are much
# wider than tall (206 cols vs 21 rows), so 3 spacer columns ~ 1 spacer row
# in physical size.
GX, GY = 3, 1
Mp = np.insert(M, np.repeat(rb[1:-1], GY), np.nan, axis=0)
Mp = np.insert(Mp, np.repeat(cb[1:-1], GX), np.nan, axis=1)

fig, ax = plt.subplots(figsize=(20, 7))
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
cmap = plt.get_cmap("RdYlGn").copy()
cmap.set_bad("white")
im = ax.imshow(Mp, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

# Thin border around each block box; no axes rim.
from matplotlib.patches import Rectangle

for i in range(len(rb) - 1):
    for j in range(len(cb) - 1):
        ax.add_patch(
            Rectangle(
                (cb[j] + j * GX - 0.5, rb[i] + i * GY - 0.5),
                cb[j + 1] - cb[j],
                rb[i + 1] - rb[i],
                fill=False,
                edgecolor="black",
                lw=1.2,
                clip_on=False,
            )
        )
for spine in ax.spines.values():
    spine.set_visible(False)

# Model names per row (offset by the gaps before each row's block).
ypos = [i + GY * sum(1 for b in rb[1:-1] if b <= i) for i in range(len(row_models))]
ax.set_yticks(ypos)
ax.set_yticklabels([m.split("/")[-1] for m in row_models], fontsize=15)
# Company labels at block centers (offset by the gaps before each block).
ax.set_xticks([(cb[i] + cb[i + 1]) / 2 - 0.5 + i * GX for i in range(len(cb) - 1)])
ax.set_xticklabels([f"{col_co[cb[i]]}" for i in range(len(cb) - 1)], fontsize=18)
ax.set_ylabel("Target model", fontsize=20)
ax.set_xlabel("News story", fontsize=20)
cb_ = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb_.set_label("Mean Favorability $Y$", fontsize=18)
cb_.ax.tick_params(labelsize=15)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig("outputs/heatmap_blocked.png", dpi=150, bbox_inches="tight")
os.makedirs("outputs/pdf", exist_ok=True)
plt.savefig("outputs/pdf/heatmap_blocked.pdf", bbox_inches="tight")
print(
    "Wrote outputs/heatmap_blocked.png and outputs/pdf/heatmap_blocked.pdf",
    "| matrix",
    M.shape,
    "| NaN cells:",
    int(np.isnan(M).sum()),
)
