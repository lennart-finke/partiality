#!/usr/bin/env python3
"""Fetch the published dataset and stage the local files the pipeline expects.

Usage:
    python load_from_hf.py                          # download + unpack everything
    python load_from_hf.py --parquet local.parquet  # use a local copy instead
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ID = "lennart-finke/partiality"
REMOTE_PARQUET = "data/train-00000-of-00001.parquet"
DEFAULT_DEST = Path("data/partiality.parquet")
SCORES_DIR = Path("data/scores")
RESPONSES_DIR = Path("data/responses")
ENRICHED_PARQUET = Path("outputs/enriched_scores.parquet")
CUTOFFS = Path("data/model_cutoffs.json")


def download_parquet(dest: Path) -> Path:
    """Download the dataset Parquet from the Hugging Face Hub to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - guidance path
        raise SystemExit(
            "huggingface_hub is required to download the dataset "
            "(`pip install huggingface_hub`), or pass --parquet with a local copy."
        ) from exc
    cached = hf_hub_download(
        repo_id=REPO_ID, filename=REMOTE_PARQUET, repo_type="dataset"
    )
    shutil.copyfile(cached, dest)
    return dest


def _safe_name(response_id: str) -> str:
    """Filesystem-safe basename for a response id (slugs contain no '/')."""
    return response_id.replace("/", "_")


def _clean(value):
    """Convert numpy scalars to plain Python and NaN/NA to None for JSON."""
    import numpy as np

    if isinstance(value, np.generic):
        return None if _isna(value) else value.item()
    if value is None:
        return None
    return None if _isna(value) else value


def _isna(value) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def unpack(parquet: Path) -> int:
    """Unpack the dataset Parquet into the per-record JSON layout + a copy.

    Returns the number of samples written.
    """
    import pandas as pd

    from config import flip_inverted_scores

    df = pd.read_parquet(parquet)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    records = df.to_dict(orient="records")

    def write_one(rec: dict) -> None:
        rid = str(rec.get("response_id", ""))
        name = _safe_name(rid)
        response_path = RESPONSES_DIR / f"{name}.json"
        score_path = SCORES_DIR / f"{name}.json"

        # The target model's response, as query_targets.py would have written it.
        response = {
            "response_id": rid,
            "completion": _clean(rec.get("completion")),
            "target_company": _clean(rec.get("target_company")),
            "target_model": _clean(rec.get("target_model")),
            "target_model_name": _clean(rec.get("target_model_name")),
            "story_id": _clean(rec.get("story_id")),
            "story_company": _clean(rec.get("story_company")),
            "story_headline": _clean(rec.get("story_headline")),
            "story_url": _clean(rec.get("story_url")),
            "story_date": _clean(rec.get("story_date")),
            "template_id": _clean(rec.get("template_id")),
            "rep": _clean(rec.get("rep")),
            "error": _clean(rec.get("error")),
        }
        response_path.write_text(
            json.dumps(response, ensure_ascii=False), encoding="utf-8"
        )

        # The judge's score record, as query_judges.py would have written it:
        # RAW (un-recoded) so the downstream flip_inverted_scores lands correct.
        score = {
            "response_id": rid,
            "story_id": _clean(rec.get("story_id")),
            "story_headline": _clean(rec.get("story_headline")),
            "story_company": _clean(rec.get("story_company")),
            "story_date": _clean(rec.get("story_date")),
            "target_company": _clean(rec.get("target_company")),
            "target_model": _clean(rec.get("target_model")),
            "judge_model": _clean(rec.get("judge_model")),
            "template_id": _clean(rec.get("template_id")),
            "rep": _clean(rec.get("rep")),
            "language_score": _clean(rec.get("language_score")),
            "content_score": _clean(rec.get("content_score")),
            "completeness_score": _clean(rec.get("completeness_score")),
            "score": _clean(rec.get("score")),
            "reasoning": _clean(rec.get("judge_reasoning")),
            "response_file": str(response_path),
        }
        score = flip_inverted_scores(score)  # corrected -> raw (self-inverse)
        score_path.write_text(json.dumps(score, ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(write_one, records))

    # per_model_analysis.py reads the enriched (corrected) Parquet directly.
    ENRICHED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parquet, ENRICHED_PARQUET)

    return len(records)


def check_model_cutoffs(path: Path) -> None:
    """Sanity-check the curated per-model knowledge-cutoff file.

    This file is a checked-in input (not generated). The knowledge-cutoff
    sensitivity (``sensitivity_analyses.py`` / ``final_results.py``) maps each
    target model to a ``YYYY-MM-DD`` cutoff; keys prefixed with ``_`` are
    documentation and are ignored by the analysis.
    """
    if not path.exists():
        print(
            f"WARNING: {path} not found. The knowledge-cutoff sensitivity "
            "(steps 11b, 12) needs it; without it the contrast degrades to the "
            "primary test."
        )
        return
    cutoffs = json.loads(path.read_text(encoding="utf-8"))
    n = len([k for k in cutoffs if not k.startswith("_")])
    print(f"Found {path} ({n} model cutoff dates)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Use a local Parquet instead of downloading; copied to "
        f"{DEFAULT_DEST}.",
    )
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    if args.parquet is not None:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        if args.parquet.resolve() != args.dest.resolve():
            shutil.copyfile(args.parquet, args.dest)
        print(f"Staged dataset at {args.dest}")
    else:
        print(f"Downloading {REPO_ID}:{REMOTE_PARQUET} ...")
        download_parquet(args.dest)
        print(f"Staged dataset at {args.dest}")

    print("Unpacking into per-record JSON (this writes ~2 files per sample) ...")
    n = unpack(args.dest)
    print(
        f"Wrote {n:,} score records to {SCORES_DIR}/, "
        f"{n:,} responses to {RESPONSES_DIR}/, and copied {ENRICHED_PARQUET}"
    )

    check_model_cutoffs(CUTOFFS)
    print("\nDone. Next: python export_scores_csv.py  (readme step 10)")


if __name__ == "__main__":
    main()
