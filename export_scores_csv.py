#!/usr/bin/env python3
"""Export judge scores to the analysis CSV consumed by hypothesis_testing.R.

This is intentionally offline-only: it reads existing score records, joins
story dates from data/stories/top_per_company.json when needed, and writes a
CSV with the column names expected by the R analysis script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SCORES_DIR = Path("data/scores")
DEFAULT_STORIES = Path("data/stories/top_per_company.json")
DEFAULT_FALLBACK_STORIES = Path("data/judge/combined.json")
DEFAULT_OUT = Path("data/scores/scores.csv")

CSV_FIELDS = [
    "score",
    "company_id",
    "story_id",
    "story_company_id",
    "judge_model_id",
    "target_model_id",
    "prompt_template_id",
    "repetition_id",
    "story_date",
]


def load_story_file(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    stories = json.loads(path.read_text(encoding="utf-8"))
    return {str(s["id"]): s for s in stories}


def load_stories(primary: Path, fallback: Path | None) -> dict[str, dict]:
    stories = load_story_file(fallback) if fallback is not None else {}
    stories.update(load_story_file(primary))
    return stories


def iter_score_records(scores_dir: Path, scores_jsonl: Path | None) -> list[dict]:
    if scores_jsonl is None:
        scores_jsonl = scores_dir / "scores.jsonl"

    records: list[dict] = []
    if scores_jsonl.exists():
        for line in scores_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    for path in sorted(scores_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def analysis_row(rec: dict, stories: dict[str, dict]) -> dict:
    story_id = str(rec.get("story_id", ""))
    story = stories.get(story_id, {})
    story_date = (
        rec.get("story_date")
        or rec.get("date")
        or story.get("date")
        or story.get("story_date")
        or ""
    )
    story_company = (
        rec.get("story_company") or story.get("story_company") or story.get("company")
    )

    return {
        "score": rec.get("score"),
        "company_id": rec.get("target_company"),
        "story_id": story_id,
        "story_company_id": story_company,
        "judge_model_id": rec.get("judge_model"),
        "target_model_id": rec.get("target_model"),
        "prompt_template_id": rec.get("template_id"),
        "repetition_id": rec.get("rep"),
        "story_date": story_date,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--scores-jsonl", type=Path, default=None)
    parser.add_argument("--stories-file", type=Path, default=DEFAULT_STORIES)
    parser.add_argument(
        "--fallback-stories-file", type=Path, default=DEFAULT_FALLBACK_STORIES
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    stories = load_stories(args.stories_file, args.fallback_stories_file)
    records = iter_score_records(args.scores_dir, args.scores_jsonl)
    rows = [analysis_row(rec, stories) for rec in records]

    missing_dates = [r["story_id"] for r in rows if not r["story_date"]]
    if missing_dates:
        sample = ", ".join(missing_dates[:10])
        raise SystemExit(
            f"Missing story dates for {len(missing_dates)} score rows; sample: {sample}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows to {args.out}")


if __name__ == "__main__":
    main()
