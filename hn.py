"""Fetch Hacker News front-page stories and save each day's snapshot as a CSV.

This script uses the Algolia HN archive API only (no Firebase/live path).

Usage:
    python hn.py
    python hn.py --start 2024-04-29 --end 2024-05-07

No API key required.
"""

import argparse
import csv
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

DATA_DIR = Path("data/hn")

ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
DEFAULT_LIMIT = 30
DEFAULT_START = date(2024, 4, 29)
DEFAULT_END = date(2024, 4, 29)

FIELDS = [
    "id",
    "time",
    "by",
    "title",
    "url",
    "score",
    "descendants",
    "type",
    "text",
]


# ---------------------------------------------------------------------------
# Historical (Algolia) — past dates
# ---------------------------------------------------------------------------


def fetch_algolia_day(day: date, limit: int, retries: int = 5) -> list[dict]:
    """Return up to `limit` top stories for `day` using the Algolia search API."""
    start_ts = int(
        datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    )
    end_ts = start_ts + 86400

    url = f"{ALGOLIA_BASE}/search"
    params = {
        "tags": "story",
        "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts},points>0",
        "hitsPerPage": limit,
        "attributesToRetrieve": "objectID,created_at,author,title,url,points,num_comments,_tags,story_text",
    }
    wait = 4
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                if attempt == retries:
                    r.raise_for_status()
                retry_after = int(r.headers.get("Retry-After", wait))
                print(f"rate-limited, waiting {retry_after}s ...", end=" ", flush=True)
                time.sleep(retry_after)
                wait *= 2
                continue
            r.raise_for_status()
            hits = r.json().get("hits", [])
            return [flatten_algolia_hit(h) for h in hits]
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(wait)
            wait *= 2
    return []


def flatten_algolia_hit(hit: dict) -> dict:
    tags = hit.get("_tags") or []
    item_type = next((t for t in tags if t in ("story", "job", "poll")), "story")
    return {
        "id": hit.get("objectID"),
        "time": hit.get("created_at"),
        "by": hit.get("author"),
        "title": hit.get("title"),
        "url": hit.get("url"),
        "score": hit.get("points"),
        "descendants": hit.get("num_comments"),
        "type": item_type,
        "text": hit.get("story_text"),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def save_csv(items: list[dict], path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(items)


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got: {value!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Hacker News stories (Algolia archive)."
    )
    parser.add_argument(
        "--start",
        metavar="YYYY-MM-DD",
        type=parse_date,
        default=DEFAULT_START,
        help=f"First date to fetch (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        metavar="YYYY-MM-DD",
        type=parse_date,
        default=DEFAULT_END,
        help=f"Last date to fetch (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Stories per day (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between Algolia requests (default: 0.5)",
    )
    args = parser.parse_args()

    if args.start > args.end:
        parser.error(f"--start {args.start} is after --end {args.end}.")

    days = list(date_range(args.start, args.end))
    print(f"Fetching {len(days)} day(s): {args.start} → {args.end}\n")
    total, failed = 0, []

    for i, day in enumerate(days):
        out_path = DATA_DIR / f"hn_{day}.csv"
        label = str(day)
        if out_path.exists():
            print(f"[{i+1}/{len(days)}] {label} already saved, skipping.")
            continue

        print(f"[{i+1}/{len(days)}] {label} ...", end=" ", flush=True)
        try:
            items = fetch_algolia_day(day, args.limit)
            save_csv(items, out_path)
            print(f"{len(items):>3} stories  →  {out_path}")
            total += len(items)
        except Exception as e:
            print(f"error — skipped ({e})")
            failed.append((label, str(e)))

        if i < len(days) - 1:
            time.sleep(args.delay)

    print(f"\nDone. {total:,} stories saved across {len(days) - len(failed)} day(s).")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for label, reason in failed:
            print(f"  {label}: {reason}")


if __name__ == "__main__":
    main()
