"""
Fetch NYT Archive API articles month-by-month and save each month as a CSV.

Usage:
    python nyt.py [--start YYYY-MM] [--end YYYY-MM] [--api-key KEY]

Defaults to a fixed range for reproducibility: 2024-04 → 2026-04.
Set NYT_API_KEY env var or pass --api-key.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data/nyt")

FIELDS = [
    "_id",
    "pub_date",
    "headline",
    "abstract",
    "lead_paragraph",
    "web_url",
    "section_name",
    "subsection_name",
    "type_of_material",
    "word_count",
    "byline",
    "keywords",
]


def fetch_month(year: int, month: int, api_key: str, retries: int = 5) -> list[dict]:
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
    wait = 12  # seconds between retries on 429
    for attempt in range(1, retries + 1):
        r = requests.get(url, params={"api-key": api_key}, timeout=30)
        if r.status_code == 429:
            if attempt == retries:
                r.raise_for_status()
            retry_after = int(r.headers.get("Retry-After", wait))
            print(
                f"rate-limited, waiting {retry_after}s (attempt {attempt}/{retries}) ...",
                end=" ",
                flush=True,
            )
            time.sleep(retry_after)
            wait *= 2
            continue
        r.raise_for_status()
        return r.json()["response"]["docs"]
    r.raise_for_status()  # unreachable, satisfies type checkers


def flatten_doc(doc: dict) -> dict:
    return {
        "_id": doc.get("_id"),
        "pub_date": doc.get("pub_date"),
        "headline": (doc.get("headline") or {}).get("main"),
        "abstract": doc.get("abstract"),
        "lead_paragraph": doc.get("lead_paragraph"),
        "web_url": doc.get("web_url"),
        "section_name": doc.get("section_name"),
        "subsection_name": doc.get("subsection_name"),
        "type_of_material": doc.get("type_of_material"),
        "word_count": doc.get("word_count"),
        "byline": (doc.get("byline") or {}).get("original"),
        "keywords": "|".join(kw.get("value", "") for kw in (doc.get("keywords") or [])),
    }


def save_csv(docs: list[dict], year: int, month: int) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{year}_{month:02d}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for doc in docs:
            writer.writerow(flatten_doc(doc))
    return path


def month_range(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def parse_ym(value: str) -> tuple[int, int]:
    try:
        year, month = value.split("-")
        return int(year), int(month)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM, got: {value!r}")


def main() -> None:
    default_start = "2024-04"
    default_end = "2026-04"

    parser = argparse.ArgumentParser(
        description="Fetch NYT Archive API data by month range."
    )
    parser.add_argument(
        "--start",
        default=default_start,
        metavar="YYYY-MM",
        help=f"First month to fetch (default: {default_start})",
    )
    parser.add_argument(
        "--end",
        default=default_end,
        metavar="YYYY-MM",
        help=f"Last month to fetch (default: {default_end})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NYT_API_KEY", ""),
        help="NYT API key (or set NYT_API_KEY env var)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between requests (default: 1.0)",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Provide --api-key or set the NYT_API_KEY environment variable.")

    start_year, start_month = parse_ym(args.start)
    end_year, end_month = parse_ym(args.end)

    if (start_year, start_month) > (end_year, end_month):
        parser.error(f"--start {args.start} is after --end {args.end}.")

    months = list(month_range(start_year, start_month, end_year, end_month))
    print(f"Fetching {len(months)} month(s): {args.start} → {args.end}\n")

    total = 0
    failed = []

    for i, (year, month) in enumerate(months):
        label = f"{year}-{month:02d}"
        out_path = DATA_DIR / f"{year}_{month:02d}.csv"
        if out_path.exists():
            print(f"[{i+1}/{len(months)}] {label} already saved, skipping.")
            continue

        print(f"[{i+1}/{len(months)}] {label} ...", end=" ", flush=True)
        try:
            docs = fetch_month(year, month, args.api_key)
            path = save_csv(docs, year, month)
            print(f"{len(docs):>6,} articles  →  {path}")
            total += len(docs)
        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code} — skipped")
            failed.append((label, str(e)))
        except requests.RequestException as e:
            print(f"request error — skipped ({e})")
            failed.append((label, str(e)))
        except (KeyError, ValueError, TypeError) as e:
            print(f"parse error — skipped ({e})")
            failed.append((label, str(e)))

        if i < len(months) - 1:
            time.sleep(args.delay)

    print(
        f"\nDone. {total:,} articles saved across {len(months) - len(failed)} month(s)."
    )
    if failed:
        print(f"\nFailed months ({len(failed)}):")
        for label, reason in failed:
            print(f"  {label}: {reason}")


if __name__ == "__main__":
    main()
