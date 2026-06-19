"""
Analyse NYT and Hacker News CSVs and count how many articles mention each
AI company.

NYT search fields:  headline, abstract, lead_paragraph, keywords
HN  search fields:  title, text
An article is counted once per company even if the name appears multiple times.

Usage:
    python company_coverage.py [--nyt-dir data/nyt] [--hn-dir data/hn]
                               [--sort {count,name}] [--show-samples N]
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Company definitions
# Each entry: display name → list of regex patterns (case-insensitive, word-boundary)
# ---------------------------------------------------------------------------
COMPANIES: dict[str, list[str]] = {
    "Google": [r"Google"],
    "Anthropic": [r"Anthropic"],
    "OpenAI": [r"OpenAI"],
    "xAI": [r"xAI", r"x\.ai"],
    "DeepSeek": [r"DeepSeek"],
    "Qwen (Alibaba)": [r"Qwen", r"Alibaba"],
    "Meta": [r"Meta(?:\s+(?:AI|Platforms?))?", r"Facebook\s+AI"],
    "Mistral": [r"Mistral"],
    "Microsoft": [r"Microsoft"],
    "Nous Research": [r"Nous\s+Research"],
    "Moonshot AI": [r"Moonshot\s*AI", r"\bKimi\b"],
    "ZAI / 01.AI": [r"\b01\.?AI\b", r"\bZAI\b", r"Zero\s+One\s+AI", r"\bGLM\b"],
    "Xiaomi": [r"Xiaomi"],
    "TNG Technology": [r"TNG\s+Technology"],
    "Arcee": [r"\bArcee\b"],
    "StepFun": [r"StepFun", r"Step\s+Fun"],
    "Tencent": [r"Tencent"],
}

NYT_FIELDS = ["headline", "abstract", "lead_paragraph", "keywords"]
HN_FIELDS = ["title", "text"]


def build_pattern(patterns: list[str]) -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(r"\b(?:" + combined + r")\b", re.IGNORECASE)


def row_text(row: dict, fields: list[str]) -> str:
    return " ".join(row.get(f, "") or "" for f in fields)


def nyt_year_month(path: Path) -> tuple[int, int] | None:
    """Extract (year, month) from filenames like 2026_03.csv."""
    parts = path.stem.split("_")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None


def hn_year_month(path: Path) -> tuple[int, int] | None:
    """Extract (year, month) from filenames like hn_2026-03-15.csv."""
    stem = path.stem  # e.g. "hn_2026-03-15"
    if not stem.startswith("hn_"):
        return None
    date_part = stem[3:]  # "2026-03-15"
    parts = date_part.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None


def scan_source(
    files: list[Path],
    fields: list[str],
    id_field: str,
    date_field: str,
    date_slice: slice,
    headline_field: str,
    patterns: dict[str, re.Pattern],
    show_samples: int,
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, int]],
    dict[str, list[tuple[str, str]]],
    int,
]:
    """Scan a list of CSV files and return (matches, monthly, samples, total)."""
    matches: dict[str, set[str]] = {name: set() for name in COMPANIES}
    monthly: dict[str, dict[str, int]] = {name: defaultdict(int) for name in COMPANIES}
    samples: dict[str, list[tuple[str, str]]] = {name: [] for name in COMPANIES}
    total = 0
    for path in files:
        ym = nyt_year_month(path) or hn_year_month(path)
        month_label = f"{ym[0]}-{ym[1]:02d}" if ym else "unknown"
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                text = row_text(row, fields)
                article_id = row.get(id_field, "")
                for name, pat in patterns.items():
                    if pat.search(text):
                        matches[name].add(article_id)
                        monthly[name][month_label] += 1
                        if show_samples and len(samples[name]) < show_samples:
                            samples[name].append(
                                (
                                    (row.get(date_field, "") or "")[date_slice],
                                    row.get(headline_field, "") or "",
                                )
                            )
    return matches, monthly, samples, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count NYT + HN articles mentioning AI companies."
    )
    parser.add_argument(
        "--nyt-dir",
        default="data/nyt",
        metavar="DIR",
        help="NYT monthly CSVs directory (default: data/nyt)",
    )
    parser.add_argument(
        "--hn-dir",
        default="data/hn",
        metavar="DIR",
        help="HN daily CSVs directory (default: data/hn)",
    )
    parser.add_argument(
        "--sort",
        choices=["count", "name"],
        default="count",
        help="Sort by total count (desc) or name (asc) (default: count)",
    )
    parser.add_argument(
        "--show-samples",
        metavar="N",
        type=int,
        default=0,
        help="Sample headlines per company per source (0 to disable)",
    )
    args = parser.parse_args()

    patterns = {name: build_pattern(pats) for name, pats in COMPANIES.items()}

    # ── NYT ─────────────────────────────────────────────────────────────────
    nyt_files: list[Path] = []
    nyt_dir = Path(args.nyt_dir)
    if nyt_dir.is_dir():
        nyt_files = sorted(
            (p for p in nyt_dir.glob("*.csv") if nyt_year_month(p)),
            key=lambda p: nyt_year_month(p),
        )
    else:
        print(f"Warning: NYT directory not found: {nyt_dir}", file=sys.stderr)

    # ── HN ──────────────────────────────────────────────────────────────────
    hn_files: list[Path] = []
    hn_dir = Path(args.hn_dir)
    if hn_dir.is_dir():
        hn_files = sorted(
            (p for p in hn_dir.glob("hn_*.csv") if hn_year_month(p)),
            key=lambda p: hn_year_month(p),
        )
    else:
        print(f"Warning: HN directory not found: {hn_dir}", file=sys.stderr)

    if not nyt_files and not hn_files:
        sys.exit("No files found in either source.")

    # ── Scan ────────────────────────────────────────────────────────────────
    nyt_matches, nyt_monthly, nyt_samples, nyt_total = scan_source(
        nyt_files,
        NYT_FIELDS,
        "_id",
        "pub_date",
        slice(0, 10),
        "headline",
        patterns,
        args.show_samples,
    )
    hn_matches, hn_monthly, hn_samples, hn_total = scan_source(
        hn_files,
        HN_FIELDS,
        "id",
        "time",
        slice(0, 10),
        "title",
        patterns,
        args.show_samples,
    )

    # ── Print ────────────────────────────────────────────────────────────────
    date_range = ""
    all_months = sorted(
        {
            f"{ym[0]}-{ym[1]:02d}"
            for p in nyt_files + hn_files
            for ym in [(nyt_year_month(p) or hn_year_month(p))]
            if ym
        }
    )
    if all_months:
        date_range = f"{all_months[0]} → {all_months[-1]}"

    print(f"NYT : {len(nyt_files):>4} file(s)  |  {nyt_total:>7,} articles")
    print(f"HN  : {len(hn_files):>4} file(s)  |  {hn_total:>7,} stories")
    if date_range:
        print(f"Range: {date_range}")
    print()

    # Combine matches for total count and sorting
    combined_totals = {
        name: len(nyt_matches[name]) + len(hn_matches[name]) for name in COMPANIES
    }

    items = list(COMPANIES.keys())
    if args.sort == "count":
        items.sort(key=lambda n: combined_totals[n], reverse=True)
    else:
        items.sort(key=str.lower)

    col = 20
    print(f"{'Company':<{col}} {'NYT':>6}  {'HN':>6}  {'Total':>7}")
    print("-" * 45)

    for name in items:
        nyt_n = len(nyt_matches[name])
        hn_n = len(hn_matches[name])
        total = combined_totals[name]

        print(f"{name:<{col}} {nyt_n:>6,}  {hn_n:>6,}  {total:>7,}")

        if args.show_samples:
            for date_str, headline in nyt_samples[name]:
                truncated = (headline[:80] + "…") if len(headline) > 80 else headline
                print(f"    [NYT] {date_str}  {truncated}")
            for date_str, headline in hn_samples[name]:
                truncated = (headline[:80] + "…") if len(headline) > 80 else headline
                print(f"    [HN]  {date_str}  {truncated}")
        print()


if __name__ == "__main__":
    main()
