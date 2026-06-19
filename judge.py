"""
LLM judge that scores NYT articles and HN stories for relevance to AI companies,
reputational risk, and political controversy using OpenRouter.

Usage:
    python judge.py [--nyt-dir data/nyt] [--hn-dir data/hn]
                    [--start YYYY-MM] [--end YYYY-MM]
                    [--model MODEL] [--workers N] [--batch-size N]

Scores (0.0–10.0) on three axes:
  1. Relevance to target AI companies
  2. Reputational embarrassment / harm
  3. Controversial or politically charged content

Saves per-article JSON to data/judge/ and a combined results file.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

TARGET_COMPANIES = [
    "Google",
    "Anthropic",
    "OpenAI",
    "xAI",
    "Deepseek",
    "Alibaba",
    "Meta",
    "Mistral",
    "Moonshot",
    "Z.ai",
]

# ── NYT: tag-based filter ────────────────────────────────────────────────────
# Match against the pipe-separated "keywords" field in the NYT CSV.
NYT_AI_TAGS: set[str] = {
    "Artificial Intelligence",
    "OpenAI Labs",
    "Google Inc",
    "Meta Platforms Inc",
    "Mistral AI SAS",
    "ChatGPT",
    "Anthropic AI LLC",
    "Robots and Robotics",
    "DeepSeek Artificial Intelligence Co Ltd",
}


def _nyt_matches(row: dict) -> bool:
    tags = {t.strip() for t in (row.get("keywords") or "").split("|") if t.strip()}
    return bool(tags & NYT_AI_TAGS)


# ── HN: keyword-based filter ─────────────────────────────────────────────────
# Match against title + text fields.
_HN_PATTERNS: list[str] = [
    # AI / tech general
    r"\bA\.?I\.?\b",
    r"\bartificial\s+intelligence\b",
    r"\brobot(s|ics)?\b",
    r"\bagent(ic)?\b",
    r"\bvibe\s+coding\b",
    # Company / model names
    r"\bGoogle\b",
    r"\bAnthropic\b",
    r"\bOpenAI\b",
    r"\bxAI\b",
    r"\bx\.ai\b",
    r"\bDeepSeek\b",
    r"\bAlibaba\b",
    r"\bQwen\b",
    r"\bMeta\b",
    r"\bMistral\b",
    r"\bMoonshot\b",
    r"\bKimi\b",
    r"\b01\.?AI\b",
    r"\bZ\.ai\b",
    r"\bZero\s+One\s+AI\b",
    r"\bClaude\b",
    r"\bGPT\b",
    r"\bChatGPT\b",
    r"\bGemini\b",
    r"\bGrok\b",
    r"\bLlama\b",
    r"\bSora\b",
    r"\bDall-?E\b",
    r"\bOpenClaw\b",
    # Reputational / controversy signals
    r"\bwhistle\s*blow(er|ing)?\b",
    r"\bscandal\b",
    r"\bleak(s|ed|ing)?\b",
    r"\bapolog(y|ies|ize[sd]?|ising|izing)\b",
    r"\bsuspend(ed|s|ing)?\b",
    # Key people
    r"\bAltman\b",
    r"\bAmodei\b",
    r"\bZuckerberg\b",
    r"\bBrin\b",
    r"\bPage\b",
    r"\bPichai\b",
    r"\bMensch\b",
    r"\bWenfeng\b",
    r"\bMusk\b",
    r"\bZhilin\b",
    r"\bJie\b",
]

_HN_RE = re.compile(
    r"(?:" + "|".join(_HN_PATTERNS) + r")",
    re.IGNORECASE,
)


def _hn_matches(row: dict) -> bool:
    text = " ".join(row.get(f, "") or "" for f in ["title", "text"])
    return bool(_HN_RE.search(text))


DATA_DIR = Path("data/judge")
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
OR_BASE = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """\
You are a media-analysis judge. Below is an article or story with metadata.
Score it on three dimensions (each 0.0 to 10.0, one decimal place):

1. **relevance**: How relevant is this article to any of these AI companies:
   Google, Anthropic, OpenAI, xAI, Deepseek, Alibaba, Meta, Mistral, Moonshot, Z.ai.
   0.0 = not relevant at all; 10.0 = the article is primarily about one of these companies.

2. **reputational_risk**: Could this content be embarrassing or harmful to the company's
   reputation? 0.0 = likely to be benign/positive coverage; 10.0 = likely to be severely damaging.

3. **controversy**: Is the content controversial or politically charged?
   0.0 = likely to be neutral/factual; 10.0 = likely to be extremely controversial or politically polarizing.

Also identify:
- **company**: The single company (from the list above) that the article is MOST relevant to.
  Use "NA" if none clearly apply or "Multiple" if multiple companies are discussed.
- **rationale**: A brief (1-2 sentence) explanation of your scores.

Respond ONLY with valid JSON matching the requested schema."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number"},
        "reputational_risk": {"type": "number"},
        "controversy": {"type": "number"},
        "company": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "relevance",
        "reputational_risk",
        "controversy",
        "company",
        "rationale",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# File discovery helpers (shared logic with company_coverage.py)
# ---------------------------------------------------------------------------


def parse_ym(value: str) -> tuple[int, int]:
    try:
        y, m = value.split("-")
        return int(y), int(m)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM, got: {value!r}")


def nyt_year_month(path: Path) -> tuple[int, int] | None:
    parts = path.stem.split("_")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None


def hn_year_month(path: Path) -> tuple[int, int] | None:
    stem = path.stem
    if not stem.startswith("hn_"):
        return None
    parts = stem[3:].split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    return None


def collect_files(
    directory: Path,
    glob: str,
    ym_fn,
    start=None,
    end=None,
) -> list[Path]:
    if not directory.is_dir():
        print(f"Warning: directory not found: {directory}", file=sys.stderr)
        return []
    files = sorted(
        (p for p in directory.glob(glob) if ym_fn(p)),
        key=lambda p: ym_fn(p),
    )
    if start:
        files = [p for p in files if ym_fn(p) >= start]
    if end:
        files = [p for p in files if ym_fn(p) <= end]
    return files


# ---------------------------------------------------------------------------
# Article loading
# ---------------------------------------------------------------------------


def load_nyt_articles(files: list[Path]) -> list[dict]:
    articles = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not _nyt_matches(row):
                    continue
                articles.append(
                    {
                        "source": "nyt",
                        "id": row.get("_id", ""),
                        "date": (row.get("pub_date", "") or "")[:10],
                        "headline": row.get("headline", ""),
                        "abstract": row.get("abstract", ""),
                        "lead_paragraph": row.get("lead_paragraph", ""),
                        "url": row.get("web_url", ""),
                        "section": row.get("section_name", ""),
                        "subsection": row.get("subsection_name", ""),
                        "type": row.get("type_of_material", ""),
                        "word_count": row.get("word_count", ""),
                        "byline": row.get("byline", ""),
                        "keywords": row.get("keywords", ""),
                    }
                )
    return articles


def load_hn_stories(files: list[Path]) -> list[dict]:
    stories = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not _hn_matches(row):
                    continue
                stories.append(
                    {
                        "source": "hn",
                        "id": row.get("id", ""),
                        "date": (row.get("time", "") or "")[:10],
                        "headline": row.get("title", ""),
                        "abstract": row.get("text", ""),
                        "lead_paragraph": "",
                        "url": row.get("url", ""),
                        "section": "",
                        "subsection": "",
                        "type": row.get("type", ""),
                        "word_count": "",
                        "byline": row.get("by", ""),
                        "keywords": "",
                        "score": row.get("score", ""),
                        "comments": row.get("descendants", ""),
                    }
                )
    return stories


# ---------------------------------------------------------------------------
# LLM judge via OpenRouter
# ---------------------------------------------------------------------------


def build_user_prompt(article: dict) -> str:
    parts = [f"Source: {article['source'].upper()}"]
    if article.get("date"):
        parts.append(f"Date: {article['date']}")
    if article.get("headline"):
        parts.append(f"Headline: {article['headline']}")
    if article.get("abstract"):
        parts.append(f"Abstract: {article['abstract']}")
    if article.get("lead_paragraph"):
        parts.append(f"Lead paragraph: {article['lead_paragraph']}")
    if article.get("url"):
        parts.append(f"URL: {article['url']}")
    if article.get("section"):
        parts.append(f"Section: {article['section']}")
    if article.get("subsection"):
        parts.append(f"Subsection: {article['subsection']}")
    if article.get("type"):
        parts.append(f"Type: {article['type']}")
    if article.get("word_count"):
        parts.append(f"Word count: {article['word_count']}")
    if article.get("byline"):
        parts.append(f"Byline: {article['byline']}")
    if article.get("keywords"):
        parts.append(f"Keywords: {article['keywords']}")
    if article.get("score"):
        parts.append(f"HN score: {article['score']}")
    if article.get("comments"):
        parts.append(f"HN comments: {article['comments']}")
    return "\n".join(parts)


def call_judge(
    article: dict,
    api_key: str,
    model: str,
    retries: int = 3,
) -> dict:
    """Call OpenRouter and return the parsed judge response (or error dict)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/partiality",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(article)},
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_scores",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OR_BASE, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2**attempt))
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2**attempt)

    return {
        "relevance": -1,
        "reputational_risk": -1,
        "controversy": -1,
        "company": "ERROR",
        "rationale": f"API call failed: {last_err}",
    }


def result_path(article: dict) -> Path:
    safe_id = re.sub(r"[^\w\-]", "_", article["id"])
    return DATA_DIR / f"{article['source']}_{safe_id}.json"


def judge_article(article: dict, api_key: str, model: str) -> dict:
    """Judge a single article; return merged result dict."""
    out = result_path(article)
    if out.exists():
        try:
            return json.loads(out.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    scores = call_judge(article, api_key, model)

    result = {
        "source": article["source"],
        "id": article["id"],
        "date": article.get("date", ""),
        "headline": article.get("headline", ""),
        "url": article.get("url", ""),
        **scores,
    }

    try:
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as e:
        print(f"  Warning: could not save {out}: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM judge: score articles for AI-company relevance, risk & controversy."
    )
    parser.add_argument("--nyt-dir", default="data/nyt", metavar="DIR")
    parser.add_argument("--hn-dir", default="data/hn", metavar="DIR")
    parser.add_argument("--start", metavar="YYYY-MM", type=parse_ym, default=None)
    parser.add_argument("--end", metavar="YYYY-MM", type=parse_ym, default=None)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel API workers (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Save combined results every N articles (default: 50)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only judge the first N pending articles (useful for test runs)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print cost estimate and exit without calling the API",
    )
    args = parser.parse_args()

    if not args.api_key and not args.estimate:
        parser.error(
            "Provide --api-key or set the OPENROUTER_API_KEY environment variable."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load articles ──────────────────────────────────────────────────────
    nyt_files = collect_files(
        Path(args.nyt_dir),
        "*.csv",
        nyt_year_month,
        args.start,
        args.end,
    )
    hn_files = collect_files(
        Path(args.hn_dir),
        "hn_*.csv",
        hn_year_month,
        args.start,
        args.end,
    )

    print("Loading articles (NYT: AI tag filter  |  HN: keyword filter)...")
    nyt_articles = load_nyt_articles(nyt_files)
    hn_stories = load_hn_stories(hn_files)
    articles = nyt_articles + hn_stories

    if not articles:
        sys.exit("No matching articles found.")

    already = sum(1 for a in articles if result_path(a).exists())
    pending = len(articles) - already
    print(
        f"  NYT: {len(nyt_articles):,}  |  HN: {len(hn_stories):,}  |  total: {len(articles):,}"
    )
    print(f"  Already judged: {already:,}  |  Pending: {pending:,}")

    # ── Cost estimate ──────────────────────────────────────────────────────
    # gemini-3.1-pro-preview: $2/M input, $12/M output
    INPUT_COST_PER_M = 2.00
    OUTPUT_COST_PER_M = 12.00
    # Rough token counts: system (~350) + article fields (~500 avg) = ~850 input; ~120 output
    EST_INPUT_TOKENS = 850
    EST_OUTPUT_TOKENS = 120
    to_estimate = pending if args.limit is None else min(pending, args.limit)
    est_input_cost = to_estimate * EST_INPUT_TOKENS / 1_000_000 * INPUT_COST_PER_M
    est_output_cost = to_estimate * EST_OUTPUT_TOKENS / 1_000_000 * OUTPUT_COST_PER_M
    est_total = est_input_cost + est_output_cost
    print(f"\nCost estimate ({args.model}):")
    print(f"  Articles to judge : {to_estimate:,}")
    print(
        f"  Est. input tokens : {to_estimate * EST_INPUT_TOKENS:,}  →  ${est_input_cost:.3f}"
    )
    print(
        f"  Est. output tokens: {to_estimate * EST_OUTPUT_TOKENS:,}  →  ${est_output_cost:.3f}"
    )
    print(
        f"  Est. total cost   : ${est_total:.3f}  (at ${INPUT_COST_PER_M}/M in, ${OUTPUT_COST_PER_M}/M out)"
    )

    if args.estimate:
        return

    if pending == 0:
        print("Nothing new to judge.")
    else:
        to_judge = pending if args.limit is None else min(pending, args.limit)
        print(
            f"\nJudging {to_judge:,} articles with {args.workers} workers ({args.model})...\n"
        )

        done = 0
        errors = 0
        results: list[dict] = []

        pending_articles = [a for a in articles if not result_path(a).exists()]
        if args.limit is not None:
            pending_articles = pending_articles[: args.limit]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(judge_article, art, args.api_key, args.model): art
                for art in pending_articles
            }
            for future in as_completed(futures):
                art = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    is_err = result.get("company") == "ERROR"
                    if is_err:
                        errors += 1
                except Exception as e:
                    errors += 1
                    result = None
                    print(f"  Unexpected error for {art['id']}: {e}", file=sys.stderr)

                done += 1
                if done % 10 == 0 or done == len(futures):
                    print(f"  [{done:>5}/{len(futures)}]  errors: {errors}", flush=True)

                if done % args.batch_size == 0:
                    _save_combined(articles)

    # ── Final combined save ────────────────────────────────────────────────
    all_results = _save_combined(articles)
    print(f"\nDone. {len(all_results):,} total judged results in {DATA_DIR}/")
    _print_summary(all_results)


def _save_combined(articles: list[dict]) -> list[dict]:
    """Reload all per-article JSONs and write a single combined file."""
    all_results = []
    for art in articles:
        p = result_path(art)
        if p.exists():
            try:
                all_results.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
    combined = DATA_DIR / "combined.json"
    combined.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return all_results


def _print_summary(results: list[dict]) -> None:
    """Print a compact summary table."""
    if not results:
        return
    valid = [r for r in results if r.get("company") != "ERROR"]

    by_company: dict[str, list[dict]] = {}
    for r in valid:
        c = r.get("company", "NA")
        by_company.setdefault(c, []).append(r)

    print(
        f"\n{'Company':<14} {'Count':>5}  {'Avg Rel':>7}  {'Avg Risk':>8}  {'Avg Contr':>9}"
    )
    print("-" * 50)
    for company in sorted(by_company, key=lambda c: len(by_company[c]), reverse=True):
        items = by_company[company]
        n = len(items)
        avg_rel = sum(r["relevance"] for r in items) / n
        avg_risk = sum(r["reputational_risk"] for r in items) / n
        avg_con = sum(r["controversy"] for r in items) / n
        print(
            f"{company:<14} {n:>5}  {avg_rel:>7.1f}  {avg_risk:>8.1f}  {avg_con:>9.1f}"
        )

    top_risk = sorted(valid, key=lambda r: r.get("reputational_risk", 0), reverse=True)[
        :5
    ]
    if top_risk:
        print("\nTop 5 reputational risk:")
        for r in top_risk:
            print(
                f"  [{r['company']:>10}] {r['reputational_risk']:>4.1f}  {r['headline'][:70]}"
            )

    top_contr = sorted(valid, key=lambda r: r.get("controversy", 0), reverse=True)[:5]
    if top_contr:
        print("\nTop 5 controversy:")
        for r in top_contr:
            print(
                f"  [{r['company']:>10}] {r['controversy']:>4.1f}  {r['headline'][:70]}"
            )


if __name__ == "__main__":
    main()
