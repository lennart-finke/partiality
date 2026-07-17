"""
Pick the top N stories per company from data/judge/combined.json.

Selection:
  - filter to results with company == one of the configured companies
  - filter by relevance >= --min-relevance (default 8.0)
  - sort by max(reputational_risk, controversy) descending, then relevance
  - cluster near-duplicate stories (same underlying topic) with an LLM judge
    and keep only one representative per cluster; the cluster input is the
    top min(corpus, max(--cluster-pool, --top-n)) ranked stories so dedup always
    spans the slice you may pick from when --top-n exceeds --cluster-pool
  - take the top --top-n (default 3) per company

Reads:  data/judge/combined.json      (output of judge.py)
Writes: data/stories/top_per_company.json     (one record per (company, story))
        data/stories/clusters/<company>.json  (per-company cluster cache)

Usage:
    python top_stories.py
    python top_stories.py --top-n 3 --min-relevance 9 --companies Anthropic OpenAI
    python top_stories.py --no-cluster   # skip the dedup step
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from config import DEFAULT_COMPANIES

load_dotenv()

JUDGE_COMBINED = Path("data/judge/combined.json")
NYT_DIR = Path("data/nyt")
HN_DIR = Path("data/hn")
OUT_DIR = Path("data/stories")
CLUSTER_DIR = OUT_DIR / "clusters"
MANUAL_CSV_DIR = Path("data")
DEFAULT_MANUAL_CSVS = [
    MANUAL_CSV_DIR / "expert_added_urls.csv",
    MANUAL_CSV_DIR / "manually_added_urls.csv",
]

CLUSTER_MODEL = "google/gemini-3.1-pro-preview"
OR_BASE = "https://openrouter.ai/api/v1/chat/completions"


def _max_risk(r: dict) -> float:
    return max(float(r.get("reputational_risk", 0)), float(r.get("controversy", 0)))


def _load_judge_results(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Judge results not found: {path}. Run judge.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _index_nyt_articles(nyt_dir: Path) -> dict[str, dict]:
    """Index NYT rows by _id so we can attach abstract / lead_paragraph."""
    index: dict[str, dict] = {}
    if not nyt_dir.is_dir():
        return index
    for csv_path in nyt_dir.glob("*.csv"):
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rid = row.get("_id")
                    if rid:
                        index[rid] = row
        except OSError:
            continue
    return index


def _index_hn_stories(hn_dir: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not hn_dir.is_dir():
        return index
    for csv_path in hn_dir.glob("hn_*.csv"):
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rid = row.get("id")
                    if rid:
                        index[rid] = row
        except OSError:
            continue
    return index


def _enrich(story: dict, nyt_idx: dict[str, dict], hn_idx: dict[str, dict]) -> dict:
    """Attach abstract / lead_paragraph / text from the source CSV."""
    enriched = dict(story)
    if story["source"] == "nyt":
        row = nyt_idx.get(story["id"])
        if row:
            enriched["abstract"] = row.get("abstract", "")
            enriched["lead_paragraph"] = row.get("lead_paragraph", "")
            enriched["section"] = row.get("section_name", "")
            enriched["byline"] = row.get("byline", "")
            enriched["keywords"] = row.get("keywords", "")
    elif story["source"] == "hn":
        row = hn_idx.get(story["id"])
        if row:
            enriched["text"] = row.get("text", "")
            enriched["score"] = row.get("score", "")
            enriched["by"] = row.get("by", "")
    return enriched


def _load_manual_csvs(
    csv_paths: list[Path], valid_companies: list[str]
) -> dict[str, list[dict]]:
    """Load manually curated story CSVs and group by canonical company name.

    Each CSV has columns: headline, company, date, url.
    Returns {company: [story_dict, ...]}.
    """
    by_company: dict[str, list[dict]] = {}
    seen_urls: set[str] = set()
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"  Note: manual CSV not found, skipping: {csv_path}")
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                company = row.get("company", "").strip()
                story_id = f"manual_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
                story = {
                    "source": "manual",
                    "id": story_id,
                    "date": (row.get("date") or "").strip(),
                    "headline": (row.get("headline") or "").strip(),
                    "url": url,
                    "relevance": None,
                    "reputational_risk": None,
                    "controversy": None,
                    "company": company,
                    "rationale": "Manually curated story.",
                    "csv_source": csv_path.name,
                }
                by_company.setdefault(company, []).append(story)
        print(f"  {csv_path.name}: loaded {sum(1 for _ in open(csv_path)) - 1} rows")
    return by_company


# ---------------------------------------------------------------------------
# LLM-based clustering of near-duplicate stories
# ---------------------------------------------------------------------------

CLUSTER_SYSTEM_PROMPT = """\
You are a text analysis judge. You will receive a numbered list of news
stories about a single AI company. Your job is to cluster stories that cover
the SAME underlying news event or topic. Different angles, opinion pieces, or
follow-ups on the SAME event belong in the same cluster. Stories about
different events — even if they involve the same company — must be in
DIFFERENT clusters.

For every cluster of two or more stories, also pick one representative: the
single most high-quality and comprehensive story in the cluster (prefer
in-depth news reporting over thin aggregations, op-eds, or social posts;
prefer NYT articles over short HN link posts when content is comparable;
prefer the more recent / definitive version when one supersedes another).

Rules:
- Every story id you output must come from the input list.
- A story id may appear in AT MOST one cluster.
- Singleton stories (no duplicate) should be OMITTED from the output — only
  return clusters of size >= 2.
- The representative id MUST be one of the ids in that cluster.

Respond ONLY with valid JSON matching the requested schema."""

CLUSTER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                    "representative_id": {"type": "string"},
                    "topic": {"type": "string"},
                },
                "required": ["ids", "representative_id", "topic"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["clusters"],
    "additionalProperties": False,
}


def _cluster_story_blurb(sid: str, story: dict) -> str:
    """Compact one-paragraph blurb for an in-prompt story listing."""
    headline = story.get("headline", "") or ""
    date = story.get("date", "") or ""
    source = (story.get("source") or "").upper()
    url = story.get("url", "") or ""
    rationale = story.get("rationale", "") or ""

    body = (
        story.get("abstract") or story.get("lead_paragraph") or story.get("text") or ""
    )
    body = body.replace("\n", " ").strip()
    if len(body) > 400:
        body = body[:400].rstrip() + "..."

    parts = [
        f"[{sid}] ({source}, {date}) {headline}",
        f"  url: {url}" if url else "",
        f"  body: {body}" if body else "",
        f"  judge_rationale: {rationale}" if rationale else "",
    ]
    return "\n".join(p for p in parts if p)


def _build_cluster_prompt(company: str, items: list[tuple[str, dict]]) -> str:
    lines = [
        f"Company: {company}",
        f"Number of stories: {len(items)}",
        "",
        "Stories:",
        "",
    ]
    lines.extend(_cluster_story_blurb(sid, s) for sid, s in items)
    lines += [
        "",
        "Cluster the stories above by underlying topic / event. Only return"
        " clusters of size >= 2. For each cluster, pick the single most"
        " comprehensive, high-quality story as the representative.",
    ]
    return "\n".join(lines)


def _call_cluster_judge(
    company: str,
    items: list[tuple[str, dict]],
    api_key: str,
    model: str,
    retries: int = 3,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/partiality",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
            {"role": "user", "content": _build_cluster_prompt(company, items)},
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "story_clusters",
                "strict": True,
                "schema": CLUSTER_RESPONSE_SCHEMA,
            },
        },
    }

    last_err: str | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OR_BASE, headers=headers, json=payload, timeout=120)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2**attempt))
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError("API returned null content")
            return json.loads(content)
        except (
            requests.RequestException,
            KeyError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2**attempt)

    return {"clusters": [], "error": f"API call failed: {last_err}"}


def _cluster_company_stories(
    company: str,
    pool: list[dict],
    api_key: str,
    model: str,
    cache_dir: Path,
    refresh: bool = False,
) -> dict:
    """Return {sid -> {representative_id, cluster_index, ids, topic}} mapping.

    Stories not in any returned cluster are singletons (mapped to themselves,
    representative_id == sid, cluster_index == None).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_company = company.replace("/", "_").replace(" ", "_")
    cache_path = cache_dir / f"{safe_company}.json"

    items: list[tuple[str, dict]] = [(f"S{i+1:03d}", s) for i, s in enumerate(pool)]
    sid_to_story = {sid: s for sid, s in items}
    pool_signature = [
        {"sid": sid, "source": s.get("source"), "id": s.get("id")} for sid, s in items
    ]

    cached: dict | None = None
    if cache_path.exists() and not refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("pool_signature") != pool_signature:
                cached = None
        except (json.JSONDecodeError, OSError):
            cached = None

    if cached is None:
        if not pool:
            response: dict = {"clusters": []}
        elif not api_key:
            sys.exit(
                f"Cluster cache for {company!r} is missing or stale and no API key is "
                "available. Provide --api-key or set OPENROUTER_API_KEY, or use --no-cluster."
            )
        else:
            print(f"  [cluster] {company}: querying {model} on {len(pool)} stories...")
            response = _call_cluster_judge(company, items, api_key, model)
        cached = {
            "company": company,
            "model": model,
            "pool_signature": pool_signature,
            "response": response,
        }
        try:
            cache_path.write_text(
                json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            print(f"  Warning: could not save {cache_path}: {e}", file=sys.stderr)

    response = cached.get("response", {}) or {}
    clusters = response.get("clusters", []) or []

    sid_info: dict[str, dict] = {}
    seen: set[str] = set()
    for ci, cl in enumerate(clusters):
        ids = [sid for sid in (cl.get("ids") or []) if sid in sid_to_story]
        if len(ids) < 2:
            continue
        rep = cl.get("representative_id")
        if rep not in ids:
            rep = ids[0]
        topic = cl.get("topic", "") or ""
        for sid in ids:
            if sid in seen:
                continue
            seen.add(sid)
            sid_info[sid] = {
                "representative_id": rep,
                "cluster_index": ci,
                "cluster_size": len(ids),
                "cluster_ids": ids,
                "cluster_topic": topic,
            }

    for sid in sid_to_story:
        if sid not in sid_info:
            sid_info[sid] = {
                "representative_id": sid,
                "cluster_index": None,
                "cluster_size": 1,
                "cluster_ids": [sid],
                "cluster_topic": "",
            }

    return {
        "items": items,
        "sid_to_story": sid_to_story,
        "sid_info": sid_info,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Select top-N stories per company.")
    p.add_argument("--judge-file", type=Path, default=JUDGE_COMBINED)
    p.add_argument("--nyt-dir", type=Path, default=NYT_DIR)
    p.add_argument("--hn-dir", type=Path, default=HN_DIR)
    p.add_argument("--out", type=Path, default=OUT_DIR / "top_per_company.json")
    p.add_argument(
        "--txt-out",
        type=Path,
        default=Path("data/articles.txt"),
        help="Path to write the extracted URLs as a plain text file.",
    )
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--min-relevance", type=float, default=7.0)
    p.add_argument(
        "--min-risk-or-controversy",
        type=float,
        default=7.0,
        help="Lower bound on max(rep_risk, controversy). Default 7.",
    )
    p.add_argument("--companies", nargs="+", default=DEFAULT_COMPANIES)
    p.add_argument(
        "--cluster",
        dest="cluster",
        action="store_true",
        default=True,
        help="Cluster near-duplicate stories with an LLM judge (default).",
    )
    p.add_argument(
        "--no-cluster",
        dest="cluster",
        action="store_false",
        help="Skip the LLM clustering / dedup step.",
    )
    p.add_argument(
        "--cluster-pool",
        type=int,
        default=150,
        help="Max stories per company to send to the cluster judge "
        "(default: 150). Raised automatically to --top-n when "
        "that is larger, so dedup spans the ranks you intend to "
        "pick from.",
    )
    p.add_argument(
        "--cluster-model",
        default=CLUSTER_MODEL,
        help=f"OpenRouter model for clustering (default: {CLUSTER_MODEL}).",
    )
    p.add_argument(
        "--cluster-cache",
        type=Path,
        default=CLUSTER_DIR,
        help="Directory to cache per-company cluster JSON.",
    )
    p.add_argument(
        "--refresh-clusters",
        action="store_true",
        help="Ignore the cluster cache and re-query the LLM.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (or set OPENROUTER_API_KEY).",
    )
    p.add_argument(
        "--manual-csvs",
        nargs="*",
        type=Path,
        default=DEFAULT_MANUAL_CSVS,
        help="CSV files with manually curated stories (headline,company,date,url). "
        "These are appended to the output without judge scoring or clustering.",
    )
    p.add_argument(
        "--no-manual",
        action="store_true",
        help="Skip loading manual CSV stories.",
    )
    args = p.parse_args()

    results = _load_judge_results(args.judge_file)
    print(f"Loaded {len(results):,} judged items from {args.judge_file}")

    by_company: dict[str, list[dict]] = {c: [] for c in args.companies}
    for r in results:
        c = r.get("company")
        if c not in by_company:
            continue
        by_company[c].append(r)

    nyt_idx = _index_nyt_articles(args.nyt_dir)
    hn_idx = _index_hn_stories(args.hn_dir)
    print(f"Indexed NYT rows: {len(nyt_idx):,}  |  HN rows: {len(hn_idx):,}")

    manual_by_company: dict[str, list[dict]] = {}
    if not args.no_manual and args.manual_csvs:
        print("\nLoading manually curated stories...")
        manual_by_company = _load_manual_csvs(args.manual_csvs, args.companies)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected: list[dict] = []
    manual_count = 0

    for company in args.companies:
        picked: list[dict] = []

        # 1. Add manual stories first
        manual_stories = manual_by_company.get(company, [])
        for story in manual_stories:
            enriched = dict(story)
            enriched["story_company"] = company
            enriched["rank"] = len(picked) + 1
            enriched["max_risk_contr"] = None
            picked.append(enriched)
            manual_count += 1

        picked_urls = {s.get("url") for s in picked if s.get("url")}

        # 2. Add judged stories to fill up to top-n
        items = by_company[company]
        items.sort(
            key=lambda r: (_max_risk(r), float(r.get("relevance", 0))),
            reverse=True,
        )

        ranked_enriched = []
        for s in items:
            enriched = _enrich(s, nyt_idx, hn_idx)
            if enriched.get("url") and enriched["url"] in picked_urls:
                continue
            ranked_enriched.append(enriched)

        sid_info: dict[str, dict] = {}
        story_to_sid: dict[int, str] = {}
        sid_to_enriched: dict[str, dict] = {}
        pool_size = 0

        if args.cluster and ranked_enriched and len(picked) < args.top_n:
            desired = max(args.cluster_pool, args.top_n)
            pool_size = min(desired, len(ranked_enriched))
            if desired > args.cluster_pool:
                print(
                    f"    {company}: cluster judge input "
                    f"{pool_size}/{len(ranked_enriched)} stories "
                    f"(max(--cluster-pool {args.cluster_pool}, --top-n "
                    f"{args.top_n}))",
                    flush=True,
                )
            pool = ranked_enriched[:pool_size]
            cluster_data = _cluster_company_stories(
                company,
                pool,
                args.api_key,
                args.cluster_model,
                args.cluster_cache,
                refresh=args.refresh_clusters,
            )
            sid_info = cluster_data["sid_info"]
            for sid, story in cluster_data["items"]:
                story_to_sid[id(story)] = sid
                sid_to_enriched[sid] = story

        used_reps: set[str] = set()
        for story in ranked_enriched:
            if len(picked) >= args.top_n:
                break
            sid = story_to_sid.get(id(story))
            info = sid_info.get(sid) if sid else None
            rep = info["representative_id"] if info else None

            # When a non-representative cluster member is encountered first
            # (ranked higher than its representative), emit the representative
            # at this position instead of skipping the whole cluster.
            if info and sid != rep:
                if rep in used_reps:
                    continue  # representative already emitted via an earlier member
                emit_story = sid_to_enriched.get(rep)
                if emit_story is None:
                    continue  # representative not in pool (shouldn't happen)
                emit_sid = rep
                emit_info = sid_info.get(rep)
            else:
                if rep and rep in used_reps:
                    continue  # already emitted (representative seen after a member)
                emit_story = story
                emit_sid = sid
                emit_info = info

            if rep:
                used_reps.add(rep)
            elif emit_sid:
                used_reps.add(emit_sid)

            if _max_risk(emit_story) < args.min_risk_or_controversy:
                continue
            if float(emit_story.get("relevance", 0)) < args.min_relevance:
                continue

            enriched = dict(emit_story)
            enriched["rank"] = len(picked) + 1
            enriched["story_company"] = company
            enriched["max_risk_contr"] = _max_risk(emit_story)
            if emit_info is not None:
                enriched["cluster_index"] = emit_info["cluster_index"]
                enriched["cluster_size"] = emit_info["cluster_size"]
                enriched["cluster_topic"] = emit_info["cluster_topic"]
                enriched["cluster_sid"] = emit_sid
                enriched["cluster_member_sids"] = emit_info["cluster_ids"]
                enriched["is_cluster_representative"] = emit_info["cluster_size"] > 1
            picked.append(enriched)

        selected.extend(picked)

        n_clusters = sum(
            1
            for sid, info in sid_info.items()
            if info["cluster_size"] > 1 and info["representative_id"] == sid
        )
        n_dupes = sum(
            1
            for sid, info in sid_info.items()
            if info["cluster_size"] > 1 and info["representative_id"] != sid
        )
        suffix = (
            f"  (clustered_top={pool_size}, "
            f"clusters={n_clusters}, deduped={n_dupes})"
            if args.cluster and pool_size > 0
            else ""
        )
        manual_str = f" ({len(manual_stories)} manual)" if manual_stories else ""
        print(f"  {company:<10}  picked {len(picked)}/{args.top_n}{manual_str}{suffix}")

    args.out.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with args.txt_out.open("w", encoding="utf-8") as f:
        for s in selected:
            if url := s.get("url"):
                f.write(f"{url}\n")

    print(
        f"\nWrote {len(selected)} stories → {args.out} and {args.txt_out}"
        f"  (auto: {len(selected) - manual_count}, manual: {manual_count})"
    )


if __name__ == "__main__":
    main()
