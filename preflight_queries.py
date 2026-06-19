"""
Script that writes the parameters to be sent as queries to a file.
This does the "pre-flight" stuff (loading stories, full texts, templates, and rendering prompts)
so that the actual querying scripts just read the jobs and execute them.

Produces two artefacts:

  * ``data/jobs.json`` -- one record per ``(story, target_model, template, rep)``
    cell, with the rendered target prompt. Consumed by ``query_targets.py``.

  * ``data/judge_jobs.json`` -- one record per ``(target cell, judge_model)``
    pair (K of these per target cell), with the rendered rubric, the judge
    context block (for headline-only templates), and the deterministic
    judge-model assignment from the balanced incomplete block design.
    Consumed by ``query_judges.py`` once the target responses exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    COMPANY_CEO,
    DEFAULT_COMPANIES,
    PROMPT_TEMPLATES,
    TARGET_MODELS,
    PromptTemplate,
    render_prompt,
    render_rubric,
)

from config import target_company_of_model as _target_company_of_model

ARTICLES_DIR = Path("data/articles")

# Template IDs whose target prompt only contains the headline. The judge
# needs the full item text to evaluate accuracy / completeness.
HEADLINE_ONLY_TEMPLATES = {1}


DEFAULT_K_JUDGES = 1
TARGET_SLOT_MULTIPLIER = 2
DEFAULT_STORIES = Path("data/stories/top_per_company.json")
DEFAULT_FALLBACK_STORIES = Path("data/judge/combined.json")
DEFAULT_FULLTEXT_INDEX = Path("data/processed-full-texts/url_index.json")
DEFAULT_JOBS_OUT = Path("data/jobs.json")
DEFAULT_JUDGE_JOBS_OUT = Path("data/judge_jobs.json")
DEFAULT_MANUAL_CSV_FILES = [
    Path("data/expert_added_urls.csv"),
    Path("data/manually_added_urls.csv"),
]


def strip_repeated_headline(text: str, headline: str) -> str:
    """Remove a leading markdown/plain headline when it is rendered separately."""

    def _normalize_headline(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^#{1,6}\s+", "", s).strip()  # markdown heading
        s = re.sub(r"\s*\[[^\]]*\]\s*$", "", s).strip()  # trailing [..]
        s = re.sub(r"\s*(?:\.{3}|…|\[\.\.\.\])\s*$", "", s).strip()  # trailing ellipsis
        s = re.sub(r"\s+", " ", s)
        return s

    headline = headline.strip()
    if not headline:
        return text.strip()

    lines = text.strip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    first = lines[0].strip()
    first_plain = re.sub(r"^#{1,6}\s+", "", first).strip()
    norm_first = _normalize_headline(first_plain)
    norm_headline = _normalize_headline(headline)

    # Treat headings like "# Headline [...]" as duplicated headline.
    # Keep this conservative to avoid stripping unrelated headings.
    looks_like_match = False
    if norm_first and norm_headline:
        if norm_first == norm_headline:
            looks_like_match = True
        elif (len(norm_first) >= 25 and len(norm_headline) >= 25) and (
            norm_first.startswith(norm_headline) or norm_headline.startswith(norm_first)
        ):
            looks_like_match = True

    if looks_like_match:
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def load_fulltext_index(path: Path) -> dict[str, str]:
    """Load URL-indexed processed markdown produced by process_full_texts.py."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load fulltext index {path}: {exc}", file=sys.stderr)
        return {}

    index: dict[str, str] = {}
    for url, record in raw.items():
        if record.get("status") != "ok" or not record.get("markdown_path"):
            continue
        md_path = Path(record["markdown_path"])
        try:
            markdown = md_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(
                f"Warning: could not read processed fulltext {md_path}: {exc}",
                file=sys.stderr,
            )
            continue
        if markdown:
            index[url] = markdown
    return index


def _safe_id(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)


def cached_article_path(story_source: str, story_id: str, articles_dir: Path) -> Path:
    return articles_dir / f"{story_source}_{_safe_id(story_id)}.txt"


def load_full_article(
    story: dict,
    *,
    articles_dir: Path = ARTICLES_DIR,
    fulltext_index: dict[str, str] | None = None,
) -> str:
    """Return the best-available full-text body for a story.

    Preference order:
      1. ``data/articles/<source>_<id>.txt``  (populated externally)
      2. URL-indexed processed markdown from ``process_full_texts.py``
      3. abstract + lead_paragraph (NYT) or text (HN) from the story dict
    Returns "" if nothing is available.
    """
    cache = cached_article_path(story["source"], story["id"], articles_dir)
    if cache.exists():
        try:
            t = cache.read_text(encoding="utf-8").strip()
            if t:
                return t
        except OSError:
            pass

    if fulltext_index and story.get("url") in fulltext_index:
        return fulltext_index[story["url"]]

    parts: list[str] = []
    if story.get("headline"):
        parts.append(story["headline"])
    if story["source"] == "nyt":
        if story.get("byline"):
            parts.append(story["byline"])
        if story.get("abstract"):
            parts.append(story["abstract"])
        if story.get("lead_paragraph"):
            parts.append(story["lead_paragraph"])
    else:
        if story.get("text"):
            parts.append(story["text"])
    return "\n\n".join(p for p in parts if p).strip()


def build_context_block(template_id: int, full_text: str) -> str:
    """Optional 'full text for context' block injected into the judge prompt."""
    if template_id not in HEADLINE_ONLY_TEMPLATES:
        return ""
    if not full_text:
        return ""
    return (
        "\nFor context — the model to be evaluated only saw the headline above; the full story "
        "the headline refers to is reproduced below."
        "Full story:\n<document>\n" + full_text.strip() + "\n</document>\n"
    )


def build_item_text(
    story: dict,
    fulltext_index: dict[str, str] | None = None,
    *,
    include_headline: bool = True,
) -> str:
    """Full text of an HN or NYT item."""
    parts: list[str] = []
    if include_headline and story.get("headline"):
        parts.append(story["headline"])

    if fulltext_index is not None and story.get("url") in fulltext_index:
        markdown = fulltext_index[story["url"]]
        if not include_headline:
            markdown = strip_repeated_headline(markdown, story.get("headline", ""))
        if markdown:
            parts.append(markdown)

    return "\n\n".join(p for p in parts if p)


def render_for_story(
    template: PromptTemplate,
    story: dict,
    story_company: str,
    fulltext_index: dict[str, str] | None = None,
) -> str:
    title = story.get("headline", "")
    url = story.get("url", "")
    date = story.get("date") or story.get("story_date", "")

    main_text = build_item_text(story, fulltext_index, include_headline=False)

    formatted_headline = f"“<document_title>{title}</document_title>\n<document_date>{date}</document_date>\n<document_url>{url}</document_url>”"
    formatted_item = f"“\n<document_title>{title}</document_title>\n<document_date>{date}</document_date>\n<document_url>{url}</document_url>\n<document>{main_text}</document>\n”"

    return render_prompt(
        template,
        headline=formatted_headline,
        item=formatted_item,
        company=story_company,
        ceo=COMPANY_CEO.get(story_company, ""),
    )


def _filter_valid_stories(
    stories: list[dict], fulltext_index: dict[str, str] | None
) -> list[dict]:
    """Drop stories whose URL is missing from the full-text index.

    Manual stories are not exempt: a story without an extracted full text
    cannot be used in the balanced design regardless of provenance.
    """
    valid: list[dict] = []
    missing_urls: set[str] = set()
    skipped = 0
    for story in stories:
        url = story.get("url", "")
        if fulltext_index is not None and (not url or url not in fulltext_index):
            if url:
                missing_urls.add(url)
            skipped += 1
            continue
        valid.append(story)
    if skipped > 0:
        print(
            f"Warning: Missing full text for {len(missing_urls)} unique URLs. "
            f"Skipped {skipped} stories."
        )
    return valid


def build_jobs(
    stories: list[dict],
    target_companies: list[str],
    template_ids: list[int],
    target_models_map: dict[str, list[str]],
    max_models: int | None,
    fulltext_index: dict[str, str] | None = None,
    slot_multiplier: int = TARGET_SLOT_MULTIPLIER,
) -> list[dict]:
    """Balanced design described in readme step 9.

    Per target model in company ``c``, generate exactly
    ``2 * slot_multiplier * (N_tot - n_min)`` target jobs split equally
    between stories about ``c`` ("same" side) and stories about other
    companies ("other" side), where ``N_tot`` is the total number of
    stories across all ``target_companies`` and ``n_min`` is the smallest
    per-company count. Combined with ``K`` judges per slot (see
    ``build_judge_jobs``) this yields
    ``2 * slot_multiplier * (N_tot - n_min) * K`` ratings per target
    model.

    Stories cycle the fastest (alternating same / other every slot),
    prompt templates cycle the slowest. Since ``gcd(2, T) = 1`` for the
    25-template design, every (template, side) pair appears equally
    often within each 2T-slot block, so the chi contrast is orthogonal
    to template by construction.
    """
    template_by_id = {t.id: t for t in PROMPT_TEMPLATES}
    templates = [template_by_id[i] for i in template_ids if i in template_by_id]
    if not templates:
        return []

    valid_stories = _filter_valid_stories(stories, fulltext_index)

    by_company: dict[str, list[dict]] = defaultdict(list)
    for s in valid_stories:
        by_company[s["story_company"]].append(s)
    for lst in by_company.values():
        lst.sort(key=lambda s: s["id"])

    counted_companies = [c for c in target_companies if by_company.get(c)]
    if not counted_companies:
        return []
    all_stories_sorted = [s for c in counted_companies for s in by_company[c]]
    n_tot = len(all_stories_sorted)
    n_min = min(len(by_company[c]) for c in counted_companies)
    s_per_side = 2 * slot_multiplier * (n_tot - n_min)
    t = len(templates)

    print(
        f"Balanced design: N_tot={n_tot}, n_min={n_min}, "
        f"slot_multiplier={slot_multiplier}, "
        f"slots/model = {2 * s_per_side} (per side: {s_per_side})"
    )

    jobs: list[dict] = []
    for c_target in target_companies:
        same = list(by_company.get(c_target, []))
        if not same:
            continue
        other = [s for s in all_stories_sorted if s["story_company"] != c_target]
        if not other:
            continue
        models = target_models_map.get(c_target, [])
        if max_models is not None:
            models = models[:max_models]

        for model in models:
            rep_counter: dict[tuple[str, int], int] = defaultdict(int)
            for k in range(2 * s_per_side):
                if k % 2 == 0:
                    story = same[(k // 2) % len(same)]
                else:
                    story = other[(k // 2) % len(other)]
                tpl = templates[k % t]
                rep_key = (story["id"], tpl.id)
                rep_counter[rep_key] += 1
                rep = rep_counter[rep_key]
                story_company = story["story_company"]
                rendered = render_for_story(tpl, story, story_company, fulltext_index)
                jobs.append(
                    {
                        "story_id": story["id"],
                        "story_source": story["source"],
                        "story_date": story.get("date") or story.get("story_date", ""),
                        "story_company": story_company,
                        "story_headline": story.get("headline", ""),
                        "story_url": story.get("url", ""),
                        "target_company": c_target,
                        "target_model": model,
                        "template_id": tpl.id,
                        "rep": rep,
                        "slot_idx": k,
                        "prompt": rendered,
                    }
                )
    return jobs


def build_judge_jobs(
    target_jobs: list[dict],
    *,
    k: int,
    judges_override: list[str] | None,
    stories_index: dict[str, dict] | None,
    articles_dir: Path,
    fulltext_index: dict[str, str] | None,
) -> list[dict]:
    """K judges per target job, assigned by cycling through all judge models.

    All models in DEFAULT_COMPANIES (or ``judges_override``) are eligible as
    judges for every target job -- no exclusions based on story or target
    company.  A single per-target-model counter advances by K per slot so
    that judges are spread uniformly across stories, templates, and the
    same/other split.

    For headline-only templates we also pre-render the full-text context
    block consumed by ``query_judges.py``.
    """
    template_by_id = {t.id: t for t in PROMPT_TEMPLATES}

    all_judge_models: list[str] = sorted(
        m for c in DEFAULT_COMPANIES for m in TARGET_MODELS.get(c, [])
    )

    by_model: dict[str, list[dict]] = defaultdict(list)
    for j in target_jobs:
        by_model[j["target_model"]].append(j)

    judge_jobs: list[dict] = []
    for target_model, jobs in by_model.items():
        pool = judges_override if judges_override is not None else all_judge_models
        n = len(pool)

        # Group jobs by story and iterate stories in a deterministic order
        # while assigning judges from a global counter mod n.  Each story
        # thus receives a contiguous run of judges (good story x judge
        # balance), and the global counter guarantees that judges are
        # distributed across the target model's 4*(N_tot - n_min) slots
        # within +/- 1 of perfectly uniform (good target x judge balance).
        # Stagger the starting counter per target model so that judges are
        # spread across (judge, template) cells across targets as well.
        jobs_by_story: dict[str, list[dict]] = defaultdict(list)
        for j in jobs:
            jobs_by_story[j["story_id"]].append(j)
        for lst in jobs_by_story.values():
            lst.sort(key=lambda j: (j.get("slot_idx", 0), j["template_id"], j["rep"]))

        target_seed = int(hashlib.sha256(target_model.encode()).hexdigest(), 16) % n
        counter = target_seed
        sorted_jobs: list[dict] = []
        for sid in sorted(jobs_by_story):
            sorted_jobs.extend(jobs_by_story[sid])

        for j in sorted_jobs:
            template = template_by_id.get(j["template_id"])
            if template is None:
                continue
            story_company = j["story_company"]
            target_company = (
                j.get("target_company") or _target_company_of_model(target_model) or "?"
            )

            judges = [pool[(counter + i) % n] for i in range(k)]
            counter += k

            context_block = ""
            if (
                stories_index is not None
                and j["template_id"] in HEADLINE_ONLY_TEMPLATES
            ):
                story = stories_index.get(j["story_id"])
                if story is not None:
                    full_text = load_full_article(
                        story,
                        articles_dir=articles_dir,
                        fulltext_index=fulltext_index,
                    )
                    context_block = build_context_block(j["template_id"], full_text)

            rubric = render_rubric(
                template,
                company=story_company,
                ceo=COMPANY_CEO.get(story_company, ""),
            )

            for judge in judges:
                judge_jobs.append(
                    {
                        "story_id": j["story_id"],
                        "story_source": j["story_source"],
                        "story_date": j.get("story_date", ""),
                        "story_company": story_company,
                        "story_headline": j.get("story_headline", ""),
                        "story_url": j.get("story_url", ""),
                        "target_company": target_company,
                        "target_model": target_model,
                        "template_id": j["template_id"],
                        "rep": j["rep"],
                        "slot_idx": j.get("slot_idx", 0),
                        "judge_model": judge,
                        "rubric": rubric,
                        "context_block": context_block,
                    }
                )
    return judge_jobs


def report_balance(target_jobs: list[dict], judge_jobs: list[dict]) -> None:
    """Print two-way balance diagnostics for the realised design.

    For every pair of {target_model, story, judge, template} we report cell
    coverage and the coefficient of variation of the per-cell counts (CV =
    sd / mean). Coverage less than 100% means some pairs never co-occur.
    """

    def _key(d: dict) -> tuple:
        return (d["target_model"], d["story_id"], d["template_id"], d["rep"])

    tj_index = {_key(t): t for t in target_jobs}
    rows: list[dict] = []
    for j in judge_jobs:
        t = tj_index.get(_key(j))
        if t is None:
            continue
        rows.append(
            {
                "target": t["target_model"],
                "story": t["story_id"],
                "story_company": t["story_company"],
                "judge": j["judge_model"],
                "judge_company": _target_company_of_model(j["judge_model"])
                or j["judge_model"],
                "template": str(t["template_id"]),
                "same": int(t["story_company"] == t["target_company"]),
            }
        )
    if not rows:
        return

    def _cv(counts: Counter) -> float:
        values = list(counts.values())
        mu = sum(values) / len(values)
        if mu == 0:
            return float("nan")
        var = sum((v - mu) ** 2 for v in values) / len(values)
        return math.sqrt(var) / mu

    factors = ["target", "story", "story_company", "judge", "judge_company", "template"]
    print("\nTwo-way balance (CV = sd/mean of per-cell counts):")
    print(
        f"  {'Pair':<22}  {'cells':>10}  {'min':>4}  {'mean':>6}  "
        f"{'max':>4}  {'CV':>6}  {'cover':>7}"
    )
    print(
        f"  {'-' * 22}  {'-' * 10}  {'-' * 4}  {'-' * 6}  "
        f"{'-' * 4}  {'-' * 6}  {'-' * 7}"
    )
    for a, b in itertools.combinations(factors, 2):
        ct = Counter((r[a], r[b]) for r in rows)
        levels_a = len({r[a] for r in rows})
        levels_b = len({r[b] for r in rows})
        expected = levels_a * levels_b
        values = list(ct.values())
        mu = sum(values) / len(values)
        cover = 100 * len(ct) / expected if expected else 0.0
        print(
            f"  {a + ' x ' + b:<22}  {len(ct):>4}/{expected:<5}  "
            f"{min(values):>4}  {mu:>6.1f}  {max(values):>4}  "
            f"{_cv(ct) * 100:>5.1f}%  {cover:>6.1f}%"
        )

    # chi (same/other) versus each factor: per-level "same" fraction range.
    print("\nFraction same-company per level (target chi covariate; ideal = 0.500):")
    for f in ("target", "story_company", "judge", "judge_company", "template"):
        per_level: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            per_level[r[f]].append(r["same"])
        fracs = [sum(v) / len(v) for v in per_level.values() if v]
        print(
            f"  by {f:<8}  min={min(fracs):.3f}  max={max(fracs):.3f}  "
            f"spread={max(fracs) - min(fracs):.3f}"
        )


def _load_manual_csv_stories(
    csv_paths: list[Path], valid_companies: list[str]
) -> list[dict]:
    """Load manually curated story CSVs (headline,company,date,url).

    Returns a flat list of story dicts ready to be merged into the main stories
    list.  Each dict has ``source="manual"`` and ``story_company`` set so it is
    handled correctly by ``build_jobs``.
    """
    stories: list[dict] = []
    seen_urls: set[str] = set()
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"  Note: manual CSV not found, skipping: {csv_path}")
            continue
        loaded = 0
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                url = (row.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                company = (row.get("company") or "").strip()
                if valid_companies and company not in valid_companies:
                    continue
                story_id = f"manual_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
                stories.append(
                    {
                        "source": "manual",
                        "id": story_id,
                        "date": (row.get("date") or "").strip(),
                        "headline": (row.get("headline") or "").strip(),
                        "url": url,
                        "story_company": company,
                        "csv_source": csv_path.name,
                    }
                )
                loaded += 1
        print(f"  {csv_path.name}: {loaded} stories loaded")
    return stories


def _load_stories_index(
    stories_file: Path, fallback_stories_file: Path | None
) -> dict[str, dict] | None:
    """Merge ``stories_file`` over ``fallback_stories_file`` by story id."""
    if not stories_file or not str(stories_file).strip():
        return None
    try:
        index: dict[str, dict] = {}
        if fallback_stories_file and fallback_stories_file.exists():
            fallback = json.loads(fallback_stories_file.read_text(encoding="utf-8"))
            index.update({s["id"]: s for s in fallback})
        primary = json.loads(stories_file.read_text(encoding="utf-8"))
        index.update({s["id"]: s for s in primary})
        return index
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"Warning: could not load stories file ({exc}); "
            f"full-text augmentation disabled.",
            file=sys.stderr,
        )
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-flight: render queries and save to file."
    )
    p.add_argument("--stories-file", type=Path, default=DEFAULT_STORIES)
    p.add_argument(
        "--fallback-stories-file",
        type=Path,
        default=DEFAULT_FALLBACK_STORIES,
        help="Additional story metadata used to fill dates / full-text "
        "for older entries when building judge jobs.",
    )
    p.add_argument("--out-file", type=Path, default=DEFAULT_JOBS_OUT)
    p.add_argument(
        "--judge-jobs-out",
        type=Path,
        default=DEFAULT_JUDGE_JOBS_OUT,
        help="Pre-rendered judge-job parameter file (consumed by query_judges.py).",
    )
    p.add_argument(
        "--skip-judge-jobs",
        action="store_true",
        help="Do not pre-render judge jobs (only emit target jobs).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K_JUDGES,
    )
    p.add_argument(
        "--slot-multiplier",
        type=int,
        default=TARGET_SLOT_MULTIPLIER,
        help="Multiplier on the per-side slot count "
        "(s_per_side = 2 * slot_multiplier * (N_tot - n_min)). Default 2 "
        "doubles target prompts per model relative to the minimum balanced "
        "design (which is slot_multiplier=1).",
    )
    p.add_argument(
        "--judges-override",
        nargs="+",
        default=None,
        help="Use these models as the judge pool (smoke tests).",
    )
    p.add_argument(
        "--target-companies",
        nargs="+",
        default=DEFAULT_COMPANIES,
        help="Which companies' models to query.",
    )
    p.add_argument(
        "--templates",
        nargs="+",
        type=int,
        default=[t.id for t in PROMPT_TEMPLATES],
        help="Template IDs to run (1..25).",
    )
    p.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Cap stories per company (for test runs).",
    )
    p.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Cap models per company (for test runs).",
    )
    p.add_argument(
        "--models-override",
        nargs="+",
        default=None,
        help="Replace TARGET_MODELS with this flat list (used for ALL "
        "--target-companies). Useful for smoke tests.",
    )
    args = p.parse_args()

    stories = json.loads(args.stories_file.read_text(encoding="utf-8"))

    if args.max_stories is not None:
        by_company: dict[str, list[dict]] = {}
        for s in stories:
            by_company.setdefault(s["story_company"], []).append(s)
        stories = [
            s for items in by_company.values() for s in items[: args.max_stories]
        ]

    print("\nLoading manually curated stories...")
    manual_stories = _load_manual_csv_stories(
        DEFAULT_MANUAL_CSV_FILES, args.target_companies
    )
    existing_urls: set[str] = {s.get("url") for s in stories if s.get("url")}
    new_manual = [s for s in manual_stories if s["url"] not in existing_urls]
    if new_manual:
        print(
            f"  Merging {len(new_manual)} manual stories not already in {args.stories_file.name}"
        )
        stories.extend(new_manual)
    else:
        print("  All manual stories are already present in the stories file.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    target_models_map = TARGET_MODELS
    if args.models_override:
        target_models_map = {
            c: list(args.models_override) for c in args.target_companies
        }

    fulltext_index = load_fulltext_index(DEFAULT_FULLTEXT_INDEX)
    if fulltext_index:
        print(
            f"Loaded processed full texts for {len(fulltext_index)} URLs from {DEFAULT_FULLTEXT_INDEX}"
        )

    # Print per-company counts after the full-text filter so the numbers
    # reflect what actually feeds the balanced design.
    filtered_stories = _filter_valid_stories(stories, fulltext_index or None)
    counts: dict[str, int] = defaultdict(int)
    for s in filtered_stories:
        counts[s["story_company"]] += 1
    print("\nStories per company (after full-text filter):")
    for company, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {company}: {count}")

    jobs = build_jobs(
        stories,
        args.target_companies,
        args.templates,
        target_models_map,
        args.max_models,
        fulltext_index,
        slot_multiplier=args.slot_multiplier,
    )

    print(
        f"Stories: {len(stories)} | target companies: {len(args.target_companies)} "
        f"| templates: {len(args.templates)}"
    )
    print(f"Total target jobs: {len(jobs):,}")

    args.out_file.write_text(
        json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Jobs written to {args.out_file}")

    if args.skip_judge_jobs:
        return

    stories_index = _load_stories_index(args.stories_file, args.fallback_stories_file)
    judge_jobs = build_judge_jobs(
        jobs,
        k=args.k,
        judges_override=args.judges_override,
        stories_index=stories_index,
        articles_dir=ARTICLES_DIR,
        fulltext_index=fulltext_index or None,
    )
    args.judge_jobs_out.parent.mkdir(parents=True, exist_ok=True)
    args.judge_jobs_out.write_text(
        json.dumps(judge_jobs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Judge jobs: {len(judge_jobs):,}  (K={args.k} judges per target cell)")
    print(f"Judge jobs written to {args.judge_jobs_out}")

    report_balance(jobs, judge_jobs)


if __name__ == "__main__":
    main()
