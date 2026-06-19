"""
Process saved HTML full-text fetches into content-only Markdown indexed by URL.

Reads:
    articles.txt
    data/full-texts/**/*.html

Writes:
    data/processed-full-texts/by-url/<sha256>.md
    data/processed-full-texts/url_index.json
    data/processed-full-texts/manifest.jsonl
    data/processed-full-texts/coverage_report.txt

The source HTML files do not have an external URL-to-file map, so this script
infers one from canonical/meta/JSON-LD URLs, then falls back to same-host slug
matching when a saved page omits usable metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import trafilatura
from bs4 import BeautifulSoup


DEFAULT_ARTICLES = Path("data/articles.txt")
DEFAULT_HTML_DIR = Path("data/full-texts")
DEFAULT_OUT_DIR = Path("data/processed-full-texts")
DEFAULT_EXTRA_URL_FILES = [
    Path("data/expert_added_urls.csv"),
    Path("data/manually_added_urls.csv"),
]

MIN_WORDS = 80

PAYWALL_PATTERNS = [
    r"\bsubscribe to read\b",
    r"\bsubscription required\b",
    r"\bsubscriber-only\b",
    r"\bthis article is for subscribers\b",
    r"\bpaywall\b",
]

LOGIN_PATTERNS = [
    r"\blog in\b",
    r"\blogin\b",
    r"\bsign in\b",
    r"\bsign up\b",
    r"\bcreate an account\b",
    r"\bbefore you continue\b",
]

FETCH_FAILURE_PATTERNS = [
    r"\baccess denied\b",
    r"\bjust a moment\b",
    r"\benable javascript\b",
    r"\bjavascript is not available\b",
    r"\bjavascript is required\b",
    r"\bsite not found\b",
    r"\bpage not found\b",
    r"\b404\b",
    r"\bwordpress error\b",
    r"\bchecking your browser\b",
    r"\bverify you are human\b",
]

NOISE_LINE_PATTERNS = [
    r"^advertisement$",
    r"^supported by$",
    r"^skip advertisement$",
    r"^\[?skip advertisement\]?",
    r"^skip to (main )?content$",
    r"^share full article$",
    r"^thanks for reading",
    r"^read more$",
    r"^continue reading$",
    r"^sign up for .+ newsletter$",
    r"^subscribe to .+$",
    r"^advertisement[·\s·]+scroll to continue$",
    r"^democracy dies in darkness$",
    r"^follow us$",
    r"^get instant access to .+",
    r"^\[sign in\]$",
    r"^\[sign up for .+\]",
    r"^get (the free|on the free list).+",
    r"^already a member\?",
    r"^no spam ever",
    # NYT section/column rubrics
    r"^guest essay$",
    r"^the shift$",
    r"^hard fork$",
    r"^times insider$",
    r"^nonfiction$",
    r"^dealbook$",
    r"^transcript$",
    # GitHub PR UI buttons
    r"^\[open in github",
    r"^\[open in a new github",
    r"^\[open in codespace",
    # Misc site labels
    r"^threat intelligence$",
    r"^by\s*$",
    # Bare engagement-count lines from X/Twitter embeds ("46", "2.1K", "1.7K")
    r"^\d+(\.\d+)?[kmKM]?\s*$",
    # The Verge: cookie consent notice injected around embedded content
    r"^this content isn't visible due to your cookie preferences",
    # The Verge: subscribe-to-continue paywall prompt
    r"^\[subscribe to the verge\]",
    # App-download CTA embedded in article body (Sky News, etc.)
    r"^install the .+ app for free",
    # Twitter/X: "Read 1.4K replies" engagement counter (decimal + K/M suffix)
    r"^read \d+\.?\d*[kmKM]? repl",
    # Washington Post: AI reader-comments summary header and label
    r"^what readers are saying",
    r"^this summary is ai.generated",
    # Substack / blogging platforms: read-time metadata line ("- 9 min read")
    r"^-\s*\d+ min read$",
    # Euronews: page-jump anchors to comment widget and accessibility menu
    r"^\[comments\]\(https?://",
    r"^\[go to accessibility shortcuts\]",
    # Sky News: "Read more [topic]: [related article title]" inline promo
    r"^read more \w[\w\s]*:",
    # News site publication date-time stamp as a standalone line
    # ("Wednesday 4 June 2025 16:45, UK")
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+\d{1,2}\s+\w+\s+\d{4}",
    # Bare "[" — orphaned start of a Markdown link (ticker/related-article artifact)
    r"^\[$",
    # Semafor: "In this article:" inline navigation label
    r"^in this article:?$",
    # The Hill: CMS topic/person tag navigation links at end of article
    r"^\[[^\]]+\]\(https?://thehill\.com/(people|social-tags)/",
]

# Inline text substitutions applied to the full markdown before line-by-line cleaning.
# Each entry is (compiled_pattern, replacement).
INLINE_NOISE_SUBSTITUTIONS = [
    # WSJ inline stock ticker links: "[Meta Platforms](wsj.com/market-data/...) [META -0.98%...](wsj.com/market-data/...)"
    (
        re.compile(
            r"\[[^\]]*\]\(https://www\.wsj\.com/market-data/quotes/[^\)]+\)\s*"
            r"(?:\[[^\]]*\]\(https://www\.wsj\.com/market-data/quotes/[^\)]+\)\s*)?",
            re.I,
        ),
        "",
    ),
    # Sherwood/Robinhood inline stock tickers: "[MetaMETA $622.05 (0.88%)](robinhood.com/...)"
    (
        re.compile(
            r"\[[A-Z][A-Za-z ]*[A-Z]{1,5}\s+\$[\d.]+\s*\([+-]?\d+\.?\d*%\)\]\(https://robinhood\.com[^\)]*\)\s*",
            re.I,
        ),
        "",
    ),
    # Washington Post: "What readers are saying" UI tag concatenated into author/byline lines
    (re.compile(r"What readers are saying", re.I), ""),
    # Trafilatura artifact: lone * on its own line (image element with no alt text)
    (re.compile(r"(?m)^\*\s*$(?:\n\*\s*$)*"), ""),
    # CNN/MarketWatch: orphaned ticker fragment "(GOOG)](url)" at start of a line,
    # left when "[Company\n[(TICKER)](url)]" inline stock links are split across lines.
    (re.compile(r"(?m)^\([A-Z]{1,6}\)\]\(https?://[^\)]+\)\s*", re.I), ""),
]

FOOTER_STOP_PATTERNS = [
    r"^\[?order reprints\]?",
    r"^\[?today.s paper\]?",
    r"^## related content$",
    r"^related content$",
    r"^copyright ©",
    # "Read N replies" — updated to handle decimal+K/M notation (e.g. "Read 1.4K replies")
    r"^read \d+[.,\d]*[kmKM]? repl",
    # NYT Opinion: editorial footer ("The Times is committed to publishing a diversity of letters…")
    r"^the times is committed to publishing",
    # Blog/CMS "explore more posts" marketing section
    r"^explore our latest insights",
    # Washington Post: newsletter upsell section ("NewsletterWeekdays", "NewsletterAs news breaks")
    r"^newsletter",
    # WSJ: standalone "Copyright" line when the © falls on the following line
    r"^copyright\s*$",
    # Broken-link artifact "](" at line start — always signals non-article sidebar content
    r"^\]\(https?://",
    # Copyright footer without © symbol (The Hill/Nexstar, AP, Reuters, etc.)
    r"^copyright \d{4}\b",
]


@dataclass
class HtmlCandidate:
    path: Path
    title: str
    urls: set[str]
    normalized_urls: set[str]
    text_for_matching: str


def normalize_url(raw_url: str) -> str:
    """Normalize URLs enough to match saved canonical URLs to articles.txt."""
    url = html.unescape((raw_url or "").strip()).replace("\\/", "/")
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if not parsed.netloc:
        return ""

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in {"twitter.com", "mobile.twitter.com"}:
        host = "x.com"
    if host == "old.reddit.com":
        host = "reddit.com"

    path = unquote(parsed.path or "/")
    path = re.sub(r"/{2,}", "/", path)
    parts = [p for p in path.split("/") if p]
    if host == "x.com" and len(parts) >= 3 and parts[1] == "status":
        path = f"/{parts[0]}/status/{parts[2]}"
    elif (
        host == "reddit.com"
        and len(parts) >= 4
        and parts[0] == "r"
        and parts[2] == "comments"
    ):
        path = "/" + "/".join(parts[:4]) + "/"

    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, "", ""))


def host_dir_name(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host.replace(".", "-")


def url_slug_tokens(url: str) -> set[str]:
    path = unquote(urlsplit(url).path)
    tokens = set(re.findall(r"[a-z0-9]{3,}", path.lower()))
    return {t for t in tokens if not t.isdigit()}


def path_tokens(path: Path) -> set[str]:
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}(?:_\d{2}-\d{2}-\d{2})?_", "", path.stem)
    tokens = set(re.findall(r"[a-z0-9]{3,}", stem.lower()))
    return {t for t in tokens if not t.isdigit()}


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(title or "")).strip()


def iter_json_values(obj: Any) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"url", "@id", "mainEntityOfPage", "sameAs"}:
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, dict):
                    nested = value.get("@id") or value.get("url")
                    if isinstance(nested, str):
                        values.append(nested)
                elif isinstance(value, list):
                    values.extend(v for v in value if isinstance(v, str))
            values.extend(iter_json_values(value))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(iter_json_values(item))
    return values


def extract_urls_and_title(path: Path, html_text: str) -> tuple[set[str], str]:
    urls: set[str] = set()
    soup = BeautifulSoup(html_text, "lxml")
    title = clean_title(soup.title.string if soup.title and soup.title.string else "")

    for tag in soup.find_all(["link", "meta"]):
        if tag.name == "link":
            rel = {str(v).lower() for v in (tag.get("rel") or [])}
            href = tag.get("href") or ""
            if "canonical" in rel or tag.get("itemprop") == "url":
                urls.add(href)
            elif tag.get("type") == "application/rss+xml" and href.endswith(
                "/index.xml"
            ):
                urls.add(href.removesuffix("/index.xml") + "/")
        elif tag.name == "meta":
            key = (
                tag.get("property") or tag.get("name") or tag.get("itemprop") or ""
            ).lower()
            if key in {"og:url", "twitter:url", "al:web:url", "url"}:
                urls.add(tag.get("content") or "")

    for script in soup.find_all("script", {"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        urls.update(iter_json_values(data))

    for match in re.finditer(
        r"""["'](?:url|mainEntityOfPage|@id|original_url|reqUrl)["']\s*[:=]\s*["']([^"']+)""",
        html_text[:500_000],
    ):
        value = match.group(1)
        if value.startswith("http"):
            urls.add(value)
        elif value.startswith("/"):
            host = path.parent.name.replace("-", ".")
            urls.add(f"https://{host}{value}")

    for match in re.finditer(
        r"""original_url["'\]]*\s*=\s*["'](https?://[^"']+)""", html_text[:500_000]
    ):
        urls.add(match.group(1))

    saved_url_match = re.search(
        r"<!--\s*saved_url:\s*(https?://.+?)\s*-->", html_text[:2000], re.IGNORECASE
    )
    if saved_url_match:
        urls.add(saved_url_match.group(1))

    return {u for u in urls if u and u.startswith("http")}, title


def load_html_candidates(html_dir: Path) -> list[HtmlCandidate]:
    candidates: list[HtmlCandidate] = []
    for path in sorted(html_dir.rglob("*.html")):
        try:
            html_text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        urls, title = extract_urls_and_title(path, html_text)
        normalized = {normalize_url(u) for u in urls}
        normalized.discard("")
        candidates.append(
            HtmlCandidate(
                path=path,
                title=title,
                urls=urls,
                normalized_urls=normalized,
                text_for_matching=f"{title} {path.stem}".lower(),
            )
        )
    return candidates


def choose_best_candidate(
    url: str,
    candidates_by_url: dict[str, list[HtmlCandidate]],
    candidates_by_dir: dict[str, list[HtmlCandidate]],
) -> tuple[HtmlCandidate | None, str, float]:
    normalized = normalize_url(url)
    exact = candidates_by_url.get(normalized, [])
    if exact:
        best = sorted(exact, key=lambda c: (len(c.normalized_urls), str(c.path)))[0]
        return best, "metadata", 1.0

    target_host_dir = host_dir_name(url)
    same_host = candidates_by_dir.get(target_host_dir, [])
    if not same_host and target_host_dir.startswith("www-"):
        same_host = candidates_by_dir.get(target_host_dir[4:], [])
    if not same_host:
        host_tokens = {
            t
            for t in re.findall(r"[a-z0-9]{3,}", target_host_dir)
            if t not in {"www", "com", "org", "net", "news"}
        }
        for dirname, dir_candidates in candidates_by_dir.items():
            dir_tokens = {
                t
                for t in re.findall(r"[a-z0-9]{3,}", dirname)
                if t not in {"www", "com", "org", "net", "news"}
            }
            if host_tokens & dir_tokens:
                same_host.extend(dir_candidates)
    if not same_host:
        return None, "none", 0.0

    url_tokens = url_slug_tokens(url)
    if not url_tokens:
        return None, "none", 0.0

    scored: list[tuple[float, HtmlCandidate]] = []
    for candidate in same_host:
        tokens = path_tokens(candidate.path) | set(
            re.findall(r"[a-z0-9]{3,}", candidate.title.lower())
        )
        overlap = len(url_tokens & tokens)
        score = overlap / max(1, len(url_tokens))
        if score:
            scored.append((score, candidate))

    if not scored:
        if len(same_host) == 1 and re.search(
            r"\b(site not found|page not found|404|wordpress error|subscribe to read)\b",
            same_host[0].title,
            re.I,
        ):
            return same_host[0], "host_failure", 0.1
        return None, "none", 0.0

    scored.sort(key=lambda item: (-item[0], str(item[1].path)))
    score, candidate = scored[0]
    if score >= 0.45:
        return candidate, "slug", score
    return None, "none", score


def preprocess_html(html_text: str) -> str:
    """Remove known non-article elements before extraction.

    Strips navigation, sidebars, and voting UI that confuse trafilatura,
    so the main article body is what gets extracted.
    """
    soup = BeautifulSoup(html_text, "lxml")

    # Generic structural noise: navigation and sidebars
    for tag in soup.find_all(["aside", "nav"]):
        tag.decompose()

    # The Register: "related articles" section appended after the article body
    for tag in soup.find_all("section", class_="related"):
        tag.decompose()

    # old.reddit.com: voting controls produce "NNN pointsNNN points" artefacts
    for tag in soup.find_all("div", class_="midcol"):
        tag.decompose()
    for tag in soup.find_all("span", class_=re.compile(r"\bscore\b")):
        tag.decompose()

    # old.reddit.com: sidebar (subreddit rules, related subreddits, moderators list, etc.)
    for tag in soup.find_all("div", class_="side"):
        tag.decompose()

    # Comment sections on tech news sites (Tom's Hardware, Ars Technica, etc.)
    # Trafilatura's favor_recall=True otherwise pulls in user comments.
    for tag in soup.find_all(
        ["section", "div", "article"],
        id=re.compile(r"^comments?", re.I),
    ):
        tag.decompose()
    for tag in soup.find_all(
        ["section", "div"],
        class_=re.compile(
            r"\bcomments?[-_]?(wrapper|section|list|thread|block|area|container)?\b",
            re.I,
        ),
    ):
        tag.decompose()

    return str(soup)


def markdown_from_html(html_text: str, url: str) -> str:
    cleaned_html = preprocess_html(html_text)
    markdown = trafilatura.extract(
        cleaned_html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        include_links=True,
        favor_recall=True,
        deduplicate=True,
        output_format="markdown",
    )
    return deduplicate_markdown(clean_markdown(markdown or ""))


def clean_markdown(markdown: str) -> str:
    for pattern, replacement in INLINE_NOISE_SUBSTITUTIONS:
        markdown = pattern.sub(replacement, markdown)
    lines: list[str] = []
    prev_heading_text: str = ""
    for raw_line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lowered = line.lower().strip("*_# ")
        if any(re.search(pattern, lowered, re.I) for pattern in FOOTER_STOP_PATTERNS):
            break
        if any(re.search(pattern, lowered, re.I) for pattern in NOISE_LINE_PATTERNS):
            continue
        # Drop a heading whose text is identical to the immediately preceding heading
        # (NPR and similar CMSs emit the title at two heading levels).
        if re.match(r"^#{1,6}\s", line):
            heading_text = re.sub(r"^#{1,6}\s+", "", line).strip()
            if heading_text == prev_heading_text:
                continue
            prev_heading_text = heading_text
        else:
            prev_heading_text = ""
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def deduplicate_markdown(markdown: str) -> str:
    """Remove verbatim duplicate of the article body when trafilatura extracts it twice.

    Some sites (certain Substack themes, wheresyoured.at) embed the full article in two
    separate DOM nodes.  Trafilatura then concatenates both, producing an exact repeat.
    We detect this by searching for a 400-character probe taken from the early body in
    the second half of the text.
    """
    if len(markdown) < 800:
        return markdown
    probe_start = min(100, len(markdown) // 8)
    probe_end = min(probe_start + 400, len(markdown) // 2)
    probe = markdown[probe_start:probe_end]
    pos = markdown.find(probe, probe_end)
    if pos != -1:
        return markdown[:pos].rstrip()
    return markdown


def truncate_markdown(markdown: str, max_words: int = 3000) -> str:
    words = list(re.finditer(r"\w+", markdown))
    if len(words) <= max_words:
        return markdown

    cutoff_index = words[max_words - 1].end()
    text_to_cut = markdown[:cutoff_index]

    last_match_end = -1
    for match in re.finditer(r"[.!?]['\"\]\)]*(?:\s+|$)", text_to_cut):
        last_match_end = match.end()

    if last_match_end != -1:
        truncated = text_to_cut[:last_match_end].strip()
    else:
        truncated = text_to_cut.strip()

    return truncated + "\n\n<document_shortened>"


def classify_failure(title: str, markdown: str, min_words: int) -> tuple[str, str]:
    title_lower = title.lower()
    leading = markdown[:1200].lower()
    joined_barrier_area = f"{title_lower}\n{leading}"
    words = re.findall(r"\w+", markdown)
    word_count = len(words)

    first_line = next(
        (line.strip() for line in markdown.split("\n") if line.strip()), ""
    )
    if re.match(r"^#{1,3}\s*most popular$", first_line, re.I):
        return (
            "fetch_failed",
            "sidebar/navigation content extracted (MOST POPULAR header)",
        )

    if re.search(
        r"we process your data to deliver content or advertisements", leading, re.I
    ):
        return "fetch_failed", "GDPR/cookie consent banner extracted instead of article"

    if re.search(
        r"please click the box below to let us know you.re not a robot", leading, re.I
    ):
        return "fetch_failed", "CAPTCHA/robot-check page extracted"

    if re.search(r"select your language", leading, re.I) and re.search(
        r"official languages", leading, re.I
    ):
        return "fetch_failed", "language-selector page extracted, no article content"

    if re.search(r"\d+ points\d+ points\d+ points", leading):
        return "fetch_failed", "Reddit vote-counter UI extracted without comment text"

    if re.search(
        r"press shift question mark to access a list of keyboard shortcuts",
        leading,
        re.I,
    ):
        return (
            "fetch_failed",
            "video-player keyboard-shortcut page extracted, no article content",
        )

    if re.search(
        r"\bjavascript is not available\b|\bjavascript is required\b",
        joined_barrier_area,
        re.I,
    ):
        return (
            "fetch_failed",
            "fetch failure text matched JavaScript-disabled barrier",
        )

    if word_count >= min_words:
        if (
            re.search(
                r"\b(subscribe to read|page not found|site not found|wordpress error)\b",
                title_lower,
                re.I,
            )
            and word_count < 500
        ):
            for pattern in PAYWALL_PATTERNS:
                if re.search(pattern, joined_barrier_area, re.I):
                    return "paywall", f"paywall/login barrier text matched {pattern!r}"
            for pattern in FETCH_FAILURE_PATTERNS:
                if re.search(pattern, joined_barrier_area, re.I):
                    return "fetch_failed", f"fetch failure text matched {pattern!r}"
        return "ok", ""

    host_hint = title_lower
    if word_count >= 15 and any(
        hint in host_hint for hint in [" on x", "twitter", "mastodon", "bsky"]
    ):
        return "ok", ""

    for pattern in PAYWALL_PATTERNS:
        if re.search(pattern, joined_barrier_area, re.I):
            return "paywall", f"paywall/login barrier text matched {pattern!r}"
    for pattern in LOGIN_PATTERNS:
        if re.search(pattern, joined_barrier_area, re.I):
            return "login", f"login/consent barrier text matched {pattern!r}"
    for pattern in FETCH_FAILURE_PATTERNS:
        if re.search(pattern, joined_barrier_area, re.I):
            return "fetch_failed", f"fetch failure text matched {pattern!r}"
    return "too_short", f"extracted only {word_count} words"


def markdown_path_for_url(url: str, by_url_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return by_url_dir / f"{digest}.md"


def process_url(
    url: str,
    candidate: HtmlCandidate | None,
    match_method: str,
    match_score: float,
    out_dir: Path,
    min_words: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "url": url,
        "normalized_url": normalize_url(url),
        "status": "no_html",
        "reason": "no saved HTML candidate matched this URL",
        "match_method": match_method,
        "match_score": round(match_score, 3),
        "source_html": None,
        "markdown_path": None,
        "title": "",
        "word_count": 0,
    }
    if candidate is None:
        return record

    try:
        html_text = candidate.path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        record.update(
            {
                "status": "fetch_failed",
                "reason": f"could not read HTML: {exc}",
                "source_html": str(candidate.path),
                "title": candidate.title,
            }
        )
        return record

    markdown = markdown_from_html(html_text, url)
    markdown = truncate_markdown(markdown, max_words=3000)
    status, reason = classify_failure(candidate.title, markdown, min_words)
    word_count = len(re.findall(r"\w+", markdown))
    record.update(
        {
            "status": status,
            "reason": reason,
            "source_html": str(candidate.path),
            "title": candidate.title,
            "word_count": word_count,
        }
    )

    if status == "ok":
        by_url_dir = out_dir / "by-url"
        md_path = markdown_path_for_url(url, by_url_dir)
        md_path.write_text(markdown + "\n", encoding="utf-8")
        record["markdown_path"] = str(md_path)
    return record


def print_spotchecks(records: list[dict[str, Any]], limit: int) -> None:
    import random

    ok_records = [r for r in records if r["status"] == "ok" and r.get("markdown_path")]
    if not ok_records:
        print("\nSpotcheck: no OK markdown records to inspect.")
        return
    k = min(max(0, limit), len(ok_records))
    selected = random.sample(ok_records, k)
    print(f"\nSpotchecking {len(selected)} randomly sampled markdown files:")
    for record in selected:
        md_path = Path(record["markdown_path"])
        text = md_path.read_text(encoding="utf-8")
        preview = re.sub(r"\s+", " ", text[:500]).strip()
        print()
        print(f"- {record['url']}")
        print(f"  html: {record['source_html']}")
        print(f"  words: {record['word_count']}")
        print(f"  preview: {preview}")


def write_outputs(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by-url").mkdir(parents=True, exist_ok=True)

    url_index = {record["url"]: record for record in records}
    (out_dir / "url_index.json").write_text(
        json.dumps(url_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(record["status"] for record in records)
    lines = [
        "Full-text coverage report",
        "=========================",
        "",
        f"total_urls: {len(records)}",
        *[f"{status}: {counts[status]}" for status in sorted(counts)],
        "",
        "Failures:",
    ]
    for record in records:
        if record["status"] != "ok":
            lines.append(f"- {record['status']}: {record['url']} ({record['reason']})")
    (out_dir / "coverage_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles-file", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--html-dir", type=Path, default=DEFAULT_HTML_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--spotcheck", type=int, default=5)
    parser.add_argument(
        "--extra-url-files",
        nargs="*",
        type=Path,
        default=DEFAULT_EXTRA_URL_FILES,
        help="Additional URL list files (one URL per line) to merge in.",
    )
    args = parser.parse_args()

    if not args.articles_file.exists():
        sys.exit(f"Articles file not found: {args.articles_file}")
    if not args.html_dir.exists():
        sys.exit(f"HTML directory not found: {args.html_dir}")

    urls = [
        line.strip()
        for line in args.articles_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Merge extra URL files, deduplicating by normalized form.
    seen_normalized: set[str] = {normalize_url(u) for u in urls}
    seen_normalized.discard("")
    for extra_path in args.extra_url_files or []:
        if not extra_path.exists():
            print(f"Note: extra URL file not found, skipping: {extra_path}")
            continue
        # Support both plain text (one URL per line) and CSV (url column).
        extra_urls: list[str] = []
        if extra_path.suffix == ".csv":
            with extra_path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    url = (row.get("url") or "").strip()
                    if url:
                        extra_urls.append(url)
        else:
            extra_urls = [
                line.strip()
                for line in extra_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        added = 0
        for u in extra_urls:
            norm = normalize_url(u)
            if norm and norm not in seen_normalized:
                seen_normalized.add(norm)
                urls.append(u)
                added += 1
        print(f"  {extra_path.name}: {len(extra_urls)} total, {added} new unique")
    candidates = load_html_candidates(args.html_dir)

    candidates_by_url: dict[str, list[HtmlCandidate]] = defaultdict(list)
    candidates_by_dir: dict[str, list[HtmlCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_dir[candidate.path.parent.name].append(candidate)
        for normalized_url in candidate.normalized_urls:
            candidates_by_url[normalized_url].append(candidate)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "by-url").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for url in urls:
        candidate, method, score = choose_best_candidate(
            url, candidates_by_url, candidates_by_dir
        )
        records.append(
            process_url(url, candidate, method, score, args.out_dir, args.min_words)
        )

    write_outputs(records, args.out_dir)

    counts = Counter(record["status"] for record in records)
    print("Coverage:")
    print(f"  total_urls: {len(records)}")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    print(f"  output: {args.out_dir}")
    print_spotchecks(records, args.spotcheck)


if __name__ == "__main__":
    main()
