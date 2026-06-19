from __future__ import annotations

import csv
import html
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent / "data"
ARTICLES_FILE = ROOT / "articles.txt"
EXPERT_URLS_CSV = ROOT / "expert_added_urls.csv"
MANUAL_URLS_CSV = ROOT / "manually_added_urls.csv"
FETCHED_DIR = ROOT / "full-texts"

CANONICAL_URL_PATTERN = re.compile(
    r"""<link\b(?=[^>]*\brel=["'][^"']*\bcanonical\b[^"']*["'])"""
    r"""(?=[^>]*\bhref=["'](https?://[^"']+)["'])[^>]*>""",
    re.IGNORECASE,
)
META_URL_PATTERN = re.compile(
    r"""<meta\b(?=[^>]*(?:property|name|itemprop)=["']"""
    r"""(?:og:url|twitter:url|al:web:url|url)["'])"""
    r"""(?=[^>]*\bcontent=["'](https?://[^"']+)["'])[^>]*>""",
    re.IGNORECASE,
)
SCRIPT_URL_PATTERN = re.compile(
    r"""["'](?:url|mainEntityOfPage|@id|original_url|reqUrl)["']"""
    r"""\s*[:=]\s*["'](https?://[^"']+)["']""",
    re.IGNORECASE,
)
SAVED_URL_PATTERN = re.compile(
    r"<!--\s*saved_url:\s*(https?://.+?)\s*-->",
    re.IGNORECASE,
)


def normalize_url(raw_url: str) -> str:
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


def get_fetched_urls() -> set[str]:
    fetched_urls: set[str] = set()
    if not FETCHED_DIR.exists():
        return fetched_urls

    for path in FETCHED_DIR.rglob("*.html"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")[:500_000]
        except OSError as exc:
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            continue
        urls = (
            SAVED_URL_PATTERN.findall(content)
            + CANONICAL_URL_PATTERN.findall(content)
            + META_URL_PATTERN.findall(content)
            + SCRIPT_URL_PATTERN.findall(content)
        )
        for url in urls:
            normalized = normalize_url(url)
            if normalized:
                fetched_urls.add(normalized)
    return fetched_urls


def load_url_file(path: Path) -> list[str]:
    """Read a plain text file with one URL per line."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_csv_urls(path: Path) -> list[str]:
    """Read a CSV and return the 'url' column values."""
    if not path.exists():
        return []
    urls: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            if url:
                urls.append(url)
    return urls


fetched_urls = get_fetched_urls()
print(f"Found {len(fetched_urls)} already fetched articles.")

# Merge URLs from all sources, deduplicating by normalized form.
seen: set[str] = set()
urls: list[str] = []
for src_label, src_urls in [
    (ARTICLES_FILE.name, load_url_file(ARTICLES_FILE)),
    (EXPERT_URLS_CSV.name, load_csv_urls(EXPERT_URLS_CSV)),
    (MANUAL_URLS_CSV.name, load_csv_urls(MANUAL_URLS_CSV)),
]:
    added = 0
    for url in src_urls:
        norm = normalize_url(url)
        if norm and norm not in seen:
            seen.add(norm)
            urls.append(url)
            added += 1
    print(f"  {src_label}: {len(src_urls)} total, {added} new unique")

to_open = [url for url in urls if normalize_url(url) not in fetched_urls]
print(
    f"Total URLs: {len(urls)}, To Open: {len(to_open)} (Skipped {len(urls) - len(to_open)})"
)

for i, url in enumerate(to_open):
    subprocess.Popen(["open", "-a", "Firefox", url])
    print(f"Opened {i + 1}/{len(to_open)}: {url}")
    time.sleep(1.5)
