"""
Executes pre-rendered queries against target LLMs using OpenRouter.

Reads:  data/jobs.json (produced by preflight_queries.py)
Writes: data/responses/<source>_<storyid>__<modelslug>__t<NN>__r<L>.json
        data/responses/responses.jsonl

Usage:
    python query_targets.py --jobs-file data/jobs.json
    python query_targets.py --sample 8 --out-dir data/responses_random_spotcheck
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OR_BASE = "https://openrouter.ai/api/v1/chat/completions"
RESPONSES_DIR = Path("data/responses")
DEFAULT_JOBS_FILE = Path("data/jobs.json")
MAX_TARGET_ATTEMPTS = 3


def _slugify(s: str) -> str:
    return re.sub(r"[^\w.\-]", "_", s)


def response_path(
    story: dict,
    target_model: str,
    template_id: int,
    rep: int,
    out_dir: Path = RESPONSES_DIR,
) -> Path:
    sid = _slugify(story["id"])
    mid = _slugify(target_model)
    return out_dir / (
        f"{story['source']}_{sid}__{mid}__t{template_id:02d}__r{rep}.json"
    )


def call_target(
    prompt: str,
    model: str,
    api_key: str,
    *,
    retries: int = 3,
    max_tokens: int = 3000,
    temperature: float = 0.7,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/partiality",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OR_BASE, headers=headers, json=payload, timeout=90)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2**attempt))
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {"completion": content, "usage": usage, "error": None}
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2**attempt)
    return {"completion": "", "usage": {}, "error": last_err}


def _attempt_count(record: dict) -> int:
    raw = record.get("target_attempts")
    if isinstance(raw, int) and raw > 0:
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    if record.get("error"):
        return 1
    return 0


def _job_response_path(job: dict, out_dir: Path) -> Path:
    story_stub = {"id": job["story_id"], "source": job["story_source"]}
    return response_path(
        story_stub, job["target_model"], job["template_id"], job["rep"], out_dir
    )


def should_run_job(job: dict, out_dir: Path) -> bool:
    out = _job_response_path(job, out_dir)
    if not out.exists():
        return True
    try:
        record = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    if not record.get("error"):
        return False
    return _attempt_count(record) < MAX_TARGET_ATTEMPTS


def subsample_jobs(
    jobs: list[dict],
    n: int,
    *,
    seed: int | None = None,
) -> list[dict]:
    if n <= 0:
        raise ValueError("--sample must be positive")
    if n >= len(jobs):
        return list(jobs)
    rng = random.Random(seed)
    return rng.sample(jobs, n)


def run_preflight(jobs_file: Path) -> None:
    cmd = [sys.executable, "preflight_queries.py", "--out-file", str(jobs_file)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_job(job: dict, api_key: str, out_dir: Path) -> dict:
    out = _job_response_path(job, out_dir)
    previous_attempts = 0
    if out.exists():
        try:
            record = json.loads(out.read_text(encoding="utf-8"))
            previous_attempts = _attempt_count(record)
            if not record.get("error"):
                if previous_attempts == 0:
                    record["target_attempts"] = 1
                    record["max_target_attempts"] = MAX_TARGET_ATTEMPTS
                    try:
                        out.write_text(
                            json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except OSError as e:
                        print(f"  Warning: could not save {out}: {e}", file=sys.stderr)
                return record
            if previous_attempts >= MAX_TARGET_ATTEMPTS:
                return record
        except (json.JSONDecodeError, OSError):
            pass

    res = call_target(job["prompt"], job["target_model"], api_key)
    record = {
        **job,
        "completion": res["completion"],
        "usage": res["usage"],
        "error": res["error"],
        "target_attempts": previous_attempts + 1,
        "max_target_attempts": MAX_TARGET_ATTEMPTS,
    }
    try:
        out.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as e:
        print(f"  Warning: could not save {out}: {e}", file=sys.stderr)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description="Execute pre-rendered queries.")
    p.add_argument("--jobs-file", type=Path, default=DEFAULT_JOBS_FILE)
    p.add_argument("--out-dir", type=Path, default=RESPONSES_DIR)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first few jobs and exit.",
    )
    p.add_argument(
        "--preflight",
        action="store_true",
        help="Regenerate --jobs-file via preflight_queries.py before running.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly subsample N jobs from the jobs file.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="RNG seed for --sample (default: 0).",
    )
    p.add_argument(
        "--write-sampled-jobs",
        type=Path,
        default=None,
        help="Write the subsampled job list to this path (default: data/jobs_<out-dir-stem>.json).",
    )
    p.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        run_preflight(args.jobs_file)

    if not args.jobs_file.exists():
        sys.exit(
            f"Jobs file {args.jobs_file} not found. Run preflight_queries.py first."
        )

    jobs = json.loads(args.jobs_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(jobs):,} jobs from {args.jobs_file}")

    if args.sample is not None:
        jobs = subsample_jobs(jobs, args.sample, seed=args.sample_seed)
        sampled_jobs_path = args.write_sampled_jobs
        if sampled_jobs_path is None:
            sampled_jobs_path = Path("data") / f"jobs_{args.out_dir.name}.json"
        sampled_jobs_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_jobs_path.write_text(
            json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Random subsample: {len(jobs):,} jobs (seed={args.sample_seed}) "
            f"-> {sampled_jobs_path}"
        )

    if args.dry_run:
        for j in jobs[:3]:
            print("\n" + "=" * 80)
            print(
                f"story={j['story_company']}/{j['story_id']}  "
                f"target={j['target_model']}  template={j['template_id']}  rep={j['rep']}"
            )
            print("-" * 80)
            print(j["prompt"])
        return

    if not args.api_key:
        sys.exit("Set OPENROUTER_API_KEY or pass --api-key.")

    pending = [j for j in jobs if should_run_job(j, args.out_dir)]
    print(f"Pending: {len(pending):,}  (already done: {len(jobs) - len(pending):,})")

    done = errors = 0
    manifest_path = args.out_dir / "responses.jsonl"
    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        manifest_path.open("a", encoding="utf-8") as manifest,
    ):
        futures = {
            pool.submit(run_job, j, args.api_key, args.out_dir): j for j in pending
        }
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                if rec.get("error"):
                    errors += 1
                manifest.write(
                    json.dumps(
                        {
                            "story_id": rec["story_id"],
                            "story_date": rec.get("story_date", ""),
                            "story_company": rec["story_company"],
                            "target_model": rec["target_model"],
                            "template_id": rec["template_id"],
                            "rep": rec["rep"],
                            "error": rec.get("error"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                manifest.flush()
            except Exception as e:
                errors += 1
                print(f"  unexpected error: {e}", file=sys.stderr)
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"  [{done:>5}/{len(futures)}]  errors: {errors}", flush=True)

    print(f"\nDone. {done} jobs, {errors} errors. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
