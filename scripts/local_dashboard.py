#!/usr/bin/env python3
"""
Serve a local LeetGPU judge dashboard.

Usage:
    python scripts/local_dashboard.py
    python scripts/local_dashboard.py --results local_judge_results.jsonl --port 8765
"""

from __future__ import annotations

import argparse
import json
import math
import re
from html import unescape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "local_judge_results.jsonl"
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def slug_title(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("_"))


def challenge_sort_key(item: dict) -> tuple[int, int, str]:
    return (
        DIFFICULTY_ORDER.get(item["difficulty"], 99),
        item["number"],
        item["name"].lower(),
    )


def load_challenges() -> list[dict]:
    challenges_root = REPO_ROOT / "challenges"
    challenges: list[dict] = []
    for difficulty in ("easy", "medium", "hard"):
        difficulty_dir = challenges_root / difficulty
        if not difficulty_dir.exists():
            continue
        for challenge_dir in difficulty_dir.iterdir():
            if not challenge_dir.is_dir():
                continue
            html_path = challenge_dir / "challenge.html"
            py_path = challenge_dir / "challenge.py"
            if not html_path.exists() or not py_path.exists():
                continue

            match = re.match(r"(\d+)_(.+)", challenge_dir.name)
            if match:
                number = int(match.group(1))
                slug = match.group(2)
            else:
                number = 0
                slug = challenge_dir.name

            py_source = read_text(py_path)
            name_match = re.search(r"name\s*=\s*[\"']([^\"']+)[\"']", py_source)
            name = name_match.group(1) if name_match else slug_title(slug)

            html = read_text(html_path)
            first_paragraph = re.search(
                r"<p\b[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL
            )
            description = strip_tags(
                first_paragraph.group(1) if first_paragraph else html
            )

            rel = challenge_dir.relative_to(REPO_ROOT).as_posix()
            challenges.append(
                {
                    "id": rel,
                    "path": rel,
                    "difficulty": difficulty,
                    "number": number,
                    "slug": slug,
                    "name": name,
                    "description": description,
                    "content": html,
                }
            )

    challenges.sort(key=challenge_sort_key)
    return challenges


def load_rows(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []

    rows: list[dict] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rows.append(
                    {
                        "run_id": "parse-error",
                        "challenge": "(invalid row)",
                        "language": "",
                        "ok": False,
                        "error": f"Invalid JSON at line {line_number}",
                    }
                )
                continue
            rows.append(row)
    return rows


def timing_value(row: dict) -> float | None:
    timing = row.get("timing")
    if not isinstance(timing, dict):
        return None
    value = timing.get("time_ms", timing.get("mean_ms"))
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def aggregate(rows: list[dict], challenges: list[dict]) -> dict:
    runs: dict[str, dict] = {}
    attempts: dict[tuple, dict] = {}

    for row in rows:
        run_id = row.get("run_id") or "legacy"
        run_started_at = row.get("run_started_at") or ""
        challenge = row.get("challenge") or ""
        language = row.get("language") or ""
        solution = row.get("solution") or row.get("source_path") or ""
        snapshot = row.get("code_snapshot")
        key = (run_id, challenge, language, solution, snapshot)

        runs.setdefault(
            run_id,
            {
                "run_id": run_id,
                "run_started_at": run_started_at,
                "rows": 0,
                "attempts": 0,
            },
        )
        runs[run_id]["rows"] += 1

        attempt = attempts.setdefault(
            key,
            {
                "run_id": run_id,
                "run_started_at": run_started_at,
                "challenge": challenge,
                "challenge_name": row.get("challenge_name")
                or challenge.rsplit("/", 1)[-1],
                "language": language,
                "code_sha256": row.get("code_sha256"),
                "code_bytes": row.get("code_bytes"),
                "code_lines": row.get("code_lines"),
                "skipped": False,
                "status": row.get("status"),
                "cases": 0,
                "failures": 0,
                "errors": [],
                "suites": set(),
                "times": [],
            },
        )

        if row.get("suite") is not None:
            attempt["cases"] += 1
            attempt["suites"].add(str(row.get("suite")))

        if row.get("skipped"):
            attempt["skipped"] = True
            attempt["status"] = row.get("status") or "skipped"

        if not row.get("ok", False):
            attempt["failures"] += 1

        if row.get("error"):
            attempt["errors"].append(str(row["error"]))

        time_ms = timing_value(row)
        if time_ms is not None:
            attempt["times"].append(time_ms)

    normalized_attempts: list[dict] = []
    for attempt in attempts.values():
        times = attempt.pop("times")
        suites = sorted(attempt.pop("suites"))
        attempt["suites"] = suites
        attempt["ok"] = attempt["cases"] > 0 and attempt["failures"] == 0
        attempt["mean_ms"] = sum(times) / len(times) if times else None
        attempt["best_case_ms"] = min(times) if times else None
        attempt["errors"] = attempt["errors"][:3]
        normalized_attempts.append(attempt)

    for run in runs.values():
        run["attempts"] = sum(
            1 for attempt in normalized_attempts if attempt["run_id"] == run["run_id"]
        )

    best_by_challenge: dict[str, dict] = {}
    for attempt in normalized_attempts:
        if not attempt["ok"] or attempt["mean_ms"] is None:
            continue
        current = best_by_challenge.get(attempt["challenge"])
        if current is None or attempt["mean_ms"] < current["mean_ms"]:
            best_by_challenge[attempt["challenge"]] = attempt

    attempt_count_by_challenge: dict[str, int] = {}
    correct_count_by_challenge: dict[str, int] = {}
    for attempt in normalized_attempts:
        challenge = attempt["challenge"]
        attempt_count_by_challenge[challenge] = (
            attempt_count_by_challenge.get(challenge, 0) + 1
        )
        if attempt["ok"]:
            correct_count_by_challenge[challenge] = (
                correct_count_by_challenge.get(challenge, 0) + 1
            )

    enriched_challenges = []
    for challenge in challenges:
        best = best_by_challenge.get(challenge["id"])
        enriched_challenges.append(
            {
                **challenge,
                "attempts": attempt_count_by_challenge.get(challenge["id"], 0),
                "correct_attempts": correct_count_by_challenge.get(challenge["id"], 0),
                "solved": best is not None,
                "best_mean_ms": best.get("mean_ms") if best else None,
            }
        )

    normalized_attempts.sort(
        key=lambda item: (
            item.get("run_started_at") or "",
            item.get("run_id") or "",
            item["challenge"],
            item["language"],
        ),
        reverse=True,
    )
    best = sorted(
        best_by_challenge.values(),
        key=lambda item: (
            item["challenge"],
            item["mean_ms"] if item["mean_ms"] is not None else math.inf,
        ),
    )

    return {
        "metrics": {
            "runs": len(runs),
            "challenges": len(challenges),
            "attempts": len(normalized_attempts),
            "correct_attempts": sum(1 for item in normalized_attempts if item["ok"]),
            "solved_challenges": sum(
                1 for item in enriched_challenges if item["solved"]
            ),
        },
        "runs": sorted(
            runs.values(),
            key=lambda item: item.get("run_started_at") or item["run_id"],
            reverse=True,
        ),
        "attempts": normalized_attempts,
        "best": best,
        "challenges": enriched_challenges,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeetGPU Local</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111314;
      --surface: #1b1b1c;
      --surface-2: #202225;
      --surface-3: #151719;
      --line: #243244;
      --line-soft: #25282d;
      --text: #f7f7f7;
      --muted: #b6c2d2;
      --muted-2: #8491a3;
      --cyan: #22f2ef;
      --cyan-soft: rgba(34, 242, 239, 0.18);
      --blue: #163153;
      --green: #00d66f;
      --green-soft: #073d21;
      --yellow: #f6dd33;
      --yellow-soft: #3f360d;
      --purple: #d7b4ff;
      --purple-soft: #2a183f;
      --bad: #ff7777;
      --bad-soft: #401a1a;
      --code: #222;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    button, input, select {
      font: inherit;
    }
    button {
      cursor: pointer;
    }
    .app {
      width: calc(100vw - 12px);
      max-width: none;
      margin: 0 auto;
    }
    .top-nav {
      height: 46px;
      display: flex;
      align-items: stretch;
      gap: 18px;
      border-bottom: 1px solid var(--line);
    }
    .nav-button {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 145px;
      padding: 0 24px;
      border: 0;
      border-bottom: 2px solid transparent;
      background: transparent;
      color: var(--muted);
      white-space: nowrap;
    }
    .nav-button svg {
      width: 16px;
      height: 16px;
    }
    .nav-button.active {
      color: var(--cyan);
      border-bottom-color: var(--cyan);
    }
    .view {
      padding: 33px 0 42px;
    }
    .list-header {
      display: grid;
      grid-template-columns: 1fr minmax(260px, 1fr) auto auto auto;
      align-items: center;
      gap: 16px;
      margin-bottom: 31px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .search-wrap {
      position: relative;
      min-width: 220px;
    }
    .search-wrap svg {
      position: absolute;
      left: 12px;
      top: 50%;
      width: 18px;
      height: 18px;
      transform: translateY(-50%);
      color: var(--muted-2);
    }
    .search-input,
    .filter-select {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line-soft);
      border-radius: 7px;
      background: var(--surface-3);
      color: var(--text);
      outline: none;
    }
    .search-input {
      padding: 0 12px 0 40px;
    }
    .search-input::placeholder {
      color: var(--muted-2);
    }
    .pill-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .pill {
      height: 34px;
      border: 0;
      border-radius: 999px;
      padding: 0 17px;
      background: var(--surface-3);
      color: var(--muted);
    }
    .pill.active {
      color: var(--text);
      background: #1d3554;
    }
    .filter-select {
      min-width: 178px;
      padding: 0 12px;
      color: var(--text);
    }
    .challenge-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 24px;
    }
    .challenge-card {
      min-height: 186px;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 25px;
      text-align: left;
      color: var(--text);
      transition: border-color 140ms ease, transform 140ms ease, background 140ms ease;
    }
    .challenge-card:hover {
      border-color: #2f5d8b;
      background: #1f2022;
      transform: translateY(-1px);
    }
    .badge-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }
    .difficulty {
      display: inline-flex;
      height: 25px;
      align-items: center;
      border-radius: 999px;
      padding: 0 9px;
      font-size: 14px;
      color: var(--green);
      background: var(--green-soft);
    }
    .difficulty.medium {
      color: #79b8ff;
      background: #102a4a;
    }
    .difficulty.hard {
      color: #ff9aa2;
      background: #42181e;
    }
    .solved-dot {
      display: inline-flex;
      width: 31px;
      height: 24px;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      color: var(--green);
      background: rgba(0, 214, 111, 0.12);
    }
    .solved-dot svg {
      width: 15px;
      height: 15px;
    }
    .challenge-card h2 {
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .challenge-card p {
      margin: 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.38;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .detail-shell {
      width: 100%;
      margin: 2px 0 0;
      background: var(--surface);
      border-radius: 6px 6px 0 0;
      padding: 24px 8px 0;
      min-height: calc(100vh - 48px);
    }
    .back-button {
      height: 34px;
      margin: 0 0 16px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface-2);
      color: var(--muted);
      padding: 0 12px;
    }
    .detail-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 18px;
    }
    .detail-tab {
      display: inline-flex;
      height: 32px;
      align-items: center;
      gap: 7px;
      border: 0;
      border-radius: 6px;
      padding: 0 13px;
      color: var(--text);
      background: var(--blue);
    }
    .detail-tab svg {
      width: 15px;
      height: 15px;
    }
    .detail-tab.leaderboard {
      background: var(--yellow-soft);
      color: var(--yellow);
    }
    .detail-tab.solutions {
      background: #12381f;
      color: #4af58d;
    }
    .detail-tab.discuss {
      background: var(--purple-soft);
      color: var(--purple);
    }
    .detail-tab.disabled {
      opacity: 0.7;
      cursor: default;
    }
    .detail-tab.active {
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.13);
    }
    .detail-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 36px;
      align-items: start;
    }
    .detail-side {
      position: sticky;
      top: 16px;
      min-width: 0;
      max-height: calc(100vh - 76px);
      overflow: auto;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: #17191b;
      padding: 14px;
    }
    .problem-content {
      min-width: 0;
      color: var(--text);
      font-size: 17px;
      line-height: 1.45;
    }
    .problem-content h1 {
      font-size: 26px;
    }
    .problem-content h2 {
      margin: 26px 0 12px;
      font-size: 23px;
      line-height: 1.25;
      letter-spacing: 0;
    }
    .problem-content p {
      margin: 0 0 16px;
    }
    .problem-content ul {
      margin: 0 0 18px;
      padding-left: 20px;
    }
    .problem-content pre,
    .panel pre {
      overflow: auto;
      margin: 12px 0 22px;
      border-radius: 7px;
      background: var(--code);
      padding: 18px;
      color: #f8fafc;
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      white-space: pre;
    }
    .problem-content code {
      border: 1px solid #384253;
      border-radius: 6px;
      background: #1d2025;
      padding: 1px 6px;
      color: #e6edf7;
      font-size: 0.9em;
    }
    .math-inline {
      white-space: nowrap;
    }
    .math-display {
      display: block;
      margin: 6px 0 12px;
      overflow-x: auto;
      overflow-y: hidden;
      max-width: 100%;
      text-align: center;
    }
    .math-matrix-wrap {
      display: inline-flex;
      align-items: stretch;
      gap: 4px;
      vertical-align: middle;
      width: max-content;
      max-width: 100%;
    }
    .math-bracket {
      width: 7px;
      flex: 0 0 7px;
      border-top: 2px solid currentColor;
      border-bottom: 2px solid currentColor;
    }
    .math-bracket.left {
      border-left: 2px solid currentColor;
    }
    .math-bracket.right {
      border-right: 2px solid currentColor;
    }
    .math-matrix {
      display: grid;
      align-items: center;
      column-gap: 18px;
      row-gap: 4px;
      padding: 1px 4px;
      font: inherit;
      line-height: 1.2;
    }
    .math-cell {
      min-width: 2.2em;
      text-align: center;
      white-space: nowrap;
    }
    .panel {
      margin-top: 0;
      padding: 0;
    }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 22px;
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }
    .panel-header h2 {
      margin: 0;
    }
    .language-switch {
      display: flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line-soft);
      border-radius: 7px;
      background: var(--surface-3);
      padding: 4px;
    }
    .language-option {
      height: 28px;
      border: 0;
      border-radius: 5px;
      background: transparent;
      color: var(--muted);
      padding: 0 11px;
    }
    .language-option.active {
      background: #17283d;
      color: #b9d7ff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 14px;
      color: var(--text);
    }
    .detail-side table {
      min-width: 0;
    }
    .detail-side .panel {
      overflow-x: visible;
    }
    .detail-side th,
    .detail-side td {
      padding: 10px 8px;
      font-size: 13px;
    }
    .rank-code {
      margin-top: 18px;
      border-top: 1px solid var(--line-soft);
      padding-top: 16px;
    }
    .rank-code h3 {
      margin: 0 0 10px;
      font-size: 16px;
      line-height: 1.25;
      letter-spacing: 0;
    }
    .rank-code pre {
      max-height: 420px;
      min-height: 180px;
      overflow: auto;
      margin: 0;
      border-radius: 7px;
      background: #101214;
      color: #e6edf7;
      padding: 14px;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      white-space: pre;
    }
    th, td {
      padding: 11px 10px;
      border-bottom: 1px solid var(--line-soft);
      vertical-align: middle;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-align: left;
    }
    th {
      color: var(--muted-2);
      font-weight: 700;
      background: #17191b;
    }
    .status {
      display: inline-flex;
      min-width: 70px;
      height: 25px;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      font-weight: 700;
      font-size: 12px;
    }
    .status.ok {
      color: var(--green);
      background: rgba(0, 214, 111, 0.13);
    }
    .status.fail {
      color: var(--bad);
      background: var(--bad-soft);
    }
    .status.skipped {
      color: var(--yellow);
      background: var(--yellow-soft);
    }
    .language-tag {
      display: inline-flex;
      height: 24px;
      align-items: center;
      border-radius: 999px;
      background: #17283d;
      color: #b9d7ff;
      padding: 0 8px;
      font-size: 12px;
      font-weight: 700;
    }
    .empty {
      padding: 28px 0;
      color: var(--muted-2);
      font-size: 16px;
    }
    .global-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }
    .global-card header {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line-soft);
    }
    .global-card h2 {
      margin: 0;
      font-size: 22px;
    }
    .hidden {
      display: none !important;
    }
    @media (max-width: 1080px) {
      .app {
        width: calc(100vw - 12px);
        max-width: none;
      }
      .list-header {
        grid-template-columns: 1fr;
        gap: 12px;
      }
      .challenge-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .detail-shell {
        width: 100%;
      }
      .detail-layout {
        grid-template-columns: 1fr;
      }
      .detail-side {
        position: static;
        max-height: none;
      }
    }
    @media (max-width: 680px) {
      .challenge-grid {
        grid-template-columns: 1fr;
      }
      .top-nav {
        overflow-x: auto;
      }
      .nav-button {
        min-width: max-content;
      }
      .challenge-card {
        min-height: 150px;
      }
      .panel-header {
        align-items: flex-start;
        flex-direction: column;
      }
      .language-switch {
        width: 100%;
        overflow-x: auto;
      }
      th:nth-child(4),
      td:nth-child(4),
      th:nth-child(5),
      td:nth-child(5) {
        display: none;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <nav class="top-nav" aria-label="Primary">
      <button class="nav-button active" id="navChallenges" type="button">
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M4 6h16M4 12h16M4 18h16M8 6v12M16 6v12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        Challenges
      </button>
      <button class="nav-button" id="navGlobal" type="button">
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M8 21h8M12 17v4M7 4h10v4a5 5 0 0 1-10 0V4Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M5 6H3a4 4 0 0 0 4 4M19 6h2a4 4 0 0 1-4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        Global Leaderboard
      </button>
    </nav>

    <section class="view" id="listView">
      <div class="list-header">
        <h1>Challenges</h1>
        <div class="search-wrap">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="m21 21-4.35-4.35M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <input class="search-input" id="search" placeholder="Search challenges...">
        </div>
        <div class="pill-row" id="difficultyFilters"></div>
        <select class="filter-select" id="statusFilter">
          <option value="all">All Challenges</option>
          <option value="solved">Solved</option>
          <option value="unsolved">Unsolved</option>
        </select>
      </div>
      <div class="challenge-grid" id="challengeGrid"></div>
      <div class="empty hidden" id="listEmpty">No challenges match the filters.</div>
    </section>

    <section class="view hidden" id="detailView"></section>

    <section class="view hidden" id="globalView">
      <div class="global-card">
        <header>
          <h2>Global Leaderboard</h2>
        </header>
        <div id="globalBoard"></div>
      </div>
    </section>
  </div>

  <script>
    const state = {
      data: null,
      difficulty: "all",
      activeTab: "submissions",
      activeChallenge: null,
      leaderboardLanguage: "cuda",
    };

    const $ = (id) => document.getElementById(id);

    const icons = {
      check: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="m8.5 12 2.3 2.3 4.9-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
      submissions: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.34-5.66M4 4v5h5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
      trophy: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M8 21h8M12 17v4M7 4h10v4a5 5 0 0 1-10 0V4Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 6H3a4 4 0 0 0 4 4M19 6h2a4 4 0 0 1-4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
      code: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m8 16-4-4 4-4M16 8l4 4-4 4M14 4l-4 16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
      chat: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M21 12a8 8 0 0 1-8 8H7l-4 3v-6.5A8 8 0 1 1 21 12Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M8 12h.01M12 12h.01M16 12h.01" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>`,
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function fmtMs(value) {
      return value === null || value === undefined ? "-" : `${Number(value).toFixed(4)} ms`;
    }

    function normalizeDifficulty(value) {
      return value ? value[0].toUpperCase() + value.slice(1) : "";
    }

    function challengeAttempts(challengeId) {
      return state.data.attempts.filter((item) => item.challenge === challengeId);
    }

    function challengeLeaderboard(challengeId, language) {
      return challengeAttempts(challengeId)
        .filter((item) => {
          if (!item.ok || item.mean_ms === null || item.mean_ms === undefined) return false;
          return item.language === language;
        })
        .sort((a, b) => a.mean_ms - b.mean_ms);
    }

    function renderLanguageSwitch() {
      const languages = [
        ["cuda", "CUDA"],
        ["triton", "Triton"],
        ["pytorch", "PyTorch"],
      ];
      return `<div class="language-switch" aria-label="Leaderboard language">
        ${languages.map(([value, label]) => `
          <button class="language-option ${state.leaderboardLanguage === value ? "active" : ""}" type="button" data-leaderboard-language="${value}">
            ${label}
          </button>
        `).join("")}
      </div>`;
    }

    function setActiveView(view) {
      $("listView").classList.toggle("hidden", view !== "list");
      $("detailView").classList.toggle("hidden", view !== "detail");
      $("globalView").classList.toggle("hidden", view !== "global");
      $("navChallenges").classList.toggle("active", view !== "global");
      $("navGlobal").classList.toggle("active", view === "global");
    }

    function renderDifficultyFilters() {
      $("difficultyFilters").innerHTML = ["all", "easy", "medium", "hard"].map((difficulty) => {
        const label = difficulty === "all" ? "All" : normalizeDifficulty(difficulty);
        const active = state.difficulty === difficulty ? " active" : "";
        return `<button class="pill${active}" type="button" data-difficulty="${difficulty}">${label}</button>`;
      }).join("");
      $("difficultyFilters").querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
          state.difficulty = button.dataset.difficulty;
          renderDifficultyFilters();
          renderChallengeGrid();
        });
      });
    }

    function filteredChallenges() {
      const query = $("search").value.trim().toLowerCase();
      const status = $("statusFilter").value;
      return state.data.challenges.filter((challenge) => {
        if (state.difficulty !== "all" && challenge.difficulty !== state.difficulty) return false;
        if (status === "solved" && !challenge.solved) return false;
        if (status === "unsolved" && challenge.solved) return false;
        if (query && !`${challenge.name} ${challenge.description} ${challenge.path}`.toLowerCase().includes(query)) {
          return false;
        }
        return true;
      });
    }

    function renderChallengeGrid() {
      const challenges = filteredChallenges();
      $("listEmpty").classList.toggle("hidden", challenges.length !== 0);
      $("challengeGrid").innerHTML = challenges.map((challenge) => `
        <button class="challenge-card" type="button" data-id="${escapeHtml(challenge.id)}">
          <span class="badge-row">
            <span class="difficulty ${escapeHtml(challenge.difficulty)}">${normalizeDifficulty(challenge.difficulty)}</span>
            ${challenge.solved ? `<span class="solved-dot" title="Solved">${icons.check}</span>` : ""}
          </span>
          <h2>${escapeHtml(challenge.name)}</h2>
          <p>${escapeHtml(challenge.description)}</p>
        </button>
      `).join("");
      $("challengeGrid").querySelectorAll(".challenge-card").forEach((card) => {
        card.addEventListener("click", () => openChallenge(card.dataset.id));
      });
    }

    function renderAttemptTable(attempts, options = {}) {
      if (!attempts.length) {
        return `<div class="empty">${options.empty || "No submissions yet."}</div>`;
      }
      const statusLabel = (item) => item.skipped ? "Skipped" : (item.ok ? "Accepted" : "Failed");
      const statusClass = (item) => item.skipped ? "skipped" : (item.ok ? "ok" : "fail");
      return `<table>
        <thead>
          <tr>
            ${options.rank ? `<th style="width:70px;">Rank</th>` : ""}
            <th style="width:100px;">Status</th>
            ${options.challenge ? `<th>Challenge</th>` : ""}
            <th style="width:112px;">Language</th>
            <th style="width:130px;">Mean</th>
            <th style="width:86px;">Cases</th>
            <th style="width:220px;">Run</th>
          </tr>
        </thead>
        <tbody>
          ${attempts.map((item, index) => `
            <tr>
              ${options.rank ? `<td>#${index + 1}</td>` : ""}
              <td><span class="status ${statusClass(item)}">${statusLabel(item)}</span></td>
              ${options.challenge ? `<td title="${escapeHtml(item.challenge)}">${escapeHtml(item.challenge_name || item.challenge)}</td>` : ""}
              <td><span class="language-tag">${escapeHtml(item.language || "-")}</span></td>
              <td>${fmtMs(item.mean_ms)}</td>
              <td>${item.cases ?? "-"}</td>
              <td title="${escapeHtml(item.run_started_at || item.run_id || "-")}">${escapeHtml(item.run_started_at || item.run_id || "-")}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>`;
    }

    function renderDetailPanel(challenge) {
      if (state.activeTab === "leaderboard") {
        const leaders = challengeLeaderboard(challenge.id, state.leaderboardLanguage);
        return `<section class="panel">
          <div class="panel-header">
            <h2>Leaderboard</h2>
            ${renderLanguageSwitch()}
          </div>
          ${renderAttemptTable(leaders, {
            rank: true,
            empty: `No accepted ${state.leaderboardLanguage} submissions for this challenge yet.`,
          })}
        </section>`;
      }
      return `<section class="panel">
        <h2>Submissions</h2>
        ${renderAttemptTable(challengeAttempts(challenge.id))}
      </section>`;
    }

    function renderDetail() {
      const challenge = state.activeChallenge;
      if (!challenge) return;
      $("detailView").innerHTML = `<div class="detail-shell">
        <button class="back-button" id="backToList" type="button">Back to Challenges</button>
        <div class="detail-layout">
          <article class="problem-content">
            <h1>${escapeHtml(challenge.name)}</h1>
            <p><span class="difficulty ${escapeHtml(challenge.difficulty)}">${normalizeDifficulty(challenge.difficulty)}</span></p>
            ${challenge.content}
          </article>
          <aside class="detail-side">
            <div class="detail-tabs">
              <button class="detail-tab ${state.activeTab === "submissions" ? "active" : ""}" type="button" data-tab="submissions">${icons.submissions}Submissions</button>
              <button class="detail-tab leaderboard ${state.activeTab === "leaderboard" ? "active" : ""}" type="button" data-tab="leaderboard">${icons.trophy}Leaderboard</button>
            </div>
            ${renderDetailPanel(challenge)}
          </aside>
        </div>
      </div>`;
      $("backToList").addEventListener("click", () => {
        location.hash = "";
      });
      $("detailView").querySelectorAll("[data-tab]").forEach((button) => {
        button.addEventListener("click", () => {
          const tab = button.dataset.tab;
          state.activeTab = tab;
          renderDetail();
        });
      });
      $("detailView").querySelectorAll("[data-leaderboard-language]").forEach((button) => {
        button.addEventListener("click", () => {
          state.leaderboardLanguage = button.dataset.leaderboardLanguage;
          renderDetail();
        });
      });
      renderMath($("detailView"));
    }

    function renderMath(element) {
      const pattern = /\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g;
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
          const parent = node.parentElement;
          if (!parent || parent.closest("code, pre, script, style, .math-inline, .math-display")) {
            return NodeFilter.FILTER_REJECT;
          }
          pattern.lastIndex = 0;
          return pattern.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
      });
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach((node) => {
        const fragment = document.createDocumentFragment();
        let lastIndex = 0;
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(node.nodeValue)) !== null) {
          if (match.index > lastIndex) {
            fragment.append(document.createTextNode(node.nodeValue.slice(lastIndex, match.index)));
          }
          fragment.append(createMathNode(match[1] ?? match[2], match[1] !== undefined));
          lastIndex = pattern.lastIndex;
        }
        if (lastIndex < node.nodeValue.length) {
          fragment.append(document.createTextNode(node.nodeValue.slice(lastIndex)));
        }
        node.parentNode.replaceChild(fragment, node);
      });
    }

    function createMathNode(source, display) {
      const container = document.createElement(display ? "span" : "span");
      container.className = display ? "math-display" : "math-inline";
      const matrix = parseBmatrix(source);
      if (matrix) {
        container.append(renderMatrix(matrix));
      } else {
        container.textContent = formatLatexText(source);
      }
      return container;
    }

    function parseBmatrix(source) {
      const match = source.match(/\\begin\{bmatrix\}([\s\S]*?)\\end\{bmatrix\}/);
      if (!match) return null;
      return match[1]
        .trim()
        .split(/\\\\/)
        .map((row) => row.trim())
        .filter(Boolean)
        .map((row) => row.split("&").map((cell) => formatLatexText(cell)));
    }

    function renderMatrix(rows) {
      const wrapper = document.createElement("span");
      wrapper.className = "math-matrix-wrap";
      const left = document.createElement("span");
      left.className = "math-bracket left";
      const right = document.createElement("span");
      right.className = "math-bracket right";
      const matrix = document.createElement("span");
      matrix.className = "math-matrix";
      const columns = Math.max(1, ...rows.map((row) => row.length));
      matrix.style.gridTemplateColumns = `repeat(${columns}, max-content)`;
      rows.forEach((row) => {
        row.forEach((cell) => {
          const value = document.createElement("span");
          value.className = "math-cell";
          value.textContent = cell;
          matrix.append(value);
        });
      });
      wrapper.append(left, matrix, right);
      return wrapper;
    }

    function formatLatexText(source) {
      return source
        .trim()
        .replace(/\\times/g, "\u00d7")
        .replace(/\\leq?/g, "\u2264")
        .replace(/\\geq?/g, "\u2265")
        .replace(/\\text\{([^}]*)\}/g, "$1")
        .replace(/[{}]/g, "")
        .replace(/\\/g, "");
    }

    function openChallenge(challengeId) {
      location.hash = `challenge/${encodeURIComponent(challengeId)}`;
    }

    function renderGlobalBoard() {
      const rows = [...state.data.best]
        .filter((item) => item.ok)
        .sort((a, b) => {
          if (a.challenge === b.challenge) return (a.mean_ms ?? Infinity) - (b.mean_ms ?? Infinity);
          return a.challenge.localeCompare(b.challenge);
        });
      $("globalBoard").innerHTML = renderAttemptTable(rows, {
        rank: true,
        challenge: true,
        empty: "No accepted submissions yet.",
      });
    }

    function route() {
      const hash = location.hash.replace(/^#/, "");
      if (hash.startsWith("challenge/")) {
        const challengeId = decodeURIComponent(hash.slice("challenge/".length));
        state.activeChallenge = state.data.challenges.find((item) => item.id === challengeId);
        state.activeTab = "submissions";
        state.leaderboardLanguage = "cuda";
        if (!state.activeChallenge) {
          location.hash = "";
          return;
        }
        renderDetail();
        setActiveView("detail");
        return;
      }
      if (hash === "leaderboard") {
        renderGlobalBoard();
        setActiveView("global");
        return;
      }
      state.activeChallenge = null;
      setActiveView("list");
      renderChallengeGrid();
    }

    async function load() {
      const response = await fetch("/api/summary");
      state.data = await response.json();
      renderDifficultyFilters();
      renderChallengeGrid();
      route();
    }

    $("search").addEventListener("input", renderChallengeGrid);
    $("statusFilter").addEventListener("input", renderChallengeGrid);
    $("navChallenges").addEventListener("click", () => {
      location.hash = "";
    });
    $("navGlobal").addEventListener("click", () => {
      location.hash = "leaderboard";
    });
    window.addEventListener("hashchange", route);
    load();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    results_path: Path = DEFAULT_RESULTS

    def log_message(self, format: str, *args) -> None:
        return

    def send_bytes(
        self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/summary":
            self.send_json(aggregate(load_rows(self.results_path), load_challenges()))
            return
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local judge dashboard.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    DashboardHandler.results_path = args.results.resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard: {url}")
    print(f"Results: {DashboardHandler.results_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
