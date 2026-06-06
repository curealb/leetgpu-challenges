#!/usr/bin/env python3
"""Shortcut for running local_judge.py by challenge number."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGES_ROOT = REPO_ROOT / "challenges"
LOCAL_JUDGE = REPO_ROOT / "scripts" / "local_judge.py"
LOCAL_JUDGE_RESULTS = REPO_ROOT / "local_judge_results.jsonl"
CONDA_ENV_NAME = "leetgpu"
CONDA_ENV_PYTHONS = (
    Path(f"/home/curealb/env/miniconda3/envs/{CONDA_ENV_NAME}/bin/python"),
    Path(f"/home/curealb/anaconda3/envs/{CONDA_ENV_NAME}/bin/python"),
    Path(f"/home/curealb/miniconda3/envs/{CONDA_ENV_NAME}/bin/python"),
)
LANGUAGE_ALIASES = {
    "cuda": "cuda",
    "triton": "triton",
    "trition": "triton",
    "pytorch": "pytorch",
    "torch": "pytorch",
}


def solution_candidates(challenge_dir: Path, language: str) -> list[Path]:
    starter_dir = challenge_dir / "starter"
    if language == "cuda":
        return [starter_dir / "solution.cu", starter_dir / "starter.cu"]
    if language == "triton":
        return [
            starter_dir / "solution.triton.py",
            starter_dir / "solution.py",
            starter_dir / "starter.triton.py",
        ]
    if language == "pytorch":
        return [
            starter_dir / "solution.pytorch.py",
            starter_dir / "solution.py",
            starter_dir / "starter.pytorch.py",
        ]
    raise ValueError(f"Unsupported language: {language}")


def find_solution(challenge_dir: Path, language: str) -> Path:
    candidates = solution_candidates(challenge_dir, language)
    for path in candidates:
        if path.exists():
            return path

    pretty = "\n".join(f"  - {path.relative_to(REPO_ROOT)}" for path in candidates)
    raise SystemExit(f"No {language} solution found. Expected one of:\n{pretty}")


def find_challenge(number: str, difficulty: str | None) -> Path:
    if not number.isdigit():
        raise SystemExit(f"Challenge number must be an integer, got: {number}")

    roots = [CHALLENGES_ROOT / difficulty] if difficulty else sorted(CHALLENGES_ROOT.iterdir())
    matches: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        matches.extend(sorted(path for path in root.glob(f"{number}_*") if path.is_dir()))

    if not matches:
        scope = f" under challenges/{difficulty}" if difficulty else ""
        raise SystemExit(f"No challenge found for number {number}{scope}")

    if len(matches) > 1:
        pretty = "\n".join(f"  - {path.relative_to(REPO_ROOT)}" for path in matches)
        raise SystemExit(
            f"Multiple challenges matched number {number}; pass --difficulty.\n{pretty}"
        )

    return matches[0]


def normalize_language(language: str) -> str:
    normalized = LANGUAGE_ALIASES.get(language.lower())
    if normalized is None:
        choices = ", ".join(sorted({"cuda", "triton", "pytorch"}))
        raise SystemExit(f"Unknown language {language!r}; choose one of: {choices}")
    return normalized


def can_import_torch(python: Path) -> bool:
    completed = subprocess.run(
        [str(python), "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def judge_python() -> str:
    current = Path(sys.executable)
    if can_import_torch(current):
        return str(current)
    for candidate in CONDA_ENV_PYTHONS:
        if candidate.exists() and can_import_torch(candidate):
            return str(candidate)
    return str(current)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_rows(rows: list[dict[str, object]], path: Path, overwrite: bool) -> None:
    if not rows:
        return
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def print_summary(rows: list[dict[str, object]]) -> None:
    case_rows = [row for row in rows if "suite" in row and "case_index" in row]
    skipped_rows = [row for row in rows if row.get("skipped")]
    for row in skipped_rows:
        challenge = row.get("challenge")
        language = row.get("language")
        error = row.get("error")
        print(f"\nSkipped {challenge} [{language}]: {error}")
    if not case_rows:
        if not skipped_rows:
            print("\nNo case results were produced.")
        return

    passed = sum(1 for row in case_rows if row.get("ok") is True)
    total = len(case_rows)

    print("\nRun results:")
    for row in case_rows:
        suite = row.get("suite")
        case_index = row.get("case_index")
        ok = row.get("ok") is True
        status = "PASS" if ok else "FAIL"
        timing = row.get("timing")
        suffix = ""
        if isinstance(timing, dict) and isinstance(timing.get("mean_ms"), (int, float)):
            suffix = f" mean={timing['mean_ms']:.4f}ms"
        print(f"  {suite}[{case_index}]: {status}{suffix}")

    print(f"Passed: {passed}/{total}")

    if passed == total:
        timings = [
            timing["mean_ms"]
            for row in case_rows
            if isinstance((timing := row.get("timing")), dict)
            and isinstance(timing.get("mean_ms"), (int, float))
        ]
        if timings:
            print(f"Average mean time: {sum(timings) / len(timings):.4f}ms")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a LeetGPU challenge locally by number.",
        epilog="Examples: scripts/judge.py 1 cuda | scripts/judge.py 41 torch --quick",
    )
    parser.add_argument("number", help="Challenge number, for example 1 or 22")
    parser.add_argument("language", help="cuda, triton, pytorch, or torch")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="Disambiguate if the same number appears in multiple difficulties",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only example,functional instead of example,functional,performance",
    )
    parser.add_argument(
        "--use-starter",
        action="store_true",
        help="Run starter/starter.<language>.* instead of solution/",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue after failures")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite local_judge_results.jsonl")
    args, extra_args = parser.parse_known_args()

    challenge_dir = find_challenge(args.number, args.difficulty)
    language = normalize_language(args.language)
    solution_path = None if args.use_starter else find_solution(challenge_dir, language)
    suites = "example,functional" if args.quick else "example,functional,performance"

    cmd = [
        judge_python(),
        str(LOCAL_JUDGE),
        str(challenge_dir),
        "--language",
        language,
        "--suites",
        suites,
    ]
    if args.use_starter:
        cmd.append("--use-starter")
    else:
        cmd.extend(["--solution", str(solution_path)])
    if args.keep_going:
        cmd.append("--keep-going")
    if args.overwrite:
        cmd.append("--overwrite")
    user_output = False
    if extra_args:
        extra_args = extra_args[1:] if extra_args[0] == "--" else extra_args
        user_output = any(arg == "--output" or arg.startswith("--output=") for arg in extra_args)
        cmd.extend(extra_args)

    print(f"Running {challenge_dir.relative_to(REPO_ROOT)} [{language}]", flush=True)
    if solution_path is not None:
        print(f"Solution: {solution_path.relative_to(REPO_ROOT)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="leetgpu-judge-") as tmp:
        summary_path = Path(tmp) / "results.jsonl"
        if not user_output:
            cmd.extend(["--output", str(summary_path), "--overwrite"])

        status = subprocess.call(cmd, cwd=REPO_ROOT)
        rows = load_rows(summary_path) if not user_output else []
        if rows:
            append_rows(rows, LOCAL_JUDGE_RESULTS, overwrite=args.overwrite)
            print_summary(rows)
        elif user_output:
            print("\nSummary skipped because a custom --output was passed through.")
        return status


if __name__ == "__main__":
    raise SystemExit(main())
