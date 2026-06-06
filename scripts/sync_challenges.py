#!/usr/bin/env python3
"""Sync only the challenge set from an upstream LeetGPU repository."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_PATH = "challenges"
SUPPORTED_LOCAL_LANGUAGES = ("cuda", "triton", "pytorch")
UNSUPPORTED_STARTERS = (
    "starter/starter.cute.py",
    "starter/starter.jax.py",
    "starter/starter.mojo",
)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def remote_exists(remote: str) -> bool:
    return git("remote", "get-url", remote, check=False).returncode == 0


def ensure_remote(remote: str, upstream_url: str | None) -> None:
    if remote_exists(remote):
        return
    if not upstream_url:
        raise SystemExit(
            f"Remote {remote!r} is not configured. Add it first, or pass --upstream-url."
        )
    git("remote", "add", remote, upstream_url)
    print(f"Added remote {remote}: {upstream_url}")


def changed_paths(pathspec: str) -> list[str]:
    result = git("status", "--porcelain", "--", pathspec)
    return [line for line in result.stdout.splitlines() if line.strip()]


def ensure_clean_challenges(force: bool) -> None:
    changes = changed_paths(SYNC_PATH)
    if not changes or force:
        return
    preview = "\n".join(f"  {line}" for line in changes[:20])
    extra = "" if len(changes) <= 20 else f"\n  ... and {len(changes) - 20} more"
    raise SystemExit(
        f"{SYNC_PATH}/ has local changes. Commit or stash them before syncing, "
        f"or rerun with --force.\n{preview}{extra}"
    )


def prune_unsupported_starters() -> None:
    removed = 0
    for challenge_dir in (REPO_ROOT / SYNC_PATH).glob("*/*"):
        if not challenge_dir.is_dir():
            continue
        for rel_path in UNSUPPORTED_STARTERS:
            path = challenge_dir / rel_path
            if path.exists():
                path.unlink()
                removed += 1
    print(f"Pruned {removed} unsupported starter file(s).")


def print_dry_run(ref: str) -> None:
    result = git("diff", "--name-status", "HEAD", ref, "--", SYNC_PATH)
    output = result.stdout.strip()
    if output:
        print(output)
    else:
        print(f"No upstream changes detected under {SYNC_PATH}/.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch an upstream LeetGPU repository and restore only challenges/. "
            "Local scripts, dashboards, docs, and solutions are left untouched."
        )
    )
    parser.add_argument("--remote", default="upstream", help="Git remote to sync from")
    parser.add_argument("--branch", default="main", help="Upstream branch to sync from")
    parser.add_argument(
        "--upstream-url",
        help="Add --remote with this URL if it is not already configured",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Use the already fetched remote ref without running git fetch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show upstream changes under challenges/ without modifying files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting local changes under challenges/",
    )
    args = parser.parse_args()

    ensure_remote(args.remote, args.upstream_url)
    if not args.no_fetch:
        git("fetch", args.remote, f"{args.branch}:refs/remotes/{args.remote}/{args.branch}")

    ref = f"{args.remote}/{args.branch}"
    if args.dry_run:
        print_dry_run(ref)
        return 0

    ensure_clean_challenges(args.force)
    git("restore", "--source", ref, "--", SYNC_PATH)
    prune_unsupported_starters()

    languages = ", ".join(SUPPORTED_LOCAL_LANGUAGES)
    print(f"Synced {SYNC_PATH}/ from {ref}. Local judging still targets: {languages}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
