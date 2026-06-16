# LeetGPU

This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases,
and starter templates for the locally supported languages it can run.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for supported local judging languages

## Local Judging

Use `scripts/local_judge.py` to run the tests defined in each `challenge.py`
locally. Local judging supports CUDA, Triton, and PyTorch only. The CLI also
accepts `torch` as an alias for `pytorch`.

Activate a CUDA-capable PyTorch environment first:

```bash
conda activate leetgpu
```

Quickly judge one challenge by number. `--quick` runs `example,functional`
instead of `example,functional,performance`. By default, local judging runs the
challenge starter file for the selected language:

```bash
python scripts/local_judge.py 52 torch --quick
python scripts/local_judge.py 52 triton --quick
python scripts/local_judge.py 52 cuda --quick
```

Run every challenge against starter files for all locally supported languages:

```bash
python scripts/local_judge.py --all --language all --keep-going
```

Starter file presence defines local language support for a challenge:

| Language | Required starter |
|----------|------------------|
| CUDA | `starter/starter.cu` |
| Triton | `starter/starter.triton.py` |
| PyTorch/Torch | `starter/starter.pytorch.py` |

If the selected language's starter file is missing, local judging records the
run as `unsupported` instead of failing the challenge. CUDA has one additional
requirement: every solve signature type from `challenge.py` must be compatible
with `ctypes`. Triton and PyTorch do not use the CUDA `ctypes` signature gate;
if their starter file exists, the local judge loads and runs it normally.

Use `--overwrite` when you want a fresh results file instead of appending:

```bash
python scripts/local_judge.py 52 torch --quick --overwrite
```

By default, results are appended to `local_judge_results.jsonl`, with code
metadata and snapshots stored under `.local_judge/code/`. Use `--overwrite` for a
fresh results file.

Some challenges are PyTorch-only. For example, `41_simple_inference` passes a
`torch.nn.Module` into `solve(...)`, so CUDA local judging is marked as
unsupported because the signature is not `ctypes` compatible. Languages without
a matching starter file are also marked as unsupported.

## Upstream Challenge Sync

Treat `challenges/` as upstream-owned data and keep local tooling outside that
tree. Sync only the challenge set from an upstream remote:

```bash
git remote add upstream <upstream-leetgpu-challenges-url>
python scripts/sync_challenges.py
```

If the remote is not configured yet, the script can add it for you:

```bash
python scripts/sync_challenges.py --upstream-url <upstream-leetgpu-challenges-url>
```

Preview challenge changes without touching files:

```bash
python scripts/sync_challenges.py --dry-run
```

The sync keeps the upstream problem definitions and automatically prunes starter
files outside CUDA, Triton, and PyTorch.

## Branch Workflow

Use `main` only for syncing upstream challenge updates, and use `solution` for
personal solutions, local judging, and solution commits.

Before syncing, make sure any solution work is committed on `solution`:

```bash
git switch solution
git status
git add .
git commit -m "save current solutions"
git push origin solution
```

If `git status` is clean, there is nothing to commit.

Sync upstream challenges into `main`:

```bash
git switch main
python scripts/sync_challenges.py
git add challenges
git commit -m "Sync challenges from upstream"
git push origin main
```

Bring the latest challenges into `solution` while keeping your committed
solutions:

```bash
git switch solution
git rebase main
```

If there are conflicts, keep the new `main` version for problem definitions such
as `challenge.py`, `challenge.html`, and tests. Keep your `solution` version for
answered starter files such as `starter/starter.cu`, `starter.triton.py`, and
`starter.pytorch.py`, then adjust the answer if the upstream signature or tests
changed.

After resolving each conflict:

```bash
git add <conflict-file>
git rebase --continue
```

If you prefer merge commits over rebasing, use this instead:

```bash
git switch solution
git merge main
```

Write, judge, and commit solutions on `solution`:

```bash
git add .
git commit -m "solve <challenge-name>"
git push -u origin solution
```

If `solution` is already pushed and you rebased it, update the remote branch
safely:

```bash
git push --force-with-lease origin solution
```

To avoid accidentally pushing to the upstream challenge repository, disable
pushes to the `upstream` remote:

```bash
git remote set-url --push upstream DISABLED
```

Start the local results dashboard:

```bash
python scripts/local_dashboard.py
```

Open `http://127.0.0.1:8765` to inspect local run history, correctness, timing,
and leaderboards. The dashboard intentionally does not expose or render solution
code.

If you need a different port:

```bash
python scripts/local_dashboard.py --port 9000
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new challenges or improvements.

## License

This problem set is licensed under [CC BY‑NC‑ND 4.0 license](LICENSE).

© 2025 AlphaGPU, LLC. Commercial use, redistribution, or derivative use is prohibited.
