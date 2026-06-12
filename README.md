# LeetGPU

This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases,
and starter templates for CUDA, Triton, and PyTorch/Torch.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for CUDA, Triton, and PyTorch/Torch

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
challenge starter file (`starter.cu`, `starter.triton.py`, or `starter.pytorch.py`):

```bash
python scripts/local_judge.py 52 torch --quick
python scripts/local_judge.py 52 triton --quick
python scripts/local_judge.py 52 cuda --quick
```

Run every challenge against starter files for all locally supported languages:

```bash
python scripts/local_judge.py --all --language all --keep-going
```

Use `--overwrite` when you want a fresh results file instead of appending:

```bash
python scripts/local_judge.py 52 torch --quick --overwrite
```

By default, results are appended to `local_judge_results.jsonl`, with code
metadata and snapshots stored under `.local_judge/code/`. Use `--overwrite` for a
fresh results file.

Some challenges are PyTorch-only. For example, `41_simple_inference` passes a
`torch.nn.Module` into `solve(...)`, so CUDA/Triton local judging is marked as
unsupported instead of inventing unusable CUDA/Triton starters.

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
