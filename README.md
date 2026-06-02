# LeetGPU

This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases, and starter templates for multiple GPU programming frameworks.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for each supported framework

## Local Judging

Use `scripts/local_judge.py` to run the tests defined in each `challenge.py`
locally. Local judging supports CUDA, Triton, and PyTorch only. The CLI also
accepts `torch` as an alias for `pytorch`.

Activate a CUDA-capable PyTorch environment first:

```bash
conda activate torch
```

Quickly judge one challenge by number. `--quick` runs `example,functional`
instead of `example,functional,performance`, and `--use-starter` runs the
starter template:

```bash
python scripts/judge.py 52 torch --quick --use-starter
python scripts/judge.py 52 triton --quick --use-starter
python scripts/judge.py 52 cuda --quick --use-starter
```

Judge your own solution file:

```bash
python scripts/local_judge.py challenges/easy/52_silu --language torch --solution path/to/solution.py
python scripts/local_judge.py challenges/easy/52_silu --language triton --solution path/to/solution.triton.py
python scripts/local_judge.py challenges/easy/52_silu --language cuda --solution path/to/solution.cu
```

Run every challenge against starter files for all locally supported languages:

```bash
python scripts/local_judge.py --all --language all --use-starter --keep-going
```

Use `--overwrite` when you want a fresh results file instead of appending:

```bash
python scripts/judge.py 52 torch --quick --use-starter --overwrite
```

By default, results are appended to `local_judge_results.jsonl`, with code
metadata and snapshots stored under `.local_judge/code/`. Use `--overwrite` for a
fresh results file.

Some challenges are PyTorch-only. For example, `41_simple_inference` passes a
`torch.nn.Module` into `solve(...)`, so CUDA/Triton local judging is marked as
unsupported instead of inventing unusable CUDA/Triton starters.

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
