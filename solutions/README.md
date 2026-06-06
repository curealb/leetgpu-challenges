# Local Solutions

Keep personal solutions and experiments here instead of under `challenges/`.

The `challenges/` directory is intended to be synced from upstream as a clean
problem set. Keeping solutions outside that tree avoids conflicts when running:

```bash
python scripts/sync_challenges.py
```

Example layout:

```text
solutions/
└── 52_silu/
    ├── solution.cu
    ├── solution.triton.py
    └── solution.pytorch.py
```

Run one directly with:

```bash
python scripts/local_judge.py challenges/easy/52_silu --language cuda --solution solutions/52_silu/solution.cu
```
