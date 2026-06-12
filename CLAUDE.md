# LeetGPU Challenge Creation Guide

## Directory Structure

```
challenges/<difficulty>/<number>_<name>/
├── challenge.html        # Problem description
├── challenge.py          # Reference impl, test cases, metadata
└── starter/              # Local judging targets for supported languages
    ├── starter.cu         # Optional: CUDA support
    ├── starter.pytorch.py # Optional: PyTorch support
    └── starter.triton.py  # Optional: Triton support
```

- **Naming**: `<number>_<challenge_name>` — sequential integer, lowercase with underscores
- **Linting & contribution process**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Difficulty Levels

| Level | Parameters | Concepts | Examples |
|-------|-----------|----------|----------|
| Easy | 1-2 in + output | Single concept, basic parallelization | Vector add, transpose, element-wise ops |
| Medium | 2-4 in/out | Memory hierarchies, reductions, tiling | Tiled matmul, 2D convolution |
| Hard | Multiple with complex relationships | Warp ops, cooperative groups, heavy perf | GPU sorting, graph algorithms |

## challenge.py

Must inherit from `ChallengeBase` and follow Black formatting (line length 100).

**Reference files to read for patterns:**
- Base class: `challenges/core/challenge_base.py`
- Simple example: `challenges/easy/1_vector_add/challenge.py`
- Matrix example: `challenges/easy/3_matrix_transpose/challenge.py`
- Medium example: `challenges/medium/22_gemm/challenge.py`

### Required Methods

#### Class attributes (no `__init__`)
```python
class Challenge(ChallengeBase):
    name = "Challenge Display Name"  # Used to generate URLs — URL-friendly characters only (no parentheses or special symbols)
    atol = 1e-05           # Absolute tolerance (float32 default)
    rtol = 1e-05           # Relative tolerance (float32 default)
    num_gpus = 1
    access_tier = "free"   # "free" or "premium"
```

No `__init__` — `ChallengeBase` provides one that accepts `device` (default `"cuda"`). All tensor allocations in `generate_*_test` methods must use `device=self.device`, never hardcoded `device="cuda"`.

#### `reference_impl(self, ...)`
- Same parameters as user's `solve` function
- Must include assertions on shape and dtype (no device assertions — the runner targets multiple accelerators)
- Use PyTorch operations (not Python loops) for performance
- **Must work on both CUDA and XLA (TPU) devices.** Stick to standard PyTorch ops that have XLA lowerings. No CUDA-only kernels (`flash_attn`, `sdp_kernel`, manual cuDNN flag flips), no `.is_cuda` / `.device.type == "cuda"` checks, no `torch.cuda.*` API.

#### `get_solve_signature(self) -> Dict[str, tuple]`
Maps parameter names to `(ctype, direction)` tuples.

CUDA local judging requires every `ctype` to be `ctypes` compatible: tensor
pointers such as `ctypes.POINTER(ctypes.c_float)` or scalar ctypes such as
`ctypes.c_int` and `ctypes.c_size_t`. If any argument uses a Python object type
or another non-ctypes type, CUDA is marked as `unsupported`. Triton and PyTorch
starters are not gated by this CUDA compatibility check.

| ctypes | Use for |
|--------|---------|
| `ctypes.POINTER(ctypes.c_float)` | Tensor data |
| `ctypes.c_size_t` | Sizes/dimensions |
| `ctypes.c_int` | Integer parameters |

| Direction | Meaning |
|-----------|---------|
| `"in"` | Read-only input |
| `"out"` | Write-only output |
| `"inout"` | Read and write |

#### `generate_example_test(self) -> Dict[str, Any]`
One small, human-readable test case for display. Use literal tensor values.

#### `generate_functional_test(self) -> List[Dict[str, Any]]`
7-10 test cases with this coverage:

| Category | Sizes | Count |
|----------|-------|-------|
| Edge cases | 1, 2, 3, 4 | 2-3 |
| Power-of-2 | 16, 32, 64, 128, 256, 512, 1024 | 2-3 |
| Non-power-of-2 | 30, 100, 255 | 2-3 |
| Realistic | 1K-10K | 1-2 |

Must also include: zero inputs, negative numbers, mixed values.

#### `generate_performance_test(self) -> Dict[str, Any]`
One large test case. Size must fit 5x within 16GB (Tesla T4 VRAM).

| Operation type | Size |
|---------------|------|
| 1D | 10M-100M elements |
| 2D | 4K×4K to 8K×8K |
| Complex | 1M-10M |

## challenge.html

HTML fragment with four required sections:

1. **Problem description** — 2-3 sentences: what the function does, data types, constraints
2. **Implementation requirements** — Signature unchanged, no external libs, output location
3. **Examples** — 1-3 examples with Input/Output. The first example must match `generate_example_test()`. Format depends on data shape:
   - 1D data (vectors, sequences): use `<pre>` blocks
   - 2D/3D data (matrices, grids): use LaTeX `\begin{bmatrix}` inside `<p>` blocks
   - Be consistent within a single challenge
4. **Constraints** — Size bounds, data types, value ranges, **and performance test size**


**Formatting rules:**
- `<code>` for variables/functions; `<pre>` for 1D examples, LaTeX `\begin{bmatrix}` for matrices
- `&le;`, `&ge;`, `&times;` for math symbols
- **LaTeX underscores**: Inside `\text{}`, use plain `_` (not `\_`). The backslash-escaped form renders literally as `\_` in MathJax/KaTeX.
- **Performance test size bullet**: Must include a bullet documenting the exact parameters used in `generate_performance_test()`, formatted as:
  - `<li>Performance is measured with <code>param</code> = value</li>`
  - Use commas for numbers ≥ 1,000 (e.g., `25,000,000`)
  - Multiple parameters: `<code>M</code> = 8,192, <code>N</code> = 6,144, <code>K</code> = 4,096`

**Reference**: `challenges/easy/2_matrix_multiplication/challenge.html`

## Starter Code

Starter files declare which languages the local judge can run for a challenge.
Provide a starter only for each supported local language:

| Language | Starter file |
|----------|--------------|
| CUDA | `starter/starter.cu` |
| PyTorch | `starter/starter.pytorch.py` |
| Triton | `starter/starter.triton.py` |

If a selected language's starter file is missing, `scripts/local_judge.py`
records that run as `unsupported`. PyTorch starter files run normally when
present. Triton starter files run normally when present. CUDA starter files run
only when present and when `get_solve_signature()` is ctypes compatible.

Starters must compile/run without errors but not solve the problem. No comments
except the parameter description comment (e.g., `// A, B, C are device pointers`).

**Rules:**
- Provide all three starters only when the challenge is genuinely supported in
  CUDA, PyTorch, and Triton.
- Omit a starter for a language that cannot be represented usefully in that
  framework; the missing file is the unsupported marker.
- Easy problems: provide kernel scaffold with grid/block setup
- Medium/Hard problems: empty `solve` function only
- Match the exact style of existing starters in each framework

**Reference files** (read these for exact format):
- CUDA: `challenges/easy/1_vector_add/starter/starter.cu`
- PyTorch: `challenges/easy/1_vector_add/starter/starter.pytorch.py`
- Triton: `challenges/easy/1_vector_add/starter/starter.triton.py`

### Parameter Description Comment

Each starter file must have exactly one comment describing the parameters, placed directly before the `solve` function. Use these exact templates:

| Framework | Comment template |
|-----------|-----------------|
| CUDA | `// <params> are device pointers` |
| PyTorch, Triton | `# <params> are tensors on the GPU` |

**Rules:**
- Easy challenges: include the parenthetical `(i.e. pointers to memory on the GPU)` for CUDA (matches vector_add reference)
- Medium/Hard challenges: omit the parenthetical — just `are device pointers`
- No other comments anywhere in the starter file
- List only input/output tensor parameter names, not size parameters

## Creation Workflow

1. Create directory: `mkdir -p challenges/<difficulty>/<number>_<name>/starter`
2. Write `challenge.py` — inherit ChallengeBase, implement all 6 methods
3. Write `challenge.html` — all 4 sections
4. Write starter code for each supported local language; omit unsupported languages
5. Lint: `pre-commit run --all-files`

## Local Testing

This fork does not submit solutions to a remote service. Validate solutions with
the local judge only:

```bash
python scripts/local_judge.py 52 cuda --quick
python scripts/local_judge.py 52 triton --quick
python scripts/local_judge.py 52 torch --quick
```

## Checklist

Verify every item before relying on a local challenge.

### challenge.html
- [ ] Starts with `<p>` (problem description) — never `<h1>`
- [ ] Has `<h2>` sections for: Implementation Requirements, Example(s), Constraints (not `<h1>` or `<h3>`)
- [ ] First example matches `generate_example_test()` values
- [ ] Examples use `<pre>` for 1D data, LaTeX `\begin{bmatrix}` for matrices — consistent, never mixed
- [ ] Constraints includes `Performance is measured with <code>param</code> = value` bullet matching `generate_performance_test()`
- [ ] If the concept is spatial/structural, includes an SVG visualization after the problem description (dark theme, `#222` background)

### challenge.py
- [ ] `class Challenge` inherits `ChallengeBase`
- [ ] Five class attributes set: `name`, `atol`, `rtol`, `num_gpus`, `access_tier` (no `__init__`)
- [ ] `reference_impl` has assertions on shape and dtype only (no device-specific checks)
- [ ] `reference_impl` works on both CUDA and XLA — standard PyTorch ops only, no CUDA-only kernels or `torch.cuda.*` API
- [ ] All tensor allocations use `device=self.device`, never hardcoded `"cuda"`
- [ ] All 5 required methods present: `reference_impl`, `get_solve_signature`, `generate_example_test`, `generate_functional_test`, `generate_performance_test`
- [ ] `generate_functional_test` returns 7-10 cases: edge cases (1-4 elements), powers-of-2, non-powers-of-2, realistic sizes, zeros, negatives
- [ ] `generate_performance_test` fits 5x in 16GB VRAM (Tesla T4)

### Starter files
- [ ] Starter files are present for every supported language and omitted for unsupported languages
- [ ] Missing starter files intentionally mean local judge status `unsupported`
- [ ] CUDA starters are used only with ctypes-compatible `get_solve_signature()` entries
- [ ] Exactly 1 parameter description comment per starter file, no other comments
- [ ] CUDA uses "device pointers"; easy challenges include `(i.e. pointers to memory on the GPU)`, medium/hard omit it
- [ ] PyTorch and Triton use "tensors on the GPU"
- [ ] Starters compile/run but do NOT produce correct output

### General
- [ ] Directory follows `<number>_<name>` convention
- [ ] Linting passes: `pre-commit run --all-files`
