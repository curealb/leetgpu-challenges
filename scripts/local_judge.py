#!/usr/bin/env python3
"""
Run LeetGPU challenges locally against Python/Triton or CUDA starter files.

Examples:
    python scripts/local_judge.py 1 cuda --quick
    python scripts/local_judge.py challenges/easy/1_vector_add --language cuda
    python scripts/local_judge.py --all --language triton
    python scripts/local_judge.py --all --language cuda --suites example,functional
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGES_ROOT = REPO_ROOT / "challenges"
DEFAULT_OUTPUT = REPO_ROOT / "local_judge_results.jsonl"
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / ".local_judge" / "code"
SUPPORTED_LANGUAGES = ("cuda", "triton", "pytorch")
LANGUAGE_ALIASES = {
    "cuda": "cuda",
    "triton": "triton",
    "trition": "triton",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "all": "all",
}
CONDA_ENV_NAME = "leetgpu"
CONDA_ENV_PYTHONS = (
    Path(f"/home/curealb/env/miniconda3/envs/{CONDA_ENV_NAME}/bin/python"),
    Path(f"/home/curealb/anaconda3/envs/{CONDA_ENV_NAME}/bin/python"),
    Path(f"/home/curealb/miniconda3/envs/{CONDA_ENV_NAME}/bin/python"),
)


@dataclass(frozen=True)
class CompiledSolution:
    kind: str
    solve: Callable[..., Any]
    cdll: ctypes.CDLL | None = None


class UnsupportedLanguageForChallenge(RuntimeError):
    pass


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = LANGUAGE_ALIASES.get(language.lower())
    if normalized is None:
        choices = ", ".join([*SUPPORTED_LANGUAGES, "torch", "all"])
        raise SystemExit(f"Unknown language {language!r}; choose one of: {choices}")
    return normalized


def can_import_torch(python: Path) -> bool:
    completed = subprocess.run(
        [str(python), "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def preferred_python() -> Path:
    current = Path(sys.executable)
    if can_import_torch(current):
        return current
    for candidate in CONDA_ENV_PYTHONS:
        if candidate.exists() and can_import_torch(candidate):
            return candidate
    return current


def maybe_reexec_with_preferred_python() -> None:
    current = Path(sys.executable).resolve()
    preferred = preferred_python().resolve()
    if preferred == current:
        return
    os.execv(str(preferred), [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]])


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required because challenge.py generates CUDA tensors. "
            "Install the repository GPU dependencies, then rerun this script."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false; local GPU judging needs CUDA.")
    return torch


def module_from_path(path: Path, name_prefix: str, extra_sys_path: Iterable[Path] = ()) -> ModuleType:
    for entry in reversed([str(p) for p in extra_sys_path]):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    module_name = f"{name_prefix}_{abs(hash(path.resolve()))}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_challenge(challenge_dir: Path) -> Any:
    challenge_py = challenge_dir / "challenge.py"
    if not challenge_py.exists():
        raise FileNotFoundError(f"No challenge.py found in {challenge_dir}")
    module = module_from_path(
        challenge_py,
        "leetgpu_challenge",
        extra_sys_path=[CHALLENGES_ROOT, challenge_dir],
    )
    if not hasattr(module, "Challenge"):
        raise RuntimeError(f"{challenge_py} does not define Challenge")
    return module.Challenge()


def default_starter_path(challenge_dir: Path, language: str) -> Path:
    if language == "cuda":
        filename = "starter.cu"
    elif language == "triton":
        filename = "starter.triton.py"
    elif language == "pytorch":
        filename = "starter.pytorch.py"
    else:
        raise ValueError(f"Unsupported language: {language}")

    return challenge_dir / "starter" / filename


def load_python_solution(path: Path, challenge_dir: Path) -> CompiledSolution:
    module = module_from_path(
        path,
        "leetgpu_solution",
        extra_sys_path=[path.parent, challenge_dir, challenge_dir / "starter"],
    )
    if not hasattr(module, "solve"):
        raise RuntimeError(f"{path} does not define solve(...)")
    return CompiledSolution(kind="python", solve=module.solve)


def compile_cuda_solution(path: Path, build_root: Path) -> CompiledSolution:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc was not found on PATH")

    build_root.mkdir(parents=True, exist_ok=True)
    lib_path = build_root / f"{path.parent.parent.name}_{path.parent.name}_{path.stem}.so"
    cmd = [
        nvcc,
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        str(path),
        "-o",
        str(lib_path),
    ]
    completed = subprocess.run(cmd, cwd=path.parent, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "CUDA compilation failed:\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    cdll = ctypes.CDLL(str(lib_path))
    solve = cdll.solve
    return CompiledSolution(kind="cuda", solve=solve, cdll=cdll)


def load_solution(path: Path, language: str, challenge_dir: Path, build_root: Path) -> CompiledSolution:
    if not path.exists():
        starter = default_starter_path(challenge_dir, language)
        if path == starter:
            raise FileNotFoundError(
                f"{path} does not exist. Expected the default {language} starter file."
            )
        raise FileNotFoundError(f"{path} does not exist")
    if language == "cuda":
        return compile_cuda_solution(path, build_root)
    if language in {"pytorch", "triton"}:
        return load_python_solution(path, challenge_dir)
    raise ValueError(f"Unsupported language: {language}")


def clone_value(value: Any) -> Any:
    torch = sys.modules.get("torch")
    if torch is not None and torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def clone_case(case: dict[str, Any]) -> dict[str, Any]:
    return {key: clone_value(value) for key, value in case.items()}


def get_case_list(challenge: Any, suite: str) -> list[dict[str, Any]]:
    if suite == "example":
        generated = challenge.generate_example_test()
    elif suite == "functional":
        generated = challenge.generate_functional_test()
    elif suite == "performance":
        generated = challenge.generate_performance_test()
    else:
        raise ValueError(f"Unknown suite: {suite}")

    if isinstance(generated, dict):
        return [generated]
    return list(generated)


def signature_order(challenge: Any) -> list[str]:
    return list(challenge.get_solve_signature().keys())


def output_names(challenge: Any) -> list[str]:
    signature = challenge.get_solve_signature()
    return [name for name, (_, direction) in signature.items() if direction in {"out", "inout"}]


def args_for_case(challenge: Any, case: dict[str, Any]) -> list[Any]:
    return [case[name] for name in signature_order(challenge)]


def ctypes_args_for_case(challenge: Any, case: dict[str, Any]) -> list[Any]:
    torch = sys.modules["torch"]
    converted: list[Any] = []
    for name, (ctype, _) in challenge.get_solve_signature().items():
        value = case[name]
        if torch.is_tensor(value):
            converted.append(ctypes.cast(value.data_ptr(), ctype))
        elif isinstance(ctype, type) and issubclass(ctype, ctypes._SimpleCData):
            converted.append(ctype(value))
        else:
            converted.append(value)
    return converted


def configure_cuda_signature(solution: CompiledSolution, challenge: Any) -> None:
    if solution.kind != "cuda":
        return
    unsupported = unsupported_cuda_signature_reason(challenge)
    if unsupported:
        raise UnsupportedLanguageForChallenge(unsupported)
    solution.solve.argtypes = [entry[0] for entry in challenge.get_solve_signature().values()]
    solution.solve.restype = None


def is_ctypes_arg_type(ctype: Any) -> bool:
    if not isinstance(ctype, type):
        return False
    try:
        return issubclass(ctype, ctypes._Pointer) or issubclass(ctype, ctypes._SimpleCData)
    except TypeError:
        return False


def unsupported_cuda_signature_reason(challenge: Any) -> str | None:
    unsupported = [
        name
        for name, (ctype, _) in challenge.get_solve_signature().items()
        if not is_ctypes_arg_type(ctype)
    ]
    if not unsupported:
        return None
    return (
        "CUDA local judging only supports ctypes-compatible solve signatures; "
        f"unsupported argument(s): {', '.join(unsupported)}"
    )


def run_solve(solution: CompiledSolution, challenge: Any, case: dict[str, Any]) -> None:
    if solution.kind == "cuda":
        solution.solve(*ctypes_args_for_case(challenge, case))
    else:
        solution.solve(*args_for_case(challenge, case))


def run_reference(challenge: Any, case: dict[str, Any]) -> None:
    challenge.reference_impl(*args_for_case(challenge, case))


def compare_values(actual: Any, expected: Any, atol: float, rtol: float) -> tuple[bool, dict[str, Any]]:
    torch = sys.modules["torch"]
    if torch.is_tensor(actual) and torch.is_tensor(expected):
        if actual.shape != expected.shape:
            return False, {"reason": "shape", "actual": list(actual.shape), "expected": list(expected.shape)}
        if actual.dtype != expected.dtype:
            return False, {"reason": "dtype", "actual": str(actual.dtype), "expected": str(expected.dtype)}

        if actual.dtype.is_floating_point or actual.dtype.is_complex:
            ok = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True))
            diff = (actual - expected).detach()
            max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
            return ok, {"max_abs": max_abs, "atol": atol, "rtol": rtol}

        ok = bool(torch.equal(actual, expected))
        mismatches = int((actual != expected).sum().item()) if actual.numel() else 0
        return ok, {"mismatches": mismatches}

    return actual == expected, {"actual": repr(actual), "expected": repr(expected)}


def compare_outputs(challenge: Any, actual_case: dict[str, Any], expected_case: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    all_ok = True
    for name in output_names(challenge):
        ok, info = compare_values(actual_case[name], expected_case[name], challenge.atol, challenge.rtol)
        details[name] = {"ok": ok, **info}
        all_ok = all_ok and ok
    return all_ok, details


def time_candidate(
    solution: CompiledSolution,
    challenge: Any,
    base_case: dict[str, Any],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    torch = sys.modules["torch"]
    samples: list[float] = []

    for _ in range(warmup):
        warm_case = clone_case(base_case)
        run_solve(solution, challenge, warm_case)
    torch.cuda.synchronize()

    for _ in range(repeats):
        timed_case = clone_case(base_case)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_solve(solution, challenge, timed_case)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    mean_ms = statistics.fmean(samples) if samples else math.nan
    return {
        "samples_ms": samples,
        "min_ms": min(samples) if samples else math.nan,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples) if samples else math.nan,
        "time_ms": mean_ms,
    }


def run_one_case(
    solution: CompiledSolution,
    challenge: Any,
    base_case: dict[str, Any],
    suite: str,
    case_index: int,
    warmup: int,
    repeats: int,
    timing: bool,
) -> dict[str, Any]:
    expected_case = clone_case(base_case)
    actual_case = clone_case(base_case)

    run_reference(challenge, expected_case)
    run_solve(solution, challenge, actual_case)
    sys.modules["torch"].cuda.synchronize()
    ok, details = compare_outputs(challenge, actual_case, expected_case)

    result: dict[str, Any] = {
        "suite": suite,
        "case_index": case_index,
        "ok": ok,
        "details": details,
    }
    if timing:
        result["timing"] = time_candidate(solution, challenge, base_case, warmup, repeats)
    return result


def find_challenge(number: str, difficulty: str | None) -> Path:
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


def discover_challenges(targets: list[str], run_all: bool, difficulty: str | None) -> list[Path]:
    if run_all:
        return sorted(path.parent for path in CHALLENGES_ROOT.glob("*/*/challenge.py"))

    challenge_dirs: list[Path] = []
    for target in targets:
        if target.isdigit():
            challenge_dirs.append(find_challenge(target, difficulty))
            continue
        path = Path(target).resolve()
        if path.is_file() and path.name == "challenge.py":
            challenge_dirs.append(path.parent)
        elif (path / "challenge.py").exists():
            challenge_dirs.append(path)
        else:
            raise FileNotFoundError(f"{target} is not a challenge directory")
    return challenge_dirs


def parse_suites(raw: str) -> list[str]:
    suites = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(suites) - {"example", "functional", "performance"})
    if unknown:
        raise SystemExit(f"Unknown suite(s): {', '.join(unknown)}")
    return suites


def languages_to_run(language: str | None) -> list[str]:
    language = normalize_language(language)
    language = language or "pytorch"
    if language == "all":
        return list(SUPPORTED_LANGUAGES)
    return [language]


def make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def read_solution_source(path: Path) -> tuple[str, bytes]:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace"), raw


def snapshot_solution(
    source_path: Path,
    snapshot_root: Path,
    run_id: str,
    challenge_rel: Path,
    language: str,
) -> dict[str, Any]:
    if not source_path.exists():
        return {
            "source_path": str(source_path),
            "code_snapshot": None,
            "code_sha256": None,
            "code_bytes": None,
            "code_lines": None,
        }

    source, raw = read_solution_source(source_path)
    challenge_key = challenge_rel.as_posix().replace("/", "__")
    snapshot_dir = snapshot_root / run_id / challenge_key / language
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / source_path.name
    snapshot_path.write_bytes(raw)

    return {
        "source_path": str(source_path),
        "code_snapshot": str(snapshot_path.relative_to(REPO_ROOT)),
        "code_sha256": hashlib.sha256(raw).hexdigest(),
        "code_bytes": len(raw),
        "code_lines": len(source.splitlines()),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def print_summary(rows: list[dict[str, Any]]) -> None:
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


def unsupported_row(
    run_id: str,
    run_started_at: str,
    rel: Path,
    challenge: Any,
    language: str,
    solution_path: Path,
    code_meta: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "challenge": str(rel),
        "challenge_name": getattr(challenge, "name", None),
        "language": language,
        "solution": str(solution_path),
        **code_meta,
        "ok": False,
        "skipped": True,
        "status": "unsupported",
        "error": reason,
    }


def split_positional_language(targets: list[str], language: str | None) -> tuple[list[str], str | None]:
    if language is not None or not targets:
        return targets, language
    maybe_language = LANGUAGE_ALIASES.get(targets[-1].lower())
    if maybe_language is None:
        return targets, language
    return targets[:-1], maybe_language


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local correctness and timing checks.",
        epilog=(
            "Examples: scripts/local_judge.py 1 cuda --quick | "
            "scripts/local_judge.py challenges/easy/1_vector_add --language cuda"
        ),
    )
    parser.add_argument("targets", nargs="*", help="Challenge numbers, directories, or challenge.py files")
    parser.add_argument("--all", action="store_true", help="Run every challenge under challenges/*/*")
    parser.add_argument("--language", choices=[*SUPPORTED_LANGUAGES, "torch", "trition", "all"])
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="Disambiguate numeric challenge targets",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only example,functional instead of the configured suites",
    )
    parser.add_argument("--suites", default="example,functional", help="Comma-separated: example,functional,performance")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-timing", action="store_true", help="Only check correctness")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output JSONL instead of appending a new run")
    parser.add_argument("--keep-going", action="store_true", help="Continue after per-case failures or exceptions")
    args = parser.parse_args()

    targets, language = split_positional_language(args.targets, args.language)
    maybe_reexec_with_preferred_python()
    import_torch()
    suites = parse_suites("example,functional" if args.quick else args.suites)
    challenge_dirs = discover_challenges(targets, args.all, args.difficulty)
    languages = languages_to_run(language)
    if not challenge_dirs:
        raise SystemExit("Pass at least one challenge directory, or use --all")

    rows: list[dict[str, Any]] = []
    failures = 0
    run_id = make_run_id()
    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_results = not args.overwrite

    def finish(status: int) -> int:
        write_jsonl(args.output, rows, append=append_results)
        mode = "appended" if append_results else "wrote"
        print(f"\n{mode.capitalize()} {len(rows)} result rows to {args.output}")
        print_summary(rows)
        return status

    with tempfile.TemporaryDirectory(prefix="leetgpu-local-judge-") as tmp:
        build_root = Path(tmp)
        for challenge_dir in challenge_dirs:
            rel = challenge_dir.relative_to(REPO_ROOT)
            for language in languages:
                solution_path = default_starter_path(challenge_dir, language)
                code_meta = snapshot_solution(
                    solution_path,
                    args.snapshot_root,
                    run_id,
                    rel,
                    language,
                )
                print(f"\n== {rel} [{language}] ==")

                try:
                    challenge = load_challenge(challenge_dir)
                    if language == "cuda":
                        reason = unsupported_cuda_signature_reason(challenge)
                        if reason is not None:
                            rows.append(
                                unsupported_row(
                                    run_id,
                                    run_started_at,
                                    rel,
                                    challenge,
                                    language,
                                    solution_path,
                                    code_meta,
                                    reason,
                                )
                            )
                            print(f"unsupported: {reason}")
                            continue
                    if (
                        language == "triton"
                        and not solution_path.exists()
                        and unsupported_cuda_signature_reason(challenge) is not None
                    ):
                        reason = (
                            "Triton local judging skipped because no Triton starter file "
                            "exists for this challenge."
                        )
                        rows.append(
                            unsupported_row(
                                run_id,
                                run_started_at,
                                rel,
                                challenge,
                                language,
                                solution_path,
                                code_meta,
                                reason,
                            )
                        )
                        print(f"unsupported: {reason}")
                        continue
                    solution = load_solution(solution_path, language, challenge_dir, build_root)
                    configure_cuda_signature(solution, challenge)
                except UnsupportedLanguageForChallenge as exc:
                    row = unsupported_row(
                        run_id,
                        run_started_at,
                        rel,
                        locals().get("challenge"),
                        language,
                        solution_path,
                        code_meta,
                        str(exc),
                    )
                    rows.append(row)
                    print(f"unsupported: {exc}")
                    continue
                except Exception as exc:
                    failures += 1
                    row = {
                        "run_id": run_id,
                        "run_started_at": run_started_at,
                        "challenge": str(rel),
                        "language": language,
                        "solution": str(solution_path),
                        **code_meta,
                        "ok": False,
                        "error": str(exc),
                    }
                    rows.append(row)
                    print(f"load failed: {exc}")
                    if not args.keep_going:
                        return finish(1)
                    continue

                challenge_ok = True
                for suite in suites:
                    try:
                        cases = get_case_list(challenge, suite)
                    except Exception as exc:
                        failures += 1
                        challenge_ok = False
                        rows.append(
                            {
                                "run_id": run_id,
                                "run_started_at": run_started_at,
                                "challenge": str(rel),
                                "challenge_name": getattr(challenge, "name", None),
                                "language": language,
                                "solution": str(solution_path),
                                **code_meta,
                                "suite": suite,
                                "ok": False,
                                "error": f"test generation failed: {exc}",
                            }
                        )
                        print(f"{suite}: generation failed: {exc}")
                        if not args.keep_going:
                            return finish(1)
                        continue

                    for idx, base_case in enumerate(cases):
                        try:
                            result = run_one_case(
                                solution,
                                challenge,
                                base_case,
                                suite,
                                idx,
                                args.warmup,
                                args.repeats,
                                timing=not args.no_timing,
                            )
                            row = {
                                "run_id": run_id,
                                "run_started_at": run_started_at,
                                "challenge": str(rel),
                                "challenge_name": challenge.name,
                                "language": language,
                                "solution": str(solution_path),
                                **code_meta,
                                **result,
                            }
                            rows.append(row)
                            if result["ok"]:
                                timing = result.get("timing", {})
                                suffix = (
                                    f" mean={timing.get('mean_ms', math.nan):.4f}ms"
                                    if timing
                                    else ""
                                )
                                print(f"{suite}[{idx}]: ok{suffix}")
                            else:
                                failures += 1
                                challenge_ok = False
                                print(f"{suite}[{idx}]: FAIL")
                                if not args.keep_going:
                                    return finish(1)
                        except Exception as exc:
                            failures += 1
                            challenge_ok = False
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "run_started_at": run_started_at,
                                    "challenge": str(rel),
                                    "challenge_name": getattr(challenge, "name", None),
                                    "language": language,
                                    "solution": str(solution_path),
                                    **code_meta,
                                    "suite": suite,
                                    "case_index": idx,
                                    "ok": False,
                                    "error": str(exc),
                                }
                            )
                            print(f"{suite}[{idx}]: ERROR {exc}")
                            if not args.keep_going:
                                return finish(1)

                print("summary:", "ok" if challenge_ok else "failed")

    return finish(0 if failures == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
