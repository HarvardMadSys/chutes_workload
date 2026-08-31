#!/usr/bin/env python3
"""Check what this machine can run.

Reports three groups, and what each is missing:

  replot      python + matplotlib/pandas, and the committed data files
  rerun       the anonymized trace, plus cmake/git/a compiler for the
              caching study's libCacheSim build
  paper/      seaborn + fastparquet, and the trace

Usage:
    python scripts/verify_env.py [--data-root DIR] [--db-path PATH]
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chutes_sim import artifact

REQUIRED = [("pandas", "2.0"), ("pyarrow", "14.0"), ("numpy", "1.24"), ("matplotlib", "3.7")]
PINNED = {"duckdb": "1.5.2", "pandas": "3.0.3", "pyarrow": "24.0.0",
          "numpy": "2.4.4", "matplotlib": "3.10.9"}

OK, WARN, BAD = "  ok  ", " warn ", " MISS "


def _row(status: str, what: str, detail: str = "") -> None:
    print(f"[{status}] {what}{('  — ' + detail) if detail else ''}")


def check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    _row(OK if ok else BAD, f"python {v.major}.{v.minor}.{v.micro}",
         "published results used 3.12.3" if ok else "need >= 3.10")
    return ok


def check_packages(names: list[str]) -> bool:
    all_ok = True
    for name in names:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            _row(BAD, name, "pip install -r requirements.txt")
            all_ok = False
            continue
        got = getattr(mod, "__version__", "?")
        pin = PINNED.get(name)
        note = "" if (pin is None or got == pin) else f"published used {pin}"
        _row(OK if not note else WARN, f"{name} {got}", note)
    return all_ok


def check_tools(tools: list[str]) -> bool:
    all_ok = True
    for t in tools:
        p = shutil.which(t)
        _row(OK if p else BAD, t, p or "not on PATH")
        all_ok &= bool(p)
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=artifact.data_root())
    ap.add_argument("--db-path", default=artifact.db_path())
    args = ap.parse_args()

    root = artifact.artifact_root()
    print(f"artifact root : {root}")
    print(f"dataset root  : {args.data_root}")
    print(f"output root   : {artifact.output_root()}\n")

    print("── replot from committed data ──")
    t0 = check_python()
    t0 &= check_packages(["pandas", "pyarrow", "numpy", "matplotlib"])
    for rel in ["data/routing/figure20_sweep.csv",
                "data/caching/results.txt",
                "config/workload.json", "config/figure20.json", "config/cachesim.json"]:
        p = root / rel
        _row(OK if p.exists() else BAD, rel,
             f"{p.stat().st_size:,} B" if p.exists() else "missing")
        t0 &= p.exists()

    print("\n── rerun from the anonymized trace ──")
    present = 0
    for key in artifact.models():
        m = artifact.model(key)
        p = artifact.sessions_parquet(key, root=args.data_root)
        if p.exists():
            _row(OK, f"{key} sessions.parquet", f"{p.stat().st_size:,} B")
            present += 1
        else:
            _row(WARN, f"{key} sessions.parquet", "rebuild with: make dataset")
    print("  (caching study also needs a C/C++ toolchain for libCacheSim)")
    t1_build = check_tools(["git", "cmake", "make"]) and bool(
        shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"))
    _row(OK if t1_build else WARN, "C/C++ toolchain",
         "" if t1_build else "needed only for `make caching`")

    has_duckdb = check_packages(["duckdb"])
    trace = artifact.trace_file()
    have_trace = trace.is_file()
    _row(OK if have_trace else BAD, f"trace {trace.name}",
         f"{trace.stat().st_size:,} B" if have_trace
         else "absent — scripts/fetch_data.sh trace")
    coh = artifact.cohorts_path()
    _row(OK if coh.is_file() else BAD, f"cohorts {coh.name}",
         "ships in the repository" if coh.is_file() else f"missing at {coh}")
    view = Path(artifact.db_path())
    _row(OK if view.exists() else WARN, f"view {view.name}",
         "built" if view.exists() else "run: make trace")

    print("\n── paper/ — the paper's trace-analysis figures ──")
    has_paper_deps = check_packages(["seaborn", "fastparquet"])
    seeds = root / "data/paper/cache"
    n_seeds = len(list(seeds.glob("*.parquet"))) if seeds.is_dir() else 0
    _row(OK if n_seeds else WARN, f"cache seeds ({n_seeds} parquet)",
         "shorten a rerun" if n_seeds else "every query will run from scratch")
    _row(OK if (root / "data/paper/chute_models.csv").exists() else BAD,
         "data/paper/chute_models.csv", "chute_id -> model name")
    _row(OK if (root / "data/routing/figure20_sweep.csv").exists() else BAD,
         "05_load_balancing --part b input", "data/routing/figure20_sweep.csv")

    print("\nsummary:")
    print(f"  replot   : {'ready' if t0 else 'BLOCKED'}")
    print(f"  rerun    : {'ready' if (have_trace and has_duckdb) else 'needs the trace (scripts/fetch_data.sh trace)'}")
    if has_paper_deps and have_trace:
        paper_state = "ready (all sections)"
    elif has_paper_deps:
        paper_state = "needs the trace (scripts/fetch_data.sh trace)"
    else:
        paper_state = "BLOCKED — pip install seaborn fastparquet"
    print(f"  paper figures        : {paper_state}")
    return 0 if t0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
