#!/usr/bin/env python3
"""Fast end-to-end check of both studies against the published numbers.

Runs one routing cell (minimax, N=5, cache=25,000, all four policies) and
compares every value in its ``summary.csv`` against the corresponding row of
the committed ``data/routing/figure20_sweep.csv``. Then, if a cachesim binary
has been built, replays one algorithm on one workload and compares the miss
ratios against ``data/caching/results.txt``.

Needs ``sessions.parquet`` (``make dataset``). Takes a few minutes: the
routing cell replays all 4.2 M minimax requests four times.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-caching
    python scripts/smoke_test.py --caching-algo lru --caching-workload minimax_m25
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from chutes_sim import artifact

# libCacheSim result lines, e.g.
#   minimax_m25.oracleGeneral LRU cache size 61KiB, 4223356 req, miss ratio ...
_LINE_RE = re.compile(
    r"([\w.-]+)\.oracleGeneral\s+(\S+)\s+cache size\s+(\d+)KiB,\s+\d+ req,"
    r"\s+miss ratio ([\d.]+), byte miss ratio ([\d.]+)")

_ALGOS = ["FIFO", "LRU", "ARC", "LRB", "S3FIFO", "GDSF", "LHD", "Sieve", "BeladySize"]


def _canon_algo(raw: str) -> str:
    """LRB-BMR -> LRB, S3FIFO-0.1000-1 -> S3FIFO."""
    for a in sorted(_ALGOS, key=len, reverse=True):
        if raw.upper().startswith(a.upper()):
            return a
    return raw


def _parse_results(path: Path) -> dict:
    """-> {(workload, algo): {size_kib: (miss_ratio, byte_miss_ratio)}}"""
    out: dict = {}
    for wl, algo, size, mr, bmr in _LINE_RE.findall(path.read_text()):
        out.setdefault((wl, _canon_algo(algo)), {})[int(size)] = (float(mr), float(bmr))
    return out


REPO = artifact.artifact_root()
TOL = 1e-9

# Wall-clock bookkeeping, not a result: excluded from the comparison.
EXCLUDED_COLUMNS = {"elapsed_sec"}

# The cell used for the routing check, and its coordinates in the shipped CSV.
CELL = dict(model="minimax", num_instances=5, cache_size=25000, load_metric="total_tokens")


def _run_cell(out_root: Path) -> bool:
    cmd = [sys.executable, str(REPO / "pipeline/routing/run_sweep.py"),
           "--output-root", str(out_root),
           "--models", CELL["model"],
           "--num-instances", str(CELL["num_instances"]),
           "--cache-sizes", str(CELL["cache_size"]),
           "--max-parallel", "1"]
    print(f"routing: {' '.join(cmd[1:])}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        print("FAIL routing: sweep returned non-zero")
        return False
    return True


def _routing(keep: bool, reuse: Path | None = None) -> bool:
    parquet = artifact.sessions_parquet(CELL["model"])
    if not parquet.exists():
        print(f"SKIP routing: no dataset at {parquet}\n"
              f"     run `make dataset` first")
        return True

    published = pd.read_csv(REPO / "data/routing/figure20_sweep.csv")
    want = published[
        (published["model"] == CELL["model"])
        & (published["num_instances"] == CELL["num_instances"])
        & (published["cache_size"] == CELL["cache_size"])
        & (published["load_metric"] == CELL["load_metric"])
    ]
    if want.empty:
        print("SKIP routing: cell not present in the shipped CSV")
        return True

    if reuse is not None:
        tmp = reuse
    else:
        tmp = Path(tempfile.mkdtemp(prefix="chutes-smoke-"))
        if not _run_cell(tmp):
            return False

    cell_dir = tmp / (f"{CELL['model']}_N{CELL['num_instances']}"
                      f"_cache{CELL['cache_size']}_lm_{CELL['load_metric']}")
    if not (cell_dir / "summary.csv").exists():
        print(f"FAIL routing: no summary.csv under {cell_dir}")
        return False
    got = pd.read_csv(cell_dir / "summary.csv")

    shared = [c for c in got.columns
              if c in want.columns and c != "policy_label" and c not in EXCLUDED_COLUMNS]
    ok = True
    checked = 0
    for _, g in got.iterrows():
        w = want[want["policy_label"] == g["policy_label"]]
        if w.empty:
            print(f"  ? no published row for policy {g['policy_label']}")
            ok = False
            continue
        w = w.iloc[0]
        for c in shared:
            a, b = g[c], w[c]
            if isinstance(a, float) or isinstance(b, float):
                try:
                    same = abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(b)))
                except (TypeError, ValueError):
                    same = a == b
            else:
                same = a == b
            checked += 1
            if not same:
                print(f"  MISMATCH {g['policy_label']}.{c}: {a!r} != published {b!r}")
                ok = False
    print(f"{'PASS' if ok else 'FAIL'} routing: {checked} values compared across "
          f"{len(got)} policies vs the published cell")
    if keep:
        print(f"  (cell kept at {cell_dir})")
    return ok


def _caching(algo: str, workload: str) -> bool:
    build = artifact.cache_eviction_dir() / "build/src/libCacheSim/_build/bin/cachesim"
    trace = artifact.cache_eviction_dir() / "traces" / f"{workload}.oracleGeneral"
    if not build.exists() or not trace.exists():
        print(f"SKIP caching: need {build.name} and {trace.name}\n"
              f"     run `make caching` (or bash pipeline/caching/run_cachesim.sh)")
        return True

    cfg = artifact.load_config("cachesim")
    fracs = cfg["sweep"]["cache_size_fractions_arg"]
    deps = artifact.cache_eviction_dir() / "build/deps/lib"
    import os
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{deps}:{build.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    extra = ["-e", "move-to-main-threshold=1"] if algo == "s3fifo" else []
    print(f"caching: cachesim {trace.name} {algo}", flush=True)
    proc = subprocess.run([str(build), trace.name, "oracleGeneral", algo, fracs, *extra],
                          cwd=trace.parent, capture_output=True, text=True, env=env)
    lines = [ln for ln in proc.stdout.splitlines() if "cache size" in ln and "hour" not in ln]
    if not lines:
        print("FAIL caching: cachesim produced no result lines")
        return False

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        got_path = f.name
    got = _parse_results(Path(got_path))
    orig = _parse_results(REPO / "data/caching/results.txt")

    ok, n = True, 0
    for (wl, alg), curve in got.items():
        ref = orig.get((wl, alg))
        if ref is None:
            print(f"  ? no published curve for {wl}/{alg}")
            continue
        for size, (miss, byte_miss) in curve.items():
            if size not in ref:
                continue
            n += 1
            dm = abs(miss - ref[size][0])
            db = abs(byte_miss - ref[size][1])
            # cachesim prints 4 decimals, so a 1-ulp difference in the last
            # printed digit (1e-4) is a rounding artifact, not a divergence.
            if max(dm, db) > 1.5e-4:
                print(f"  MISMATCH {wl}/{alg}@{size}KiB: "
                      f"miss {miss:.4f} vs {ref[size][0]:.4f}, "
                      f"byte {byte_miss:.4f} vs {ref[size][1]:.4f}")
                ok = False
    print(f"{'PASS' if ok else 'FAIL'} caching: {n} cache sizes compared "
          f"for {algo} vs data/caching/results.txt")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-routing", action="store_true")
    ap.add_argument("--skip-caching", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temporary sweep cell")
    ap.add_argument("--reuse", type=Path, default=None,
                    help="compare an already-simulated sweep root instead of re-running")
    ap.add_argument("--caching-algo", default="lru")
    ap.add_argument("--caching-workload", default="minimax_m25")
    args = ap.parse_args()

    results = []
    if not args.skip_routing:
        results.append(_routing(args.keep, args.reuse))
    if not args.skip_caching:
        results.append(_caching(args.caching_algo, args.caching_workload))

    ok = all(results)
    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
