#!/usr/bin/env python3
"""Stage 2 (routing study) — run the simulation cells the figure needs.

Sweeps the figure's slice of the grid:

    2 models × 4 instance counts × 9 cache sizes × load_metric=total_tokens
    = 72 cells, each running all 4 policies  →  288 policy runs

Fixed settings (the published configuration): cache_mode = finite_lru,
session TTL = none, timeline sampling every 60 s of trace time, and
random_seed = 10, all from config/figure20.json.

One subprocess per cell (this script re-invokes itself with ``--worker``),
capped at ``--max-parallel``; each cell loads its sessions.parquet once
and runs the four policies sequentially. Output layout per cell:

    {model}_N{num_instances}_cache{tokens}_lm_{metric}/
      policy=<label>/{instance_metrics,policy_summary,instance_load_timeline}.parquet
      summary.csv / summary.md / experiment_config.json

Inputs are the ``sessions.parquet`` files of the published dataset,
resolved through ``chutes_sim.artifact`` ($CHUTES_DATA_ROOT, else
``output/<preprocess>/dataset``). Build them with ``make dataset``
or rebuild them with ``pipeline/preprocess/build_sessions.py``.

Each worker holds one sessions.parquet in memory as a list of Request
objects — budget roughly 6-8 GB of RAM per concurrent cell for v32 and
3-4 GB for minimax when choosing --max-parallel.

Usage:
    python pipeline/routing/run_sweep.py
    python pipeline/routing/run_sweep.py --output-root <dir> --max-parallel 24
    python pipeline/routing/run_sweep.py --models minimax \
        --num-instances 5 --cache-sizes 25000        # single-cell smoke test

Then:
    python pipeline/routing/build_data.py
    python pipeline/routing/plot_tradeoff.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chutes_sim import artifact
from chutes_sim.config import CacheMode, LoadMetric
from chutes_sim.experiment import ExperimentConfig, PolicySpec, run_experiment

REPO = artifact.artifact_root()
CFG = artifact.load_config("figure20")
GRID = CFG["simulation"]["grid"]
CATALOG = artifact.models()
DEFAULT_PARALLEL = max(1, min(24, (os.cpu_count() or 4) // 2))


def _sessions_parquet(model_key: str, sessions_root: Path | None = None) -> Path | None:
    """This model's sessions.parquet, or None if it is not present.

    ``sessions_root`` overrides the dataset root for one call; otherwise
    ``chutes_sim.artifact.data_root()`` decides ($CHUTES_DATA_ROOT, else
    ``output/<preprocess>/dataset``).
    """
    p = artifact.sessions_parquet(model_key, root=sessions_root)
    return p if p.exists() else None


def run_worker(
    model_key: str,
    cache_size: int,
    load_metric_value: str,
    num_instances: int,
    policy_names: list[str],
    output_root: Path,
    sessions_root: Path | None = None,
    random_seed: int = 10,
) -> int:
    """Inside-subprocess entry point: one (model, cache, load_metric, N) cell."""
    m = CATALOG[model_key]
    sessions_parquet = _sessions_parquet(model_key, sessions_root)
    if sessions_parquet is None:
        print(f"[{model_key}] missing sessions.parquet", file=sys.stderr, flush=True)
        return 2

    load_metric = LoadMetric(load_metric_value)
    name = f"{model_key}_N{num_instances}_cache{cache_size}_lm_{load_metric.value}"
    tag = f"{model_key}/N={num_instances}/cache={cache_size:,}/lm={load_metric.value}"
    print(f"[{tag}] starting policies={policy_names}", flush=True)
    t0 = time.time()

    cfg = ExperimentConfig(
        name=name,
        sessions_parquet_path=str(sessions_parquet),
        chute_id=m["chute_id"],
        model_name=m["model_name"],
        window_start=m["window_start"],
        window_end=m["window_end"],
        num_instances=num_instances,
        cache_mode=CacheMode.FINITE_LRU,
        cache_size_tokens_per_instance=cache_size,
        load_metric=load_metric,
        random_seed=random_seed,
        policies=[PolicySpec(label=n, name=n) for n in policy_names],
        preprocess_method=artifact.preprocess_name(),
        output_dir=output_root,
    )
    root = run_experiment(cfg)
    print(f"[{tag}] simulated in {time.time()-t0:.1f}s -> {root}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=artifact.routing_sweep_dir(),
                    help="directory for the sweep cells "
                         "(default: output/<preprocess>/<workload>/routing_sweep)")
    ap.add_argument("--max-parallel", type=int, default=DEFAULT_PARALLEL,
                    help=f"concurrent cell subprocesses (default {DEFAULT_PARALLEL} on this host; "
                         "each holds one sessions.parquet in RAM)")
    ap.add_argument("--random-seed", type=int, default=CFG["simulation"]["random_seed"],
                    help=f"random seed (default {CFG['simulation']['random_seed']})")
    ap.add_argument("--sessions-root", type=Path, default=None,
                    help="dataset root to load sessions.parquet from "
                         "(default: $CHUTES_DATA_ROOT or output/<preprocess>/dataset)")
    ap.add_argument("--models", nargs="+", default=GRID["models"], choices=GRID["models"])
    ap.add_argument("--num-instances", type=int, nargs="+", default=GRID["num_instances"])
    ap.add_argument("--cache-sizes", type=int, nargs="+", default=GRID["cache_size_tokens_per_instance"])
    ap.add_argument("--load-metrics", nargs="+", default=[CFG["figure_slice"]["load_metric"]],
                    choices=[m.value for m in LoadMetric])
    ap.add_argument("--policies", nargs="+", default=GRID["policies"], choices=GRID["policies"])
    ap.add_argument("--dry-run", action="store_true")
    # Internal: spawned children re-invoke this script with --worker.
    ap.add_argument("--worker", nargs=5,
                    metavar=("MODEL", "CACHE_SIZE", "LOAD_METRIC", "NUM_INSTANCES", "POLICIES"),
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Worker mode.
    if args.worker is not None:
        model_key, cache_s, lm_s, n_s, policies_s = args.worker
        return run_worker(
            model_key=model_key,
            cache_size=int(cache_s),
            load_metric_value=lm_s,
            num_instances=int(n_s),
            policy_names=[p for p in policies_s.split(",") if p],
            output_root=args.output_root,
            sessions_root=args.sessions_root,
            random_seed=args.random_seed,
        )

    # Fail early on a missing trace rather than 72 subprocesses deep.
    missing = [k for k in args.models if _sessions_parquet(k, args.sessions_root) is None]
    if missing:
        print(f"missing sessions.parquet for {missing} under {args.sessions_root or artifact.data_root()}\n"
              f"  build it:   make dataset\n"
              f"  or rebuild: python pipeline/preprocess/build_sessions.py",
              file=sys.stderr)
        return 2

    tasks = [
        (m, c, lm, n)
        for m in args.models
        for c in args.cache_sizes
        for lm in args.load_metrics
        for n in args.num_instances
    ]
    policies_csv = ",".join(args.policies)
    print(
        f"[main] {len(tasks)} cells ({len(args.models)} models × "
        f"{len(args.cache_sizes)} cache sizes × {len(args.load_metrics)} load metrics × "
        f"{len(args.num_instances)} N values); each cell runs "
        f"{len(args.policies)} policies ({', '.join(args.policies)})",
        flush=True,
    )
    if args.dry_run:
        for t in tasks:
            print(f"  would run: {t} policies={policies_csv}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    running: list[tuple[tuple, subprocess.Popen]] = []
    failures: list[tuple] = []
    t0 = time.time()

    def _spawn(task: tuple) -> subprocess.Popen:
        m, c, lm, n = task
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()),
             "--worker", m, str(c), lm, str(n), policies_csv,
             "--output-root", str(args.output_root),
             "--random-seed", str(args.random_seed)]
            + (["--sessions-root", str(args.sessions_root)] if args.sessions_root else []),
            cwd=REPO,
        )

    while pending or running:
        while pending and (args.max_parallel <= 0 or len(running) < args.max_parallel):
            task = pending.pop(0)
            running.append((task, _spawn(task)))
        time.sleep(1.0)
        still = []
        for task, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((task, proc))
            elif rc != 0:
                failures.append(task)
                print(f"[main] FAILED rc={rc}: {task}", flush=True)
        running = still

    print(f"[main] done in {time.time()-t0:.1f}s; {len(failures)} failures", flush=True)
    for f in failures:
        print(f"  failed: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
