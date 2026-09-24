#!/usr/bin/env python3
"""Rebuild the tidy table behind Figure 20 from a raw sweep directory.

Reads every cell of a ``cache_loadmetric_sweep_*`` run:

  {model}_N{num_instances}_cache{tokens}_lm_{load_metric}/
    summary.csv                       one row per policy (global metrics)
    policy=<p>/instance_metrics.parquet   one row per instance

and emits one CSV row per (model, num_instances, cache_size, load_metric,
policy) with:

  * every column of ``summary.csv`` (token_hit_rate,
    effective_replication_ratio, max_mean_token_load, cv_token_load, ...)
  * ``load_imbalance_pct``   = (max_mean_token_load - 1) * 100  ← panel (d),
    the peak instance measured against the fleet average
  * ``max_min_token_load``   = (max(active_tokens_per_sec_mean) + 1)
                             / (min(active_tokens_per_sec_mean) + 1)
    plus the same ratio on request_count and input_tokens

The max/min columns are carried for reference only — nothing this artifact
plots uses them. Their +1 smoothing keeps a fully starved instance (min = 0,
common under ``sticky``) finite instead of NaN.

Usage:
    python pipeline/routing/build_data.py
    python pipeline/routing/build_data.py --sweep-dir <dir> --out <csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chutes_sim import artifact

CELL_KEYS = ["model", "num_instances", "cache_size", "load_metric", "policy_label"]


def parse_cell_name(name: str) -> tuple[str, int, int, str] | None:
    """``deepseek_v32_N20_cache25000_lm_total_tokens`` -> (deepseek_v32, 20, 25000, total_tokens)."""
    try:
        model, rest = name.split("_N", 1)
        n_s, rest = rest.split("_cache", 1)
        cache_s, load_metric = rest.split("_lm_", 1)
        return model, int(n_s), int(cache_s), load_metric
    except ValueError:
        return None


def collect(sweep_dir: Path) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    maxmin: list[dict] = []

    for cell in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        parsed = parse_cell_name(cell.name)
        if parsed is None:
            print(f"  skip (malformed name): {cell.name}")
            continue
        model, num_instances, cache_size, load_metric = parsed

        csv = cell / "summary.csv"
        if not csv.exists():
            print(f"  skip (no summary.csv): {cell.name}")
            continue
        df = pd.read_csv(csv)
        df["model"] = model
        df["num_instances"] = num_instances
        df["cache_size"] = cache_size
        df["load_metric"] = load_metric
        summaries.append(df)

        # Per-instance spread: only recoverable from instance_metrics.parquet.
        for pol_dir in sorted(cell.glob("policy=*")):
            inst_path = pol_dir / "instance_metrics.parquet"
            if not inst_path.exists():
                continue
            inst = pd.read_parquet(inst_path, engine="pyarrow")
            atps = inst["active_tokens_per_sec_mean"].astype(float)
            rc = inst["request_count"].astype(float)
            in_tok = inst["input_tokens"].astype(float)
            maxmin.append(
                {
                    "model": model,
                    "num_instances": num_instances,
                    "cache_size": cache_size,
                    "load_metric": load_metric,
                    "policy_label": pol_dir.name.split("=", 1)[1],
                    "max_min_token_load": float((atps.max() + 1.0) / (atps.min() + 1.0)),
                    "max_min_req_count": float((rc.max() + 1.0) / (rc.min() + 1.0)),
                    "max_min_input_tokens": float((in_tok.max() + 1.0) / (in_tok.min() + 1.0)),
                }
            )

    if not summaries:
        raise SystemExit(f"no usable cells under {sweep_dir}")

    out = pd.concat(summaries, ignore_index=True).merge(
        pd.DataFrame(maxmin), on=CELL_KEYS, how="left"
    )
    out["load_imbalance_pct"] = (out["max_mean_token_load"] - 1.0) * 100.0
    return out.sort_values(CELL_KEYS).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=artifact.routing_sweep_dir(),
                    help="sweep root written by pipeline/routing/run_sweep.py "
                         "(default: output/routing_sweep)")
    ap.add_argument("--out", type=Path,
                    default=artifact.artifact_root() / "data/routing/figure20_sweep.csv",
                    help="tidy CSV to write (default: data/routing/figure20_sweep.csv)")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        print(f"no sweep directory at {args.sweep_dir}\n"
              f"  run: python pipeline/routing/run_sweep.py", file=sys.stderr)
        return 2
    print(f"reading {args.sweep_dir}")
    df = collect(args.sweep_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    missing = int(df["max_min_token_load"].isna().sum())
    print(f"wrote {len(df):,} rows -> {args.out}")
    print(
        "  cells: "
        + " × ".join(
            f"{df[c].nunique()} {c}" for c in CELL_KEYS
        )
    )
    if missing:
        print(f"  WARNING: {missing} rows missing instance_metrics.parquet (panel d will have gaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
