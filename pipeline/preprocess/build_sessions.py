#!/usr/bin/env python3
"""Stage 1 — build the ``7day_naive`` dataset from the anonymized trace.

For each model, queries the trace for its 7-day window, runs the
``naive_chain`` session reconstruction (lookback 10), and persists under
``<output-dir>/model=<chute>/<window_dir>/`` (window_dir from the config):

  sessions.parquet     the enriched trace (one row per loaded request)
  chain_summary.json   chain-length distribution + structural-reuse stats
  chain_samples.json   sample chains per chain-length bucket, as nested
                       trees — the per-turn-category sample log required
                       by the project's preprocessing convention

Windows (the busiest 7 days per model — from config/workload.json):

  v32      deepseek-ai/DeepSeek-V3.2-TEE   trace day 302 → 309
  minimax  MiniMaxAI/MiniMax-M2.5-TEE      trace day 346 → 353

Reads the DuckDB view over the anonymized trace (``make trace``); this is
the only stage that queries it. Both simulation studies then run entirely
from the ``sessions.parquet`` files it writes.

Usage:
    python pipeline/preprocess/build_sessions.py
    python pipeline/preprocess/build_sessions.py --models minimax
    python pipeline/preprocess/build_sessions.py --db-path output/trace_view.duckdb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from chutes_sim import artifact
from chutes_sim.preprocess import (
    NaiveChainParams,
    build_naive_chains,
    load_slice,
    summarize_naive_chains,
)

_WORKLOAD = artifact.workload_config()
TABLE = _WORKLOAD["source"]["table"]
NAIVE_LOOKBACK = _WORKLOAD["preprocess_spec"]["params"]["lookback"]


def sample_chains(enriched: pd.DataFrame, *, per_bucket: int = 5) -> dict[str, list]:
    """Per chain-length bucket: a few sample chains as nested trees.

    Each chain is rendered rooted at its turn-0 head, with every node
    carrying turn_id, parent, timing gap, token sizes, observed ct, and
    the structural prefix the builder assigned. Also samples turn-0 rows
    with observed ct > 0 — the cases where production saw a cache hit but
    the builder could not link a parent.
    """
    chat = enriched[enriched["link_confidence"] != "non_generation"].copy()
    if chat.empty:
        return {}

    chat["started_at"] = pd.to_datetime(chat["started_at"])
    started_lookup: dict[str, pd.Timestamp] = dict(
        zip(chat["invocation_id"].astype(str), chat["started_at"])
    )
    lengths = chat.groupby("session_id").size().rename("chain_length")
    chat = chat.merge(lengths, left_on="session_id", right_index=True)
    multi = chat[chat["chain_length"] >= 2].copy()

    def node(r) -> dict:
        parent_id = r.parent_invocation_id
        if pd.notna(parent_id) and str(parent_id) in started_lookup:
            gap = (r.started_at - started_lookup[str(parent_id)]).total_seconds()
        else:
            gap = None
        return {
            "turn_id": int(r.turn_id),
            "invocation_id": str(r.invocation_id),
            "parent_invocation_id": str(parent_id) if pd.notna(parent_id) else None,
            "instance_id": str(r.instance_id),
            "function_name": str(r.function_name),
            "started_at": str(r.started_at),
            "gap_from_parent_sec": float(gap) if gap is not None else None,
            "input_tokens": int(r.it),
            "output_tokens": int(r.ot),
            "cached_tokens": int(r.ct),
            "structural_prefix_tokens": int(r.structural_prefix_tokens),
            "link_confidence": str(r.link_confidence),
            "children": [],
        }

    out: dict[str, list] = {}
    for length, g in multi.groupby("chain_length"):
        sids = g["session_id"].unique().tolist()[:per_bucket]
        bucket = []
        for sid in sids:
            grp = g[g["session_id"] == sid].sort_values("started_at")
            nodes: dict[str, dict] = {}
            roots: list[dict] = []
            for r in grp.itertuples(index=False):
                nd = node(r)
                nodes[nd["invocation_id"]] = nd
                if nd["parent_invocation_id"] is None:
                    roots.append(nd)
            for nd in nodes.values():
                pid = nd["parent_invocation_id"]
                if pid is not None and pid in nodes:
                    nodes[pid]["children"].append(nd)
                elif pid is not None and pid not in nodes:
                    roots.append(nd)
            tids = grp["turn_id"].astype(int)
            bucket.append({
                "session_id": str(sid),
                "user_id": str(grp["user_id"].iloc[0]),
                "chain_length": int(length),
                "max_turn_id": int(tids.max()),
                "branching_factor": float(length) / (int(tids.max()) + 1),
                "distinct_instances": int(grp["instance_id"].nunique()),
                "tree": roots[0] if len(roots) == 1 else {"_multi_root": True, "roots": roots},
            })
        out[str(int(length))] = bucket

    # Also sample turn_0 rows with ct > 0 — under-covered cases where
    # production thought there was a cache hit but no parent was linked.
    turn_0 = chat[(chat["link_confidence"] == "turn_0") & (chat["ct"] > 0)]
    if not turn_0.empty:
        out["turn_0_with_ct_gt_0"] = [
            {
                "invocation_id": str(r.invocation_id),
                "user_id": str(r.user_id),
                "instance_id": str(r.instance_id),
                "function_name": str(r.function_name),
                "started_at": str(r.started_at),
                "input_tokens": int(r.it),
                "output_tokens": int(r.ot),
                "cached_tokens": int(r.ct),
            }
            for r in turn_0.head(per_bucket * 2).itertuples(index=False)
        ]

    return out


def run_one_model(
    *, db_path: str, chute_id: str, model_name: str,
    start: datetime, end: datetime, window_dir: str,
    out_root: Path, samples_per_bucket: int,
) -> Path:
    # window_dir comes from config/workload.json so the layout has one source
    # of truth; artifact.sessions_parquet() resolves the same path.
    out_dir = out_root / f"model={chute_id}" / window_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions_path = out_dir / "sessions.parquet"
    if sessions_path.exists():
        print(f"  (reusing existing {sessions_path})", file=sys.stderr)
        enriched = pd.read_parquet(sessions_path)
    else:
        t0 = time.time()
        print(f"  loading {model_name} {start} → {end} ...", file=sys.stderr, flush=True)
        df = load_slice(
            db_path=db_path, table_name=TABLE, chute_id=chute_id,
            start_time=start, duration=end - start,
        )
        print(f"  loaded {len(df):,} rows in {time.time() - t0:.1f}s", file=sys.stderr)

        t0 = time.time()
        enriched = build_naive_chains(df, params=NaiveChainParams(lookback=NAIVE_LOOKBACK))
        print(f"  built chains in {time.time() - t0:.1f}s", file=sys.stderr)
        enriched.to_parquet(sessions_path, index=False)
        print(f"  wrote {sessions_path}", file=sys.stderr)

    # Standard chain summary.
    summary = summarize_naive_chains(enriched)
    with open(out_dir / "chain_summary.json", "w") as f:
        json.dump({
            "model_name": model_name, "chute_id": chute_id,
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "mode": "naive_chain",
            "params_used": {"lookback": NAIVE_LOOKBACK},
            **summary,
        }, f, indent=2, default=str, sort_keys=True)

    # Stratified chain samples (nested trees) — one bucket per chain length.
    t0 = time.time()
    samples = sample_chains(enriched, per_bucket=samples_per_bucket)
    with open(out_dir / "chain_samples.json", "w") as f:
        json.dump({
            "model_name": model_name, "chute_id": chute_id,
            "samples_per_bucket": samples_per_bucket,
            "note": "Nested trees rooted at turn_0; one bucket per chain length, "
                    "plus a 'turn_0_with_ct_gt_0' bucket for inspection.",
            "samples_by_length": samples,
        }, f, indent=2, default=str)
    print(f"  chain_samples in {time.time() - t0:.1f}s", file=sys.stderr)

    return out_dir


def main() -> int:
    catalog = artifact.models()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(catalog), choices=list(catalog))
    ap.add_argument("--db-path", default=artifact.db_path(),
                    help="DuckDB trace path (default: $CHUTES_DB_PATH or config/workload.json)")
    ap.add_argument("--output-dir", type=Path, default=artifact.data_root(),
                    help="dataset root (default: $CHUTES_DATA_ROOT or output/<preprocess>/dataset)")
    ap.add_argument("--samples-per-bucket", type=int, default=2)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    for key in args.models:
        m = catalog[key]
        print(f"\n========== {m['model_name']} ({key}) ==========", file=sys.stderr)
        run_one_model(
            db_path=args.db_path,
            chute_id=m["chute_id"], model_name=m["model_name"],
            start=datetime.fromisoformat(m["window_start"]),
            end=datetime.fromisoformat(m["window_end"]),
            window_dir=m["window_dir"],
            out_root=args.output_dir, samples_per_bucket=args.samples_per_bucket,
        )

    print(f"\nAll 7day_naive datasets done in {time.time() - overall_t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
