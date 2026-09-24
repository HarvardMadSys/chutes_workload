#!/usr/bin/env python3
"""Reconstruct the sessions of the two models the simulation studies replay.

For each model in config/workload.json, reads its busiest 7-day window from the
trace view (make trace), reconstructs sessions with chutes_sim/sessions.py (the
rule is documented there) and writes

    output/sessions/<model>.parquet

The routing study (pipeline/routing) and the caching study (pipeline/caching)
read only these files. About 3 minutes for both models.

Usage:
    python pipeline/sessions/build_sessions.py
    python pipeline/sessions/build_sessions.py --models minimax_m25
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chutes_sim import artifact
from chutes_sim.sessions import load_trace_window, reconstruct_sessions


def main() -> int:
    config = artifact.workload_config()
    models = config["models"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(models), choices=list(models))
    ap.add_argument("--db", default=artifact.db_path(),
                    help="DuckDB trace view (default: %(default)s)")
    args = ap.parse_args()

    mismatches = 0
    for key in args.models:
        m = models[key]
        print(f"{key}: {m['model_name']}, {m['window_start']} -> {m['window_end']}", flush=True)
        requests = load_trace_window(args.db, config["trace"]["table"], m["chute_id"],
                                     datetime.fromisoformat(m["window_start"]),
                                     datetime.fromisoformat(m["window_end"]))
        sessions = reconstruct_sessions(requests)

        out = artifact.sessions_parquet(key)
        out.parent.mkdir(parents=True, exist_ok=True)
        sessions.to_parquet(out, index=False)

        n_requests, n_sessions = len(sessions), sessions["session_id"].nunique()
        matches = (n_requests, n_sessions) == (m["requests"], m["sessions"])
        mismatches += not matches
        print(f"  {n_requests:,} requests, {n_sessions:,} sessions -> {out}"
              + ("" if matches else f"   MISMATCH: published {m['requests']:,} / {m['sessions']:,}"))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
