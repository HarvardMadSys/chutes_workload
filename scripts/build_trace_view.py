#!/usr/bin/env python3
"""Expose the anonymized trace as ``all_metrics_user`` for the analyses.

Every analysis queries a DuckDB connection for a table called
``all_metrics_user``. This writes a small DuckDB file whose only content is a
view over the trace parquet, joined to the user cohort table that ships in the
repository. It is the artifact's first step: run it once, and every pipeline
stage and figure script finds the trace through it.

Time
----
Timestamps in the trace are **elapsed time from the trace's first request**,
not calendar dates: the earliest ``started_at`` is exactly
``1970-01-01 00:00:00``. The view passes them through unchanged, and every
analysis labels its time axes relatively (day index, month index). Real dates
are not part of the release.

Usage:
    python scripts/build_trace_view.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chutes_sim import artifact  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=None,
                    help="the trace parquet (default: $CHUTES_TRACE_DIR/<file>, "
                         "else output/trace/chutes_trace.parquet)")
    ap.add_argument("--out", type=Path, default=None,
                    help="DuckDB view to write (default: output/trace_view.duckdb)")
    args = ap.parse_args()

    trace = args.trace or artifact.trace_file()
    out = args.out or Path(artifact.db_path())
    cohorts = artifact.cohorts_path()

    if not trace.is_file():
        raise SystemExit(f"no trace at {trace}\n"
                         f"fetch it with scripts/fetch_data.sh")
    if not cohorts.is_file():
        raise SystemExit(f"no cohort table at {cohorts} (it ships in the repository)")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    con = duckdb.connect(str(out))
    # release_cohort is the month index (1..13) in which the user first
    # appeared, computed before id rotation; a rotated id cannot derive it,
    # so it ships as a side table in the repository.
    con.execute(f"""
        CREATE VIEW all_metrics_user AS
        SELECT t.invocation_id, t.function_name, t.chute_id, t.user_id,
               t.rehash_round, t.instance_id, t.started_at, t.completed_at,
               t.it, t.ot, t.ct, t.ttft, c.release_cohort
        FROM read_parquet('{trace}') t
        LEFT JOIN read_parquet('{cohorts}') c
          ON c.user_id = t.user_id AND c.rehash_round = t.rehash_round
    """)
    n = con.sql("SELECT COUNT(*) FROM all_metrics_user").fetchone()[0]
    lo, hi = con.sql("SELECT MIN(started_at), MAX(started_at) FROM all_metrics_user").fetchone()
    con.close()

    expected = artifact.workload_config()["trace"]["rows"]
    print(f"wrote {out}")
    print(f"  trace     : {trace}")
    print(f"  rows      : {n:,}" + ("" if n == expected else f"  (expected {expected:,})"))
    print(f"  timestamps: elapsed from the trace start (origin 1970-01-01 00:00:00)")
    print(f"  span      : {lo} → {hi}  ({(hi - lo).days} days)")
    if n != expected:
        print("\nrow count does not match config/workload.json — is the trace complete?")
        return 1
    print(f"\nrun the analyses with:  --db {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
