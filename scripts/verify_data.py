#!/usr/bin/env python3
"""Verify fetched data against what this artifact expects.

The trace check reads the expected row count, byte size and column list from
``config/workload.json`` — the artifact ships its own reference, so the
download is one parquet with nothing alongside it. ``--sessions-only`` checks
the recovered sessions instead. This checks the *data*, not any analysis; run
it after fetching.

Usage:
    python scripts/verify_data.py [--trace PATH] [--quick] [--sha256]
    python scripts/verify_data.py --sessions-only
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chutes_sim import artifact  # noqa: E402


def _sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _verify_sessions() -> int:
    """Check each model's sessions.parquet against config/workload.json."""
    bad = 0
    for key in sorted(artifact.models()):
        m = artifact.model(key)
        p = artifact.sessions_parquet(key)
        if not p.exists():
            print(f"  MISSING  {key}: {p}")
            bad += 1
            continue
        n, s = duckdb.connect().sql(
            f"SELECT COUNT(*), COUNT(DISTINCT session_id) FROM read_parquet('{p}')"
        ).fetchone()
        ok = (n == m["requests"] and s == m["sessions"])
        print(f"  {'ok  ' if ok else 'BAD '} {key}: {n:,} requests, {s:,} sessions "
              f"(expected {m['requests']:,} / {m['sessions']:,})")
        bad += 0 if ok else 1
    print(f"\n{bad} model(s) mismatched")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=None, help="the trace parquet")
    ap.add_argument("--quick", action="store_true",
                    help="check presence and size only; skip the row count")
    ap.add_argument("--sha256", action="store_true",
                    help="also hash the file (minutes, and only useful if "
                         "config/workload.json records a digest)")
    ap.add_argument("--sessions-only", action="store_true",
                    help="check output/<preprocess>/dataset instead of the trace")
    args = ap.parse_args()

    if args.sessions_only:
        return _verify_sessions()

    src = artifact.workload_config()["source"]
    trace = args.trace or artifact.trace_file()
    if not trace.is_file():
        print(f"no trace at {trace}\nfetch it with scripts/fetch_data.sh trace",
              file=sys.stderr)
        return 2

    bad = 0
    origin = src["time_origin"]
    print(f"trace: {trace}")
    print(f"  elapsed time from {origin['origin']} · {origin['span_days']} days · "
          f"user_id rotation every {src['rotation_months']} months")

    size = trace.stat().st_size
    want_size = src.get("bytes")
    if want_size and size != want_size:
        print(f"  SIZE     {size:,} != {want_size:,}")
        bad += 1
    else:
        print(f"  size     {size:,} bytes"
              + ("" if want_size else "  (no reference to compare)"))

    con = duckdb.connect()
    cols = [r[0] for r in con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{trace}')").fetchall()]
    if cols != src["columns"]:
        print(f"  COLUMNS  {cols}\n           != {src['columns']}")
        bad += 1
    else:
        print(f"  columns  {len(cols)} as expected")

    lo = con.sql(f"SELECT MIN(started_at) FROM read_parquet('{trace}')").fetchone()[0]
    if str(lo) != origin["origin"].replace("T", " ").rstrip("0").rstrip("."):
        if str(lo) != "1970-01-01 00:00:00":
            print(f"  ORIGIN   first started_at is {lo}, expected {origin['origin']}")
            bad += 1
        else:
            print(f"  origin   {lo} (zero)")
    else:
        print(f"  origin   {lo} (zero)")

    if not args.quick:
        n = con.sql(f"SELECT COUNT(*) FROM read_parquet('{trace}')").fetchone()[0]
        if n != src["rows"]:
            print(f"  ROWS     {n:,} != {src['rows']:,}")
            bad += 1
        else:
            print(f"  rows     {n:,}")

    want_hash = src.get("sha256")
    if args.sha256:
        got = _sha256(trace)
        if want_hash and got != want_hash:
            print(f"  SHA256   {got}\n           != {want_hash}")
            bad += 1
        else:
            print(f"  sha256   {got}" + ("" if want_hash else "  (nothing to compare)"))

    cohorts = artifact.cohorts_path()
    if cohorts.is_file():
        n, k = con.sql(f"SELECT COUNT(*), COUNT(DISTINCT release_cohort) "
                       f"FROM read_parquet('{cohorts}')").fetchone()
        print(f"cohorts: {n:,} (user_id, round) rows, {k} cohorts  [{cohorts.name}, in-repo]")
    else:
        print(f"cohorts: MISSING at {cohorts} — the §5 cohort figures need it")
        bad += 1

    print(f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
