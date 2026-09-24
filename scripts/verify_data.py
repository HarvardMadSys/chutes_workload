#!/usr/bin/env python3
"""Check the downloaded trace against config/workload.json.

Checks the file size, the column list, the row count, and that the first
started_at is 1970-01-01 00:00:00 (timestamps count from the first request).
--sha256 also hashes the file, which takes a few minutes.

Usage:
    python scripts/verify_data.py [--trace PATH] [--sha256]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chutes_sim import artifact  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=artifact.trace_file())
    ap.add_argument("--sha256", action="store_true", help="also hash the file (minutes)")
    args = ap.parse_args()

    expected = artifact.workload_config()["trace"]
    if not args.trace.is_file():
        print(f"no trace at {args.trace}; fetch it with scripts/fetch_data.sh", file=sys.stderr)
        return 2

    con = duckdb.connect()
    trace = f"read_parquet('{args.trace}')"
    columns = [row[0] for row in con.sql(f"DESCRIBE SELECT * FROM {trace}").fetchall()]
    rows, first = con.sql(f"SELECT COUNT(*), MIN(started_at) FROM {trace}").fetchone()
    checks = [
        ("bytes", args.trace.stat().st_size, expected["bytes"]),
        ("columns", columns, expected["columns"]),
        ("rows", rows, expected["rows"]),
        ("first started_at", str(first), "1970-01-01 00:00:00"),
    ]
    if args.sha256:
        checks.append(("sha256", sha256(args.trace), expected["sha256"]))

    problems = 0
    print(f"trace: {args.trace}")
    for name, got, want in checks:
        ok = got == want
        problems += not ok
        print(f"  {'ok ' if ok else 'BAD'}  {name}: {got}" + ("" if ok else f"   expected {want}"))
    print(f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
