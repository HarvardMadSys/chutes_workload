#!/usr/bin/env python3
"""Caching study, step 1: reconstructed sessions -> libCacheSim traces.

Each session is one cache object, and each of its requests is one access to
it, in started_at order. The oracleGeneral record of a request is

  obj_id             first 8 bytes of md5(session_id)
  obj_size           it, the request's input tokens (1 token counts as 1 byte).
                     The prompt grows turn by turn, so the object grows with it.
                     Not it + ot: the swept cache sizes are fractions of the
                     working set, the sum of each session's first-turn it.
  time               started_at, in epoch seconds
  next_access_vtime  1-based index of the session's next request, -1 if none

packed little-endian as <u4 time> <u8 obj_id> <u4 obj_size> <i8 next_access_vtime>.

Usage:
  python pipeline/caching/build_oracle_traces.py
  python pipeline/caching/build_oracle_traces.py --models minimax_m25
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chutes_sim import artifact

FRACTIONS = [float(f) for f in artifact.load_config("cachesim")["cache_size_fractions"].split(",")]
RECORD = np.dtype([("time", "<u4"), ("obj_id", "<u8"), ("obj_size", "<u4"),
                   ("next_access_vtime", "<i8")])


def md5_64(ids) -> np.ndarray:
    """Stable 64-bit ids: the first 8 bytes of each id's md5, little-endian."""
    return np.array([np.frombuffer(hashlib.md5(s.encode()).digest()[:8], "<u8")[0] for s in ids],
                    dtype=np.uint64)


def build_trace(model: str, out_dir: Path) -> None:
    requests = duckdb.sql(f"""
        SELECT epoch(started_at)::BIGINT AS ts, session_id, it::BIGINT AS it
        FROM '{artifact.sessions_parquet(model)}'
        ORDER BY started_at, session_id, turn_id
    """).df()
    n = len(requests)

    # The oracle field BeladySize needs. It is 1-based like libCacheSim's request
    # counter: with a 0-based index, back-to-back accesses score log(0) and the
    # evictor never returns.
    index = pd.Series(np.arange(1, n + 1, dtype=np.int64))
    next_vtime = index.groupby(requests["session_id"].values).shift(-1).fillna(-1).to_numpy(dtype=np.int64)

    codes, sessions = pd.factorize(requests["session_id"])
    obj_ids = md5_64(sessions)
    if len(np.unique(obj_ids)) != len(obj_ids):
        raise RuntimeError("64-bit session hash collision")

    record = np.empty(n, dtype=RECORD)
    record["time"] = requests["ts"].to_numpy(dtype=np.uint32)
    record["obj_id"] = obj_ids[codes]
    record["obj_size"] = requests["it"].to_numpy(dtype=np.uint32)
    record["next_access_vtime"] = next_vtime
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"{model}.oracleGeneral"
    record.tofile(trace)

    # libCacheSim resolves the fractional cache sizes against this working set.
    first_access = np.unique(codes, return_index=True)[1]
    working_set = int(requests["it"].to_numpy()[first_access].sum())
    print(f"{model}: {n:,} requests, {len(sessions):,} sessions, working set "
          f"{working_set:,} tokens -> {trace}\n"
          f"  cache sizes (KiB): {[int(f * working_set / 1024) for f in FRACTIONS]}")


def main() -> int:
    models = list(artifact.models())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=models, choices=models)
    ap.add_argument("--out-dir", type=Path, default=artifact.caching_dir() / "traces")
    args = ap.parse_args()
    for model in args.models:
        build_trace(model, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
