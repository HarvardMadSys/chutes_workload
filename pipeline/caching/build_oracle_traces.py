#!/usr/bin/env python3
"""Stage 2 (caching study) — sessions.parquet -> libCacheSim `oracleGeneral` traces.

This is the preprocessing behind data/caching/results.txt.
Spec (the authoritative copy lives in config/cachesim.json):

  - one cache request per row (= one chained conversation turn), replayed in
    started_at order; a whole sequence/session is the admission+eviction unit
    (most sequences get zero or full hits), so:
  - obj_id   = 64-bit hash of session_id
  - obj_size = it  (input tokens of that turn, 1 token counted as 1 "byte").
    NOT it+ot: the cache sizes the study sweeps are exactly
    {1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2} x sum(first-turn it),
    which holds exactly (ratio 1.0000 for both workloads) for obj_size = it
    and not for it+ot (0.94). Object size still grows across turns because
    each turn's input contains the accumulated conversation context.
  - time     = uint32 epoch seconds of started_at
  - next_access_vtime = request index (1-BASED, over the whole trace) of the
    session's next turn, -1 if this is the last access. This is the "oracle"
    field that lets BeladySize run; see the comment at the computation for
    why 1-based is load-bearing.

Record layout: packed little-endian, 24 bytes:
  <uint32 time> <uint64 obj_id> <uint32 obj_size> <int64 next_access_vtime>

Usage:
  python pipeline/caching/build_oracle_traces.py
  python pipeline/caching/build_oracle_traces.py --workloads minimax_m25
"""

import argparse
import glob
import hashlib
import json
import os
import pathlib
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chutes_sim import artifact


def _rel(path: str) -> str:
    """Repo-relative where possible, so the committed metadata is portable."""
    try:
        return str(pathlib.Path(path).resolve().relative_to(artifact.artifact_root()))
    except ValueError:
        return str(path)

# workload label (as it appears in the results files) -> model directory,
# both taken from config/workload.json.
WORKLOADS = {
    m["caching_workload_label"]: f"model={m['chute_id']}"
    for m in artifact.models().values()
}

# cache-size fractions of the byte working-set size
SIZE_FRACTIONS = artifact.load_config("cachesim")["sweep"]["cache_size_fractions"]

RECORD_DTYPE = np.dtype(
    [("time", "<u4"), ("obj_id", "<u8"), ("obj_size", "<u4"), ("next_access_vtime", "<i8")]
)


def hash_session_ids(session_ids: pd.Index) -> np.ndarray:
    """Stable 64-bit ids from session_id strings (first 8 bytes of md5, little-endian)."""
    out = np.empty(len(session_ids), dtype=np.uint64)
    for i, s in enumerate(session_ids):
        out[i] = np.frombuffer(hashlib.md5(s.encode()).digest()[:8], dtype="<u8")[0]
    return out


def preprocess_one(workload: str, model_dir: str, out_dir: str) -> dict:
    pattern = os.path.join(model_dir, "window=*", "sessions.parquet")
    paths = glob.glob(pattern)
    if len(paths) != 1:
        raise FileNotFoundError(f"expected exactly one parquet for {pattern}, got {paths}")
    parquet = os.path.realpath(paths[0])

    print(f"[{workload}] reading {parquet}")
    df = duckdb.sql(
        f"""
        SELECT epoch(started_at)::BIGINT AS ts, session_id, turn_id,
               it::BIGINT AS it, ot::BIGINT AS ot, ct::BIGINT AS ct
        FROM '{parquet}'
        ORDER BY started_at, session_id, turn_id
        """
    ).df()
    n = len(df)

    # oracle next-access virtual time: 1-based request index of the session's
    # next request, -1 if none. 1-based matches libCacheSim's n_req counter:
    # BeladySize scores log(next_access_vtime - n_req), so a 0-based index
    # makes an access-next-request distance of 0 -> log(0) = -inf -> its
    # eviction sampler recurses forever. (Hard-won lesson.)
    idx = pd.Series(np.arange(1, n + 1, dtype=np.int64))
    nxt = idx.groupby(df["session_id"].values).shift(-1)
    next_vtime = nxt.fillna(-1).to_numpy(dtype=np.int64)

    # 64-bit object ids per session
    codes, uniques = pd.factorize(df["session_id"])
    obj_ids = hash_session_ids(uniques)[codes]
    if len(np.unique(obj_ids[np.unique(codes, return_index=True)[1]])) != len(uniques):
        raise RuntimeError("64-bit session hash collision; use a different hash")

    rec = np.empty(n, dtype=RECORD_DTYPE)
    rec["time"] = df["ts"].to_numpy(dtype=np.uint32)
    rec["obj_id"] = obj_ids
    rec["obj_size"] = df["it"].to_numpy(dtype=np.uint32)
    rec["next_access_vtime"] = next_vtime

    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"{workload}.oracleGeneral")
    rec.tofile(trace_path)

    # working set size = sum over sessions of first-seen obj_size (what cachesim
    # computes when resolving fractional cache sizes)
    first_mask = np.zeros(n, dtype=bool)
    first_mask[np.unique(codes, return_index=True)[1]] = True
    wss_bytes = int(df["it"].to_numpy()[first_mask].sum())
    expected_sizes_kib = [int(f * wss_bytes / 1024) for f in SIZE_FRACTIONS]

    meta = {
        "workload": workload,
        "source_parquet": _rel(parquet),
        "n_requests": n,
        "n_sessions": int(len(uniques)),
        "wss_bytes": wss_bytes,
        "size_fractions": SIZE_FRACTIONS,
        "expected_cache_sizes_kib": expected_sizes_kib,
        "trace_path": _rel(trace_path),
        "record_bytes": RECORD_DTYPE.itemsize,
    }
    with open(os.path.join(out_dir, f"{workload}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # sample sessions per turn-count category (no prompt text exists in this
    # dataset, so we log per-turn token structure instead)
    write_samples(df, workload, out_dir)

    print(f"[{workload}] wrote {trace_path}: {n} req, {len(uniques)} sessions, "
          f"WSS={wss_bytes} B, expected sizes KiB={expected_sizes_kib}")
    return meta


def write_samples(df: pd.DataFrame, workload: str, out_dir: str, per_cat: int = 3):
    turns = df.groupby("session_id")["turn_id"].size()
    cats = {
        "single_turn": turns[turns == 1].index,
        "short_2_4": turns[(turns >= 2) & (turns <= 4)].index,
        "medium_5_9": turns[(turns >= 5) & (turns <= 9)].index,
        "long_10_plus": turns[turns >= 10].index,
    }
    samples = {}
    for cat, sids in cats.items():
        picked = list(sids[:per_cat])
        samples[cat] = [
            df[df["session_id"] == sid][["turn_id", "ts", "it", "ot", "ct"]]
            .to_dict(orient="records")
            for sid in picked
        ]
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    with open(os.path.join(samples_dir, f"{workload}_samples.json"), "w") as f:
        json.dump(samples, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(artifact.data_root()),
                    help="dataset root holding model=*/window=*/sessions.parquet "
                         "(default: $CHUTES_DATA_ROOT or output/<preprocess>/dataset)")
    ap.add_argument("--out-dir", default=str(artifact.cache_eviction_dir() / "traces"),
                    help="where to write the oracleGeneral traces "
                         "(default: output/<preprocess>/<workload>/cache_eviction/traces)")
    ap.add_argument("--workloads", nargs="*", default=list(WORKLOADS), choices=list(WORKLOADS))
    args = ap.parse_args()

    metas = []
    for w in args.workloads:
        model_dir = os.path.join(args.data_root, WORKLOADS[w])
        metas.append(preprocess_one(w, model_dir, args.out_dir))
    with open(os.path.join(args.out_dir, "preprocess_manifest.json"), "w") as f:
        json.dump(metas, f, indent=2)


if __name__ == "__main__":
    main()
