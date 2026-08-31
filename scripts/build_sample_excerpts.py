#!/usr/bin/env python3
"""Derive the committed session-sample excerpts from a preprocessing run.

`pipeline/preprocess/build_sessions.py` writes a full `chain_samples.json`
per model — two recovered sessions for every chain length it saw, which is
far too large to commit. This picks one session per turn-length category and
writes the small excerpt the artifact ships:

    data/sessions/chain_samples_excerpt_<model>.json

The trace carries no prompt text, so a "sample" is a recovered conversation
represented by its per-turn token counts, timing, and the production instance
each turn landed on — the per-turn-category sample log the project convention
asks for.

Usage:
    python scripts/build_sample_excerpts.py            # all models
    python scripts/build_sample_excerpts.py --models minimax
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chutes_sim import artifact

#: (category, lowest chain length, highest chain length or None for open-ended)
CATEGORIES = [("short_2_4", 2, 4), ("medium_5_9", 5, 9), ("long_10_plus", 10, None)]

NOTE = (
    "Excerpt of the full chain_samples.json (written by "
    "pipeline/preprocess/build_sessions.py): one recovered session per "
    "turn-length category, as a nested tree rooted at turn 0, plus one "
    "single-turn request that arrived with a non-zero cached-token count. "
    "The trace carries no prompt text, so each turn is represented by its "
    "token counts (it/ot/ct), timing, and the instance it landed on. "
    "Identifiers are the released (anonymized) ones."
)


def excerpt(samples: dict, model_name: str, chute_id: str, window: str) -> dict:
    by_len = samples["samples_by_length"]
    out = {
        "model_name": model_name,
        "chute_id": chute_id,
        "window": window,
        "note": NOTE,
        "samples_by_turn_category": {},
    }
    single = by_len.get("turn_0_with_ct_gt_0") or []
    if single:
        out["samples_by_turn_category"]["single_turn"] = single[0]
    lengths = sorted(int(k) for k in by_len if k.isdigit())
    for name, lo, hi in CATEGORIES:
        pick = next((n for n in lengths if n >= lo and (hi is None or n <= hi)), None)
        if pick is not None and by_len[str(pick)]:
            out["samples_by_turn_category"][name] = by_len[str(pick)][0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=sorted(artifact.models()))
    ap.add_argument("--out-dir", type=Path,
                    default=artifact.artifact_root() / "data" / "sessions")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        m = artifact.model(key)
        src = artifact.sessions_parquet(key).parent / "chain_samples.json"
        if not src.exists():
            print(f"SKIP {key}: no {src} (run `make dataset` first)")
            continue
        doc = excerpt(json.loads(src.read_text()), m["model_name"], m["chute_id"],
                      m["window_dir"])
        out = args.out_dir / f"chain_samples_excerpt_{key}.json"
        out.write_text(json.dumps(doc, indent=2) + "\n")
        cats = ", ".join(doc["samples_by_turn_category"])
        print(f"wrote {out}  ({cats})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
