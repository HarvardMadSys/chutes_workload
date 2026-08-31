"""Preprocessing for the ``7day_naive`` workload — DuckDB slice loading,
naive_chain session reconstruction, and sessions.parquet replay loading.

Verbatim merge of ``simulator/preprocess/{loader,function_filter,
naive_chain_builder,replay_loader}.py``. The naive_branch variant is
deliberately not ported — only ``naive_chain`` feeds Figure 20.

Naive chain builder: simplest possible per-user linear chains.

For each generation request B (forward time order; function_name in the
generation whitelist):
  - Scan the user's last ``lookback`` generation requests
    (any instance, any ct value).
  - A candidate A qualifies iff
        A is unpaired
        AND function_name(A) == function_name(B)     # same-fn only
        AND (it_A + ot_A) < it_B                     # strict growth
    (parent's full content fits inside child's input).
  - Pick the A with the smallest it_B − (it_A + ot_A) (tightest fit).
  - Mark A as paired (single branch, no fan-out).
  - If no candidate fits → B is a chain head (turn_id == 0).

No use of: cached_tokens (ct), instance_id, or per-user calibration.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .models import Request

try:  # progress bar is cosmetic — never a hard dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ── generation-function whitelist ────────────────────────────────────────
# A request is in scope iff its function_name is exactly one of these.
# Other function_names (get_models, embed, generate, e2e_invoke, …) are
# filtered out at SQL time and never reach the chain builder.
GENERATION_FUNCTIONS: frozenset[str] = frozenset(
    {"chat", "chat_stream", "completion", "completion_stream"}
)


def is_generation_function(fn: object) -> bool:
    if fn is None:
        return False
    return str(fn) in GENERATION_FUNCTIONS


# ── DuckDB slice loader ──────────────────────────────────────────────────

def load_slice(
    *,
    db_path: str,
    table_name: str,
    chute_id: str,
    start_time: datetime,
    duration: timedelta,
) -> pd.DataFrame:
    """Pull a (chute, time window) slice from DuckDB into a DataFrame.

    Returns columns: invocation_id, user_id, chute_id, instance_id,
    function_name, started_at, completed_at, it, ot, ct, ttft, tt.
    """
    end_time = start_time + duration
    # Generation-function whitelist (exact match) is pushed into SQL so
    # non-generation rows (e.g. e2e_invoke) never enter the chain builder.
    #
    # `tt` (total time) is a placeholder column: no analysis in this artifact
    # reads it, so the released trace does not carry it. It is emitted as NULL
    # so sessions.parquet keeps the column layout the published dataset has.
    query = f"""
        SELECT
            invocation_id, user_id, chute_id, instance_id,
            function_name, started_at, completed_at,
            it, ot, ct, ttft, CAST(NULL AS DOUBLE) AS tt
        FROM {table_name}
        WHERE chute_id = ?
          AND started_at >= ?
          AND started_at < ?
          AND user_id IS NOT NULL
          AND instance_id IS NOT NULL
          AND it IS NOT NULL
          AND ot IS NOT NULL
          AND function_name IN ('chat', 'chat_stream', 'completion', 'completion_stream')
        ORDER BY started_at
    """
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(query, [chute_id, start_time, end_time]).fetch_df()

    # Identifiers are compared and joined as strings throughout the chain
    # builder (parent_invocation_id is matched against invocation_id), while
    # the trace stores invocation_id as a dense integer — so normalize.
    for col in ("invocation_id", "user_id", "instance_id", "chute_id"):
        df[col] = df[col].astype(str)

    df["it"] = df["it"].fillna(0).astype("int64")
    df["ot"] = df["ot"].fillna(0).astype("int64")
    df["ct"] = df["ct"].fillna(0).astype("int64")
    return df


# ── naive_chain builder ──────────────────────────────────────────────────

@dataclass(frozen=True)
class NaiveChainParams:
    lookback: int = 10


def build_naive_chains(
    df: pd.DataFrame,
    *,
    params: NaiveChainParams,
    progress: bool = True,
) -> pd.DataFrame:
    df_sorted = df.sort_values("started_at").reset_index(drop=True)
    win: dict[str, deque[dict[str, Any]]] = {}
    chain_counter = 0
    out: list[dict[str, Any]] = []

    iterator = df_sorted.itertuples(index=False)
    if progress and tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(df_sorted),
            desc="naive_chain",
            unit="row",
            unit_scale=True,
            mininterval=0.5,
            file=sys.stderr,
            dynamic_ncols=True,
        )

    for row in iterator:
        invoc = str(row.invocation_id)
        user = str(row.user_id)
        inst = str(row.instance_id)
        fn = str(row.function_name) if row.function_name is not None else ""
        it = int(row.it)
        ot = int(row.ot)
        t = row.started_at

        if not is_generation_function(fn):
            out.append(
                {
                    "invocation_id": invoc,
                    "session_id": "-1",
                    "turn_id": 0,
                    "parent_invocation_id": None,
                    "structural_prefix_tokens": 0,
                    "link_confidence": "non_generation",
                }
            )
            continue

        q = win.setdefault(user, deque(maxlen=params.lookback))

        # Find best parent: unpaired, same function_name, strict growth, tightest fit.
        best: dict[str, Any] | None = None
        best_score: int = -1
        for cand in q:
            if cand["paired"]:
                continue
            if cand["fn"] != fn:
                continue
            ipo = cand["it_plus_ot"]
            if ipo < it:
                score = it - ipo
                if best is None or score < best_score:
                    best = cand
                    best_score = score

        if best is None:
            chain_counter += 1
            sid = f"{user}#{chain_counter}"
            q.append(
                {
                    "invoc": invoc,
                    "fn": fn,
                    "it_plus_ot": it + ot,
                    "paired": False,
                    "session_id": sid,
                    "turn_id": 0,
                    "started_at": t,
                    "instance_id": inst,
                }
            )
            out.append(
                {
                    "invocation_id": invoc,
                    "session_id": sid,
                    "turn_id": 0,
                    "parent_invocation_id": None,
                    "structural_prefix_tokens": 0,
                    "link_confidence": "turn_0",
                }
            )
        else:
            best["paired"] = True
            new_turn = best["turn_id"] + 1
            sid = best["session_id"]
            q.append(
                {
                    "invoc": invoc,
                    "fn": fn,
                    "it_plus_ot": it + ot,
                    "paired": False,
                    "session_id": sid,
                    "turn_id": new_turn,
                    "started_at": t,
                    "instance_id": inst,
                }
            )
            out.append(
                {
                    "invocation_id": invoc,
                    "session_id": sid,
                    "turn_id": new_turn,
                    "parent_invocation_id": best["invoc"],
                    "structural_prefix_tokens": best["it_plus_ot"],
                    "link_confidence": "naive_chain",
                }
            )

    extras = pd.DataFrame(out)
    return df_sorted.merge(extras, on="invocation_id", how="left")


def summarize_naive_chains(enriched: pd.DataFrame) -> dict[str, Any]:
    """Global chain statistics for a generation-only enriched frame."""
    n = len(enriched)
    if n == 0:
        return {
            "total_requests": 0,
            "sessions": 0,
            "linked_turns": 0,
            "chain_length_distribution": {},
            "token_reuse": {},
        }

    turn_0_mask = enriched["turn_id"] == 0
    sessions = int(turn_0_mask.sum())
    linked_turns = n - sessions

    chain_lens = enriched.groupby("session_id")["turn_id"].max()
    # Histogram is per-request over turn_id (NOT per-session over max-depth),
    # so bucket "0" == #(turn-0 rows) == sessions, by construction.
    turn_id_hist = enriched["turn_id"].value_counts().sort_index().head(20)
    length_dist = {
        "mean": float(chain_lens.mean()),
        "median": float(chain_lens.median()),
        "max": int(chain_lens.max()),
        "p90": float(chain_lens.quantile(0.9)),
        "p99": float(chain_lens.quantile(0.99)),
        "length_0_fraction": float((chain_lens == 0).sum()) / sessions,
        "turn_id_histogram": {
            int(k): int(v) for k, v in turn_id_hist.to_dict().items()
        },
    }

    input_tokens_total = int(enriched["it"].sum())
    structural_prefix_total = int(
        enriched.loc[~turn_0_mask, "structural_prefix_tokens"].sum()
    )
    token_reuse = {
        "input_tokens_total": input_tokens_total,
        "structural_prefix_tokens_total": structural_prefix_total,
        "structural_reuse_ratio": (
            structural_prefix_total / input_tokens_total if input_tokens_total else 0.0
        ),
    }

    return {
        "total_requests": n,
        "sessions": sessions,
        "linked_turns": linked_turns,
        "chain_length_distribution": length_dist,
        "token_reuse": token_reuse,
    }


# ── sessions.parquet → list[Request] ─────────────────────────────────────

_NEEDED_COLUMNS: list[str] = [
    "invocation_id",
    "user_id",
    "chute_id",
    "instance_id",
    "started_at",
    "completed_at",
    "it",
    "ot",
    "ct",
    "ttft",
    "session_id",
]


def load_session_parquet(path: str | Path) -> list[Request]:
    """Read a preprocessed sessions.parquet, return Requests sorted by
    ``started_at``.

    Each Request's ``explicit_session_id`` is set from the preprocessor's
    ``session_id`` column, so the simulator's cache state is keyed at the
    chain granularity rather than per-user.
    """
    df = pd.read_parquet(path, columns=_NEEDED_COLUMNS)
    if not df["started_at"].is_monotonic_increasing:
        df = df.sort_values("started_at").reset_index(drop=True)

    # Bulk-convert each column once instead of per-row in itertuples.
    started = df["started_at"].dt.to_pydatetime()
    completed = df["completed_at"].dt.to_pydatetime()
    invocation_ids = df["invocation_id"].astype(str).tolist()
    user_ids = df["user_id"].astype(str).tolist()
    chute_ids = df["chute_id"].astype(str).tolist()
    instance_ids = df["instance_id"].astype(str).tolist()
    session_ids = df["session_id"].astype(str).tolist()
    its = df["it"].astype("int64").tolist()
    ots = df["ot"].astype("int64").tolist()
    cts = df["ct"].astype("int64").tolist()
    ttfts = df["ttft"].tolist()  # may contain NaN

    requests: list[Request] = [None] * len(df)  # type: ignore[list-item]
    for i in range(len(df)):
        sess_raw = session_ids[i]
        if sess_raw == "-1":
            # Defensive: the SQL loader filters to generation functions,
            # so this branch is unreachable for fresh parquets. Kept for
            # back-compat with older sessions.parquet files.
            explicit_session: str | None = f"__nongeneration__:{invocation_ids[i]}"
        elif sess_raw:
            explicit_session = sess_raw
        else:
            explicit_session = None
        t = ttfts[i]
        # NaN != NaN — quicker than pd.notna in a hot Python loop.
        ttft = float(t) if t == t else None
        requests[i] = Request(
            invocation_id=invocation_ids[i],
            started_at=started[i],
            completed_at=completed[i],
            user_id=user_ids[i],
            chute_id=chute_ids[i],
            input_tokens=its[i],
            output_tokens=ots[i],
            cached_tokens_observed=cts[i],
            ttft=ttft,
            instance_id=instance_ids[i],
            explicit_session_id=explicit_session,
        )
    return requests
