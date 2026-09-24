"""Session reconstruction: which requests of the trace form one conversation.

The trace has one row per request and no conversation id. Conversations can
still be recovered from token counts, because a chat client resends the whole
conversation on every turn: the prompt of turn k+1 contains the prompt and the
answer of turn k.

The rule, applied to one model's requests in started_at order
--------------------------------------------------------------
Request B is the next turn of an earlier request A when

  1. A and B have the same user_id,
  2. A is one of that user's last LOOKBACK (10) requests,
  3. no other request has continued A yet,
  4. A and B call the same function (chat, chat_stream, completion, ...), and
  5. it_A + ot_A < it_B: A's prompt plus A's answer fit inside B's prompt.

If several requests qualify, B continues the one that fits most tightly (the
largest it_A + ot_A; ties go to the oldest). If none qualifies, B starts a new
session. Production cache hits (ct) and instance ids are not used, so the
sessions do not depend on how Chutes routed or cached the requests.

Output: output/sessions/<model>.parquet
---------------------------------------
The window's trace rows (invocation_id, user_id, chute_id, instance_id,
function_name, started_at, completed_at, it, ot, ct, ttft) plus four columns:

  session_id            "<user_id>#<n>", where n numbers the window's sessions from 1
  turn_id               0 for the first request of a session, then 1, 2, ...
  parent_invocation_id  the request this one continues (null on turn 0)
  prefix_tokens         it + ot of that parent: the part of this prompt that the
                        previous turn already contained (0 on turn 0)

How the two simulation studies use the sessions
-----------------------------------------------
Routing (pipeline/routing): the session is the unit of KV-cache reuse. Each
instance caches one entry per session, so a request reuses cached tokens only
on an instance that already served its session. load_session_parquet() below
turns the file into the simulator's requests.

Caching (pipeline/caching): each session is one cache object, and each request
is one access to it, of size it (the prompt grows turn by turn).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from .models import Request

#: How many of a user's most recent requests a new request may continue (rule 2).
LOOKBACK = 10

#: The functions that carry a conversation; other rows (embeddings, model lists) are skipped.
GENERATION_FUNCTIONS = ("chat", "chat_stream", "completion", "completion_stream")


@dataclass(slots=True)
class Turn:
    """One request, as the later requests of the same user see it."""

    invocation_id: str
    function_name: str
    context_tokens: int  # it + ot: the whole conversation up to and including this turn
    session_id: str
    turn_id: int
    continued: bool = False  # set once a later request continues this one (rule 3)


def load_trace_window(db_path: str, table: str, chute_id: str,
                      start: datetime, end: datetime) -> pd.DataFrame:
    """One model's chat and completion requests with started_at in [start, end)."""
    with duckdb.connect(db_path, read_only=True) as con:
        requests = con.execute(f"""
            SELECT invocation_id, user_id, chute_id, instance_id, function_name,
                   started_at, completed_at, it, ot, COALESCE(ct, 0) AS ct, ttft
            FROM {table}
            WHERE chute_id = ? AND started_at >= ? AND started_at < ?
              AND user_id IS NOT NULL AND instance_id IS NOT NULL
              AND it IS NOT NULL AND ot IS NOT NULL
              AND function_name IN {GENERATION_FUNCTIONS}
            ORDER BY started_at
        """, [chute_id, start, end]).fetch_df()
    # The trace stores invocation_id as an integer; ids are compared as strings.
    for col in ("invocation_id", "user_id", "chute_id", "instance_id"):
        requests[col] = requests[col].astype(str)
    requests[["it", "ot", "ct"]] = requests[["it", "ot", "ct"]].astype("int64")
    return requests


def find_parent(recent: deque[Turn], function_name: str, input_tokens: int) -> Turn | None:
    """The turn a new request continues (rules 3-5), or None if it starts a session."""
    parent = None
    for turn in recent:  # oldest first, so ties go to the oldest
        if turn.continued or turn.function_name != function_name:
            continue
        fits = turn.context_tokens < input_tokens
        if fits and (parent is None or turn.context_tokens > parent.context_tokens):
            parent = turn
    return parent


def reconstruct_sessions(requests: pd.DataFrame) -> pd.DataFrame:
    """Add session_id, turn_id, parent_invocation_id and prefix_tokens to ``requests``."""
    # The query already orders by started_at. This sort is kept because it fixes
    # the order of requests with equal timestamps, which decides a few links.
    requests = requests.sort_values("started_at").reset_index(drop=True)

    recent_turns: dict[str, deque[Turn]] = {}  # user_id -> that user's last LOOKBACK turns
    n_sessions = 0
    session_id, turn_id, parent_invocation_id, prefix_tokens = [], [], [], []

    for req in requests.itertuples(index=False):
        it, ot = int(req.it), int(req.ot)
        recent = recent_turns.setdefault(req.user_id, deque(maxlen=LOOKBACK))
        parent = find_parent(recent, req.function_name, it)

        if parent is None:  # no earlier turn fits: a new session starts
            n_sessions += 1
            turn = Turn(req.invocation_id, req.function_name, it + ot,
                        session_id=f"{req.user_id}#{n_sessions}", turn_id=0)
        else:  # the next turn of the parent's session
            parent.continued = True
            turn = Turn(req.invocation_id, req.function_name, it + ot,
                        session_id=parent.session_id, turn_id=parent.turn_id + 1)
        recent.append(turn)

        session_id.append(turn.session_id)
        turn_id.append(turn.turn_id)
        parent_invocation_id.append(None if parent is None else parent.invocation_id)
        prefix_tokens.append(0 if parent is None else parent.context_tokens)

    return requests.assign(session_id=session_id, turn_id=turn_id,
                           parent_invocation_id=parent_invocation_id,
                           prefix_tokens=prefix_tokens)


def load_session_parquet(path: str | Path) -> list[Request]:
    """A sessions parquet as the routing simulator's requests, in started_at order.

    The session id becomes the request's cache key (``explicit_session_id``).
    The simulator's routing policies never read it: a real load balancer does
    not know which conversation a request belongs to.
    """
    df = pd.read_parquet(path, columns=[
        "invocation_id", "user_id", "chute_id", "instance_id", "session_id",
        "started_at", "completed_at", "it", "ot", "ct", "ttft"])
    if not df["started_at"].is_monotonic_increasing:
        df = df.sort_values("started_at").reset_index(drop=True)

    rows = zip(
        df["invocation_id"].astype(str).tolist(), df["user_id"].astype(str).tolist(),
        df["chute_id"].astype(str).tolist(), df["instance_id"].astype(str).tolist(),
        df["session_id"].astype(str).tolist(),
        df["started_at"].dt.to_pydatetime(), df["completed_at"].dt.to_pydatetime(),
        df["it"].astype("int64").tolist(), df["ot"].astype("int64").tolist(),
        df["ct"].astype("int64").tolist(), df["ttft"].tolist(),
    )
    return [
        Request(invocation_id=inv, user_id=user, chute_id=chute, instance_id=instance,
                explicit_session_id=session, started_at=start, completed_at=end,
                input_tokens=it, output_tokens=ot, cached_tokens_observed=ct,
                ttft=float(ttft) if ttft == ttft else None)  # NaN != NaN
        for inv, user, chute, instance, session, start, end, it, ot, ct, ttft in rows
    ]
