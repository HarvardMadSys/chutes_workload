from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

SessionId = str


@dataclass(frozen=True, slots=True)
class Request:
    """One trace request handed to the simulator.

    Fields fall into two visibility groups. **Policies (load balancers)
    must only read the LB-visible fields.** The oracle / post-hoc fields
    are populated from trace metadata or downstream preprocess and are
    consumed exclusively by the cache layer, the duration model, and
    post-run metrics.

    Why the split: in production, a real LB sees the user identity, the
    prompt size, the current fleet load, and the result of probing each
    instance's cache. It does *not* know which conversation/chain the
    request belongs to (that's exactly what prefix caching has to
    discover), nor how many output tokens the response will produce,
    nor what production actually chose. Reading those in a routing
    decision would make the simulator's policy comparison unfair.

    Allowed in ``SchedulerPolicy.select_instance``:
      - ``request.user_id``       (sticky / consistent-hash routing)
      - ``request.chute_id``      (multi-model fleets)
      - ``request.input_tokens``  (prompt size)
      - ``request.started_at``    (time-of-day if needed)
      - ``inst.estimate_cache_hit(request)``  — probes the instance's
        cache and returns reusable byte count, which is the
        prefix-hash-probe stand-in. The instance internally uses
        ``session_id`` as the cache key; that's allowed *inside* the
        cache, but the policy only sees the byte count it returns.

    NOT to be used in routing decisions:
      - ``request.session_id`` / ``request.explicit_session_id``
        (oracle: chain id from preprocess; using it would let the LB
        "see" prefix structure the real system has to discover)
      - ``request.instance_id``           (production's actual choice)
      - ``request.cached_tokens_observed``(production's recorded ct)
      - ``request.output_tokens``         (future — not known at route time)
      - ``request.completed_at``          (future)
      - ``request.ttft``                  (future)
    """

    invocation_id: str

    # --- LB-visible (policies may read these) ---
    user_id: str
    chute_id: str
    input_tokens: int
    started_at: datetime

    # --- oracle / post-hoc (cache + metrics only; policies must NOT read) ---
    output_tokens: int
    completed_at: datetime
    cached_tokens_observed: int
    ttft: float | None
    instance_id: str | None = None
    # ``explicit_session_id`` is the cache layer's hidden routing key.
    # It is exposed only through ``session_id`` below and consumed by the
    # cache and metrics layers — never by ``SchedulerPolicy.select_instance``.
    explicit_session_id: str | None = None

    @property
    def session_id(self) -> SessionId:
        """Oracle prefix-hash stand-in. **Do not read from a policy.**

        Used by ``KVCache`` to key cache entries and by ``SimulationState``
        + ``MetricsCollector`` for per-session bookkeeping.
        """
        if self.explicit_session_id is not None:
            return self.explicit_session_id
        return self.user_id

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class InstanceTimelineRecord:
    """One per-instance snapshot at a fixed trace-time interval.

    Two kinds of fields:

    - **Instantaneous** (active_queries, active_tokens_per_sec): the
      in-flight load at ``timestamp``, drained via ``Instance.advance_to``.
    - **Cumulative** (request_count, input_tokens, output_tokens,
      simulated_cached_tokens, cache_adjusted_tokens, cache_hits): the
      instance's ``InstanceLoad`` counters as of ``timestamp``. Diff
      consecutive samples to derive per-interval rates (req/min, tok/min,
      hit rate, etc.) — replaces what the per-request log used to give.
    """

    timestamp: datetime
    policy_name: str
    instance_id: int
    # Instantaneous (in-flight at timestamp)
    active_queries: int
    active_tokens_per_sec: float
    # Cumulative counters as of timestamp
    request_count: int
    input_tokens: int
    output_tokens: int
    simulated_cached_tokens: int
    cache_adjusted_tokens: int
    cache_hits: int


@dataclass(frozen=True)
class InstanceMetricsRecord:
    policy_name: str
    instance_id: int
    request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    simulated_cached_tokens: int
    cache_adjusted_tokens: int
    cache_occupancy_tokens: int
    cache_entry_count: int
    evicted_tokens: int
    evicted_entries: int
    request_hit_rate: float
    token_hit_rate: float
    # Timeline summaries (over samples taken every
    # ``SimulationConfig.timeline_sample_seconds`` of trace time).
    active_queries_mean: float
    active_queries_max: int
    active_queries_p90: float
    active_tokens_per_sec_mean: float
    active_tokens_per_sec_max: float
    active_tokens_per_sec_p90: float


@dataclass(frozen=True)
class PolicySummaryRecord:
    policy_name: str
    num_instances: int
    cache_size_tokens_per_instance: int | None
    request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    simulated_cached_tokens: int
    token_hit_rate: float
    request_hit_rate: float
    # Unfairness across instances, computed on two signals each:
    #   _req_count   : cumulative per-instance request count
    #   _token_load  : per-instance time-averaged active_tokens_per_sec
    max_mean_req_count: float
    max_mean_token_load: float
    cv_req_count: float
    cv_token_load: float
    jain_req_count: float
    jain_token_load: float
    # Absolute working-set totals (from the LRU-immune shadow record):
    #   total_logical_cache_tokens  = Σ_session max(it+ot ever stored)
    #                                = workload's unique working set (policy-invariant)
    #   total_physical_cache_tokens = Σ_session Σ_instance (max it+ot ever stored)
    #                                = actual KV storage held under this policy
    total_logical_cache_tokens: int
    total_physical_cache_tokens: int
    effective_replication_ratio: float
    effective_cache_size_ratio: float
