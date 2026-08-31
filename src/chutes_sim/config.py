from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum


class CacheMode(Enum):
    INFINITE = "infinite"
    FINITE_LRU = "finite_lru"


class LoadMetric(Enum):
    """How the simulator measures per-instance load.

    All metrics are **event-driven**: when a request is scheduled onto
    an instance, the instance's load is incremented by that request's
    weight ``right away``; when the request's ``completed_at`` is
    reached (drained via ``Instance.advance_to``), the same weight is
    removed.

    Variants (per-request additive weight at schedule time):
      - REQUEST_COUNT      += 1                              [default]
      - INPUT_TOKENS       += request.input_tokens
      - TOTAL_TOKENS       += request.input_tokens + request.output_tokens
                              (assumes the LB sees output tokens at
                              schedule time)
      - REQUEST_DURATION   += duration_seconds
                              (assumes the LB sees the request's
                              execution time at schedule time)
    """

    REQUEST_COUNT = "request_count"
    INPUT_TOKENS = "input_tokens"
    TOTAL_TOKENS = "total_tokens"
    REQUEST_DURATION = "request_duration"


@dataclass(frozen=True)
class SimulationConfig:
    num_instances: int
    cache_mode: CacheMode
    cache_size_tokens_per_instance: int | None
    policy_name: str
    session_ttl: timedelta | None = None
    load_window: timedelta | None = None
    # Which signal load-aware policies (load_first, cache_first fallback,
    # sticky cold-start, etc.) compare across instances. See ``LoadMetric``.
    load_metric: LoadMetric = LoadMetric.REQUEST_COUNT
    # Fixed trace-time spacing between InstanceTimelineRecord snapshots.
    # 60s default → 1440 samples/day per instance. Set to None to disable
    # timeline sampling entirely (no per-instance over-time data).
    timeline_sample_seconds: float | None = 60.0
    # Fixed seed for the published configuration (config/figure20.json).
    random_seed: int = 10

    def __post_init__(self) -> None:
        if self.num_instances <= 0:
            raise ValueError("num_instances must be > 0")
        if self.cache_mode is CacheMode.FINITE_LRU and self.cache_size_tokens_per_instance is None:
            raise ValueError("FINITE_LRU requires cache_size_tokens_per_instance")
        if self.cache_mode is CacheMode.INFINITE and self.cache_size_tokens_per_instance is not None:
            raise ValueError("INFINITE cache must not have a capacity set")
        if self.timeline_sample_seconds is not None and self.timeline_sample_seconds <= 0:
            raise ValueError("timeline_sample_seconds must be > 0 or None")
