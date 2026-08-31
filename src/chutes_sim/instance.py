from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime

from .cache import CacheLookupResult, CacheUpdateResult, KVCache
from .config import LoadMetric
from .models import Request


@dataclass
class InstanceLoad:
    """Cumulative counters over the entire run for this instance."""

    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_adjusted_tokens: int = 0
    cache_hits: int = 0


@dataclass(order=True)
class _InFlight:
    """One in-flight request on an instance, ordered by completed_at.

    ``weights`` carries the per-LoadMetric contribution this request made
    when scheduled so completion can subtract exactly what was added.
    ``tokens_per_sec`` is tracked separately as an analytics-only signal
    (cache-adjusted serve rate), not a routable LoadMetric.
    """

    completed_at: datetime
    weights: dict[LoadMetric, float] = field(
        compare=False, default_factory=lambda: {m: 0.0 for m in LoadMetric}
    )
    tokens_per_sec: float = field(compare=False, default=0.0)
    invocation_id: str = field(compare=False, default="")


class Instance:
    def __init__(
        self,
        instance_id: int,
        cache: KVCache,
        load_metric: LoadMetric = LoadMetric.REQUEST_COUNT,
    ) -> None:
        self.instance_id: int = instance_id
        self.cache: KVCache = cache
        self.load: InstanceLoad = InstanceLoad()

        # Configured "routing load" — what ``least_loaded`` compares.
        self._load_metric: LoadMetric = load_metric

        # Time-aware load state. ``_in_flight`` is a min-heap ordered by
        # ``completed_at`` so we can pop everything that has finished
        # before the simulator's clock. ``_active_load`` tracks the
        # running sum of weights per LoadMetric — all updated on the
        # same two events (apply_request adds, advance_to subtracts).
        self._in_flight: list[_InFlight] = []
        self._active_load: dict[LoadMetric, float] = {m: 0.0 for m in LoadMetric}
        # Analytics-only signal: cache-adjusted tokens-per-second served
        # by the instance. Not a routable LoadMetric — kept here because
        # the metrics/plotting layers report fairness over it.
        self._active_tokens_per_sec: float = 0.0

    # ----- time-aware load (instantaneous, as of last advance_to call) -----

    def advance_to(self, t: datetime) -> None:
        """Drain in-flight requests whose completed_at <= t."""
        while self._in_flight and self._in_flight[0].completed_at <= t:
            done = heapq.heappop(self._in_flight)
            for metric, weight in done.weights.items():
                self._active_load[metric] -= weight
            self._active_tokens_per_sec -= done.tokens_per_sec
        # Guard against tiny float drift after many subtractions.
        for metric, value in self._active_load.items():
            if value < 0.0:
                self._active_load[metric] = 0.0
        if self._active_tokens_per_sec < 0.0:
            self._active_tokens_per_sec = 0.0

    @property
    def load_metric(self) -> LoadMetric:
        return self._load_metric

    def current_load(self, metric: LoadMetric) -> float:
        """Live in-flight load measured by ``metric``."""
        return self._active_load[metric]

    @property
    def routing_load(self) -> float:
        """Live in-flight load by the configured metric (what policies see)."""
        return self._active_load[self._load_metric]

    @property
    def active_queries(self) -> int:
        # Equals current_load(REQUEST_COUNT) but kept as an integer for
        # timeline/metrics callers that expect an int.
        return len(self._in_flight)

    @property
    def active_tokens_per_sec(self) -> float:
        return self._active_tokens_per_sec

    # ----- cache + request handling -----

    def estimate_cache_hit(self, request: Request) -> CacheLookupResult:
        return self.cache.lookup(request.session_id, request.input_tokens)

    def apply_request(
        self, request: Request, simulated_cached_tokens: int
    ) -> CacheUpdateResult:
        cache_adjusted = (
            request.input_tokens - simulated_cached_tokens
        ) + request.output_tokens

        self.load.request_count += 1
        self.load.input_tokens += request.input_tokens
        self.load.output_tokens += request.output_tokens
        self.load.total_tokens += request.total_tokens
        self.load.cached_tokens += simulated_cached_tokens
        self.load.cache_adjusted_tokens += cache_adjusted
        if simulated_cached_tokens > 0:
            self.load.cache_hits += 1

        # Each metric's per-request weight, applied to the instance
        # right away at schedule time (and subtracted at completed_at).
        duration = (request.completed_at - request.started_at).total_seconds()
        weights: dict[LoadMetric, float] = {
            LoadMetric.REQUEST_COUNT: 1.0,
            LoadMetric.INPUT_TOKENS: float(request.input_tokens),
            LoadMetric.TOTAL_TOKENS: float(
                request.input_tokens + request.output_tokens
            ),
            LoadMetric.REQUEST_DURATION: float(duration if duration > 0.0 else 0.0),
        }
        tokens_per_sec = (cache_adjusted / duration) if duration > 0.0 else 0.0

        if duration > 0.0:
            heapq.heappush(
                self._in_flight,
                _InFlight(
                    completed_at=request.completed_at,
                    weights=weights,
                    tokens_per_sec=tokens_per_sec,
                    invocation_id=request.invocation_id,
                ),
            )
            for metric, weight in weights.items():
                self._active_load[metric] += weight
            self._active_tokens_per_sec += tokens_per_sec

        # Cache stores the full conversation length the next turn can reuse:
        # this turn's input + the assistant's response.
        return self.cache.update(
            request.session_id, request.input_tokens + request.output_tokens
        )
