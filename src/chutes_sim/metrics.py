from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from math import sqrt

from .config import SimulationConfig
from .instance import Instance
from .models import (
    InstanceMetricsRecord,
    InstanceTimelineRecord,
    PolicySummaryRecord,
    SessionId,
)


class MetricsCollector:
    """Accumulates the per-instance load timeline and emits final aggregates.

    Per-request and per-session records are intentionally not logged — the
    simulator's three outputs are:

      - ``timeline_records`` (sampled every ``sample_interval`` of trace time,
        one per instance per sample) — load-over-time.
      - ``InstanceMetricsRecord`` × N instances — cumulative + timeline-derived
        scalars (mean/max/p90 of active queries and tokens/sec).
      - ``PolicySummaryRecord`` × 1 — global hit rate, unfairness metrics,
        replication ratio.
    """

    def __init__(self, policy_name: str, config: SimulationConfig) -> None:
        self.policy_name: str = policy_name
        self.config: SimulationConfig = config
        self.timeline_records: list[InstanceTimelineRecord] = []

    def record_timeline(
        self, timestamp: datetime, instances: list[Instance]
    ) -> None:
        """Snapshot every instance at ``timestamp``. Assumes the caller has
        already ``advance_to(timestamp)``'d each instance so the in-flight
        state is correct."""
        for inst in instances:
            load = inst.load
            self.timeline_records.append(
                InstanceTimelineRecord(
                    timestamp=timestamp,
                    policy_name=self.policy_name,
                    instance_id=inst.instance_id,
                    active_queries=inst.active_queries,
                    active_tokens_per_sec=float(inst.active_tokens_per_sec),
                    request_count=load.request_count,
                    input_tokens=load.input_tokens,
                    output_tokens=load.output_tokens,
                    simulated_cached_tokens=load.cached_tokens,
                    cache_adjusted_tokens=load.cache_adjusted_tokens,
                    cache_hits=load.cache_hits,
                )
            )

    def build_instance_metrics(self, instances: list[Instance]) -> list[InstanceMetricsRecord]:
        by_inst: dict[int, list[InstanceTimelineRecord]] = defaultdict(list)
        for r in self.timeline_records:
            by_inst[r.instance_id].append(r)

        records: list[InstanceMetricsRecord] = []
        for inst in instances:
            load = inst.load
            request_hit_rate = (
                load.cache_hits / load.request_count if load.request_count else 0.0
            )
            token_hit_rate = (
                load.cached_tokens / load.input_tokens if load.input_tokens else 0.0
            )
            timeline = by_inst.get(inst.instance_id, [])
            q_values = [r.active_queries for r in timeline]
            tps_values = [r.active_tokens_per_sec for r in timeline]
            q_mean, q_max, q_p90 = _summary_int(q_values)
            tps_mean, tps_max, tps_p90 = _summary_float(tps_values)
            records.append(
                InstanceMetricsRecord(
                    policy_name=self.policy_name,
                    instance_id=inst.instance_id,
                    request_count=load.request_count,
                    input_tokens=load.input_tokens,
                    output_tokens=load.output_tokens,
                    total_tokens=load.total_tokens,
                    simulated_cached_tokens=load.cached_tokens,
                    cache_adjusted_tokens=load.cache_adjusted_tokens,
                    cache_occupancy_tokens=inst.cache.current_size_tokens,
                    cache_entry_count=inst.cache.entry_count,
                    evicted_tokens=inst.cache.total_evicted_tokens,
                    evicted_entries=inst.cache.total_evicted_entries,
                    request_hit_rate=request_hit_rate,
                    token_hit_rate=token_hit_rate,
                    active_queries_mean=q_mean,
                    active_queries_max=q_max,
                    active_queries_p90=q_p90,
                    active_tokens_per_sec_mean=tps_mean,
                    active_tokens_per_sec_max=tps_max,
                    active_tokens_per_sec_p90=tps_p90,
                )
            )
        return records

    def build_policy_summary(
        self, instances: list[Instance]
    ) -> PolicySummaryRecord:
        request_count = sum(i.load.request_count for i in instances)
        input_tokens = sum(i.load.input_tokens for i in instances)
        output_tokens = sum(i.load.output_tokens for i in instances)
        total_tokens = sum(i.load.total_tokens for i in instances)
        cached_tokens = sum(i.load.cached_tokens for i in instances)
        cache_hits = sum(i.load.cache_hits for i in instances)

        token_hit_rate = cached_tokens / input_tokens if input_tokens else 0.0
        request_hit_rate = cache_hits / request_count if request_count else 0.0

        # Unfairness signals: cumulative request count and time-averaged
        # active tokens/sec, per instance.
        per_inst_tps: dict[int, list[float]] = defaultdict(list)
        for r in self.timeline_records:
            per_inst_tps[r.instance_id].append(r.active_tokens_per_sec)
        token_loads = [
            (sum(per_inst_tps.get(i.instance_id, [])) / max(1, len(per_inst_tps.get(i.instance_id, []))))
            for i in instances
        ]
        req_counts: list[float] = [float(i.load.request_count) for i in instances]

        max_mean_req_count = _max_mean_ratio(req_counts)
        max_mean_token_load = _max_mean_ratio(token_loads)
        cv_req_count = _coefficient_of_variation(req_counts)
        cv_token_load = _coefficient_of_variation(token_loads)
        jain_req_count = _jain_fairness(req_counts)
        jain_token_load = _jain_fairness(token_loads)

        # Cache totals computed inline from per-instance shadow records.
        # Per (session, instance) max it+ot ever stored — LRU-immune, so
        # this reflects the cumulative spread the routing policy created.
        per_session_max: dict[SessionId, int] = {}
        total_physical = 0
        for inst in instances:
            for sess_id, size in inst.cache.ever_snapshot_sessions().items():
                total_physical += size
                prev = per_session_max.get(sess_id, 0)
                if size > prev:
                    per_session_max[sess_id] = size
        total_logical = sum(per_session_max.values())
        if total_logical > 0:
            effective_replication = total_physical / total_logical
            effective_cache_size_ratio = 1.0 / effective_replication if effective_replication else 0.0
        else:
            effective_replication = 0.0
            effective_cache_size_ratio = 0.0

        return PolicySummaryRecord(
            policy_name=self.policy_name,
            num_instances=self.config.num_instances,
            cache_size_tokens_per_instance=self.config.cache_size_tokens_per_instance,
            request_count=request_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            simulated_cached_tokens=cached_tokens,
            token_hit_rate=token_hit_rate,
            request_hit_rate=request_hit_rate,
            max_mean_req_count=max_mean_req_count,
            max_mean_token_load=max_mean_token_load,
            cv_req_count=cv_req_count,
            cv_token_load=cv_token_load,
            jain_req_count=jain_req_count,
            jain_token_load=jain_token_load,
            total_logical_cache_tokens=int(total_logical),
            total_physical_cache_tokens=int(total_physical),
            effective_replication_ratio=effective_replication,
            effective_cache_size_ratio=effective_cache_size_ratio,
        )


def _summary_int(xs: list[int]) -> tuple[float, int, float]:
    if not xs:
        return 0.0, 0, 0.0
    xs_sorted = sorted(xs)
    mean = sum(xs_sorted) / len(xs_sorted)
    p90_idx = max(0, int(0.9 * (len(xs_sorted) - 1)))
    return float(mean), int(xs_sorted[-1]), float(xs_sorted[p90_idx])


def _summary_float(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    xs_sorted = sorted(xs)
    mean = sum(xs_sorted) / len(xs_sorted)
    p90_idx = max(0, int(0.9 * (len(xs_sorted) - 1)))
    return float(mean), float(xs_sorted[-1]), float(xs_sorted[p90_idx])


def _jain_fairness(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    floats = [float(v) for v in values]
    s = sum(floats)
    if s == 0:
        return 0.0
    sq = sum(v * v for v in floats)
    if sq == 0:
        return 0.0
    return (s * s) / (len(floats) * sq)


def _max_mean_ratio(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return max(values) / mean


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return sqrt(variance) / mean
