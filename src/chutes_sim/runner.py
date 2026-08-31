from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .cache import KVCache
from .config import CacheMode, SimulationConfig
from .instance import Instance
from .metrics import MetricsCollector
from .models import (
    InstanceMetricsRecord,
    InstanceTimelineRecord,
    PolicySummaryRecord,
    Request,
)
from .policies import SchedulerPolicy
from .state import SimulationState


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    policy_name: str
    instance_records: list[InstanceMetricsRecord]
    timeline_records: list[InstanceTimelineRecord]
    summary: PolicySummaryRecord


class Simulator:
    """Trace-driven simulator that you instantiate **once** with a request
    stream, then reuse across many (config, policy) combinations.

    Typical usage:

        sim = Simulator(requests)          # load requests once
        r1 = sim.run(cfg1, policy1)        # auto-resets state, then runs
        r2 = sim.run(cfg2, policy2)        # auto-resets again

    Or with an explicit reset:

        sim = Simulator(requests)
        sim.reset(cfg, policy)             # configure but don't run yet
        result = sim.run()                 # uses the last reset() config

    Every ``reset()`` / ``run(config, policy)`` rebuilds the instance
    list, KV caches, metrics collector, and per-session state from
    scratch — nothing leaks across runs except the request stream itself.
    """

    def __init__(self, requests: list[Request]) -> None:
        self._requests: list[Request] = requests
        # All per-run state is rebuilt by reset(); start unconfigured.
        self.config: SimulationConfig | None = None
        self.policy: SchedulerPolicy | None = None
        self.instances: list[Instance] = []
        self.state: SimulationState = SimulationState()
        self.metrics: MetricsCollector | None = None
        self._sample_interval: timedelta | None = None
        self._next_sample_at: datetime | None = None

    @property
    def num_requests(self) -> int:
        return len(self._requests)

    def reset(self, config: SimulationConfig, policy: SchedulerPolicy) -> None:
        """Discard previous run state and configure for a new run.

        Builds fresh ``Instance`` objects (with new ``KVCache``s),
        a fresh ``SimulationState``, and a fresh ``MetricsCollector``.
        Safe to call between ``run()``s to swap cache size, instance
        count, policy, or any other config knob.
        """
        self.config = config
        self.policy = policy
        self.policy.bind_rng(random.Random(config.random_seed))
        self.instances = [
            Instance(
                instance_id=i,
                cache=self._make_cache(config),
                load_metric=config.load_metric,
            )
            for i in range(config.num_instances)
        ]
        self.state = SimulationState()
        self.metrics = MetricsCollector(policy_name=policy.name, config=config)
        self._sample_interval = (
            timedelta(seconds=config.timeline_sample_seconds)
            if config.timeline_sample_seconds is not None
            else None
        )
        self._next_sample_at = None

    def run(
        self,
        config: SimulationConfig | None = None,
        policy: SchedulerPolicy | None = None,
    ) -> SimulationResult:
        """Simulate the request stream and return aggregates.

        If both ``config`` and ``policy`` are supplied, calls ``reset()``
        first (the common single-shot pattern). If neither is supplied,
        runs against the last ``reset()`` configuration. Mixing one of
        them is an error.
        """
        if config is not None and policy is not None:
            self.reset(config, policy)
        elif config is not None or policy is not None:
            raise ValueError("Pass both config and policy to run(), or neither.")
        if self.config is None or self.policy is None:
            raise RuntimeError(
                "Simulator has no configuration. Call reset(config, policy) "
                "or run(config, policy) first."
            )
        for request in self._requests:
            self._step(request)
        return self._finalize()

    # ----- internals -----

    @staticmethod
    def _make_cache(config: SimulationConfig) -> KVCache:
        capacity = (
            None
            if config.cache_mode is CacheMode.INFINITE
            else config.cache_size_tokens_per_instance
        )
        return KVCache(capacity_tokens=capacity)

    def _emit_samples_up_to(self, t: datetime) -> None:
        """Emit timeline snapshots at every interval boundary <= ``t``."""
        if self._sample_interval is None:
            return
        if self._next_sample_at is None:
            self._next_sample_at = t
        while self._next_sample_at <= t:
            for inst in self.instances:
                inst.advance_to(self._next_sample_at)
            assert self.metrics is not None
            self.metrics.record_timeline(self._next_sample_at, self.instances)
            self._next_sample_at += self._sample_interval

    def _step(self, request: Request) -> None:
        assert self.policy is not None
        self._emit_samples_up_to(request.started_at)
        for inst in self.instances:
            inst.advance_to(request.started_at)
        instance_id = self.policy.select_instance(request, self.instances, self.state)
        if not 0 <= instance_id < len(self.instances):
            raise ValueError(
                f"Policy {self.policy.name} returned invalid instance_id={instance_id}"
            )
        instance = self.instances[instance_id]
        lookup = instance.cache.lookup(request.session_id, request.input_tokens)
        instance.apply_request(request, lookup.reusable_tokens)

    def _finalize(self) -> SimulationResult:
        assert self.config is not None and self.policy is not None and self.metrics is not None
        instance_records = self.metrics.build_instance_metrics(self.instances)
        summary = self.metrics.build_policy_summary(self.instances)
        return SimulationResult(
            config=self.config,
            policy_name=self.policy.name,
            instance_records=instance_records,
            timeline_records=self.metrics.timeline_records,
            summary=summary,
        )
