"""Routing policies — base contract + the four policies Figure 20 uses.

Verbatim merge of ``simulator/policies/{base,round_robin,load_first,
cache_first,sticky}.py``.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .config import LoadMetric
from .instance import Instance
from .models import Request

if TYPE_CHECKING:
    from .state import SimulationState


LEAST_LOADED_TOP_K: int = 3


class SchedulerPolicy(ABC):
    """Routing policy contract.

    Implementations of ``select_instance`` must only read **LB-visible**
    fields from ``request`` (see ``Request`` docstring in ``models``).
    In particular:

      - ``request.session_id`` / ``request.explicit_session_id`` are
        oracle information (chain id from preprocess) and **must not**
        be read in a routing decision — doing so leaks structure the
        real load balancer cannot see in production.
      - ``request.instance_id``, ``request.output_tokens``,
        ``request.completed_at``, ``request.ttft``, and
        ``request.cached_tokens_observed`` are also off-limits (they're
        either production's choice or future / post-hoc data).
      - ``state`` is post-hoc bookkeeping; policies must not read it
        either. If a policy needs memory (e.g. sticky's
        ``user_id -> instance`` map), it should maintain that map
        inside its own instance state.

    Probing each candidate instance's cache via
    ``inst.estimate_cache_hit(request)`` is allowed — it returns a byte
    count (the prefix-hash-probe stand-in) without exposing the cache key.

    The Simulator calls ``bind_rng`` before each run to inject a seeded
    ``random.Random``. Policies should forward it to ``least_loaded`` so
    that load-balanced choices are randomized but reproducible.
    """

    name: str = "abstract"

    # Set by ``Simulator.reset`` before ``run``; ``None`` falls back to
    # deterministic behavior (strict minimum) in ``least_loaded``.
    _rng: random.Random | None = None

    def bind_rng(self, rng: random.Random) -> None:
        self._rng = rng

    @abstractmethod
    def select_instance(
        self,
        request: Request,
        instances: list[Instance],
        state: "SimulationState",
    ) -> int: ...


def least_loaded(
    instances: list[Instance],
    metric: LoadMetric | None = None,
    *,
    rng: random.Random | None = None,
    top_k: int = LEAST_LOADED_TOP_K,
) -> Instance:
    """Pick from the bottom-``top_k`` least-loaded instances.

    If ``metric`` is ``None`` (default), uses each instance's configured
    routing load (``SimulationConfig.load_metric``). Pass an explicit
    ``LoadMetric`` to override on a per-policy basis.

    If ``rng`` is ``None``, returns the strict minimum (ties broken by
    ``instance_id`` for determinism). If an RNG is supplied, samples
    uniformly from the ``min(top_k, N)`` coolest instances after the
    same deterministic ordering — so the candidate set is reproducible
    given the load state, only the choice within it is random.
    """
    if metric is None:
        key = lambda inst: (inst.routing_load, inst.instance_id)  # noqa: E731
    else:
        key = lambda inst: (inst.current_load(metric), inst.instance_id)  # noqa: E731
    ordered = sorted(instances, key=key)
    if rng is None:
        return ordered[0]
    candidates = ordered[: min(top_k, len(ordered))]
    return rng.choice(candidates)


class RoundRobinPolicy(SchedulerPolicy):
    """Assign requests to instances in a rotating cycle, ignoring load
    and cache state. Useful as a simple, deterministic balance baseline."""

    name = "round_robin"

    def __init__(self) -> None:
        self._counter: int = 0

    def select_instance(
        self,
        request: Request,
        instances: list[Instance],
        state: "SimulationState",
    ) -> int:
        choice = self._counter % len(instances)
        self._counter += 1
        return choice


class LoadFirstPolicy(SchedulerPolicy):
    """Route to the currently least-loaded instance.

    The "load" signal is always whatever ``SimulationConfig.load_metric``
    is set to — a single, unambiguous definition for the whole config.
    To compare metrics, vary ``SimulationConfig.load_metric`` across
    experiments rather than parameterizing the policy.
    """

    name = "load_first"

    def select_instance(
        self,
        request: Request,
        instances: list[Instance],
        state: "SimulationState",
    ) -> int:
        return least_loaded(instances, rng=self._rng).instance_id


class CacheFirstPolicy(SchedulerPolicy):
    name = "cache_first"

    def select_instance(
        self,
        request: Request,
        instances: list[Instance],
        state: "SimulationState",
    ) -> int:
        best: Instance | None = None
        best_reusable = 0
        for inst in instances:
            reusable = inst.estimate_cache_hit(request).reusable_tokens
            if reusable > best_reusable:
                best_reusable = reusable
                best = inst

        if best is not None and best_reusable > 0:
            return best.instance_id

        return least_loaded(instances, rng=self._rng).instance_id


class StickyPolicy(SchedulerPolicy):
    """Route a user to the same instance as their previous request.

    The router maintains its own ``user_id -> last_instance`` map built up
    from its own past routing decisions. It does NOT read ``SimulationState``,
    which is oracle bookkeeping used only for evaluation.
    """

    name = "sticky"

    def __init__(self) -> None:
        self._last_instance_by_user: dict[str, int] = {}

    def select_instance(
        self,
        request: Request,
        instances: list[Instance],
        state: "SimulationState",
    ) -> int:
        chosen = self._last_instance_by_user.get(request.user_id)
        if chosen is None:
            chosen = least_loaded(instances, rng=self._rng).instance_id
        self._last_instance_by_user[request.user_id] = chosen
        return chosen


POLICY_CLASSES: dict[str, type[SchedulerPolicy]] = {
    "round_robin": RoundRobinPolicy,
    "load_first": LoadFirstPolicy,
    "cache_first": CacheFirstPolicy,
    "sticky": StickyPolicy,
}


