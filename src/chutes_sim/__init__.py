"""Session reconstruction and the load simulator behind both simulation studies.

  sessions.reconstruct_sessions    trace requests -> sessions (the rule is in sessions.py)
  sessions.load_session_parquet    sessions parquet -> list[Request] for the simulator
  runner.Simulator                 replay against N instances + LRU KV caches
  policies                         round_robin, load_first, cache_first, sticky
  experiment.run_experiment        one (model, N, cache, load_metric) cell
  artifact                         path and config resolution

cache.py, instance.py, policies.py and runner.py carry the design notes.
"""
from .config import CacheMode, LoadMetric, SimulationConfig
from .models import (
    InstanceMetricsRecord,
    InstanceTimelineRecord,
    PolicySummaryRecord,
    Request,
)
from .runner import SimulationResult, Simulator

__all__ = [
    "CacheMode",
    "LoadMetric",
    "InstanceMetricsRecord",
    "InstanceTimelineRecord",
    "PolicySummaryRecord",
    "Request",
    "SimulationConfig",
    "SimulationResult",
    "Simulator",
]
