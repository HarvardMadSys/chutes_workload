"""Where the artifact's inputs and outputs live.

    config/*.json                      settings the pipeline stages read
    output/trace/chutes_trace.parquet  the released trace (make fetch)
    output/trace_view.duckdb           DuckDB view all_metrics_user over it (make trace)
    output/sessions/<model>.parquet    reconstructed sessions (make sessions)
    output/routing_sweep/              routing simulation cells (make routing)
    output/caching/                    libCacheSim traces, build and results (make caching)

Overrides: CHUTES_TRACE_DIR (directory holding the trace), CHUTES_DB_PATH
(the DuckDB view), CHUTES_OUTPUT_ROOT (everything under output/).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


def artifact_root() -> Path:
    """The repository root."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """``config/<name>.json``."""
    return json.loads((artifact_root() / "config" / f"{name}.json").read_text())


def workload_config() -> dict:
    """``config/workload.json``: the trace and the two simulated models."""
    return load_config("workload")


def models() -> dict[str, dict]:
    """The simulated models, keyed by the names every stage uses (deepseek_v32, minimax_m25)."""
    return workload_config()["models"]


def output_root() -> Path:
    return Path(os.environ.get("CHUTES_OUTPUT_ROOT", artifact_root() / "output")).resolve()


def trace_file() -> Path:
    trace_dir = Path(os.environ.get("CHUTES_TRACE_DIR", output_root() / "trace"))
    return trace_dir.resolve() / workload_config()["trace"]["file"]


def db_path() -> str:
    return os.environ.get("CHUTES_DB_PATH", str(output_root() / "trace_view.duckdb"))


def cohorts_path() -> Path:
    """Each user's release cohort; small enough to ship in the repository."""
    return artifact_root() / "data/trace/user_cohorts.parquet"


def sessions_dir() -> Path:
    return output_root() / "sessions"


def sessions_parquet(model: str, root: Path | None = None) -> Path:
    """``<root>/<model>.parquet``, where root defaults to output/sessions."""
    return (root or sessions_dir()) / f"{model}.parquet"


def routing_sweep_dir() -> Path:
    return output_root() / "routing_sweep"


def caching_dir() -> Path:
    return output_root() / "caching"
