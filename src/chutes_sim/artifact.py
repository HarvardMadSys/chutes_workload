"""Artifact-level path and config resolution.

Every pipeline stage resolves its inputs and outputs through this module,
so the artifact has exactly one place that knows where things live.

Layout (see README.md):

    <root>/config/{workload,figure20,cachesim}.json
    <root>/output/<preprocess>/dataset/                 sessions.parquet
    <root>/output/<preprocess>/<workload>/routing_sweep/
    <root>/output/<preprocess>/<workload>/cache_eviction/

Environment overrides:

    CHUTES_ARTIFACT_ROOT   repository root (default: inferred from this file)
    CHUTES_TRACE_DIR       directory holding the downloaded trace parquet
    CHUTES_DB_PATH         the DuckDB view over it (scripts/build_trace_view.py)
    CHUTES_DATA_ROOT       dataset root holding model=*/window=*/sessions.parquet
    CHUTES_OUTPUT_ROOT     root for generated artifacts (default: <root>/output)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "artifact_root", "config_dir", "load_config", "workload_config", "trace_dir",
    "trace_file", "cohorts_path", "models", "model", "data_root", "output_root",
    "workload_output_root", "routing_sweep_dir", "cache_eviction_dir",
    "sessions_parquet", "window_dir_name", "db_path",
]


def artifact_root() -> Path:
    """Repository root: the directory holding ``config/workload.json``."""
    env = os.environ.get("CHUTES_ARTIFACT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "config" / "workload.json").is_file():
            return cand
    # Installed non-editably: fall back to the CWD if it looks like the repo.
    cwd = Path.cwd().resolve()
    for cand in (cwd, *cwd.parents):
        if (cand / "config" / "workload.json").is_file():
            return cand
    raise RuntimeError(
        "Cannot locate the artifact root (no config/workload.json found above "
        f"{here} or {cwd}). Set CHUTES_ARTIFACT_ROOT."
    )


def config_dir() -> Path:
    return artifact_root() / "config"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """Load ``config/<name>.json`` (``.json`` optional)."""
    fn = name if name.endswith(".json") else f"{name}.json"
    return json.loads((config_dir() / fn).read_text())


def workload_config() -> dict:
    return load_config("workload")


def models() -> dict[str, dict]:
    return workload_config()["models"]


def model(key: str) -> dict:
    try:
        return models()[key]
    except KeyError:
        raise SystemExit(f"unknown model key {key!r}; known: {sorted(models())}") from None


def preprocess_name() -> str:
    return workload_config()["preprocess"]


def workload_name() -> str:
    return workload_config()["workload"]


def output_root() -> Path:
    env = os.environ.get("CHUTES_OUTPUT_ROOT")
    return Path(env).resolve() if env else artifact_root() / "output"


def workload_output_root() -> Path:
    """``output/<preprocess>/<workload>/`` — the per-experiment root."""
    return output_root() / preprocess_name() / workload_name()


def data_root() -> Path:
    """Dataset root: ``output/<preprocess>/dataset`` unless overridden."""
    env = os.environ.get("CHUTES_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return output_root() / preprocess_name() / "dataset"


def routing_sweep_dir() -> Path:
    return workload_output_root() / "routing_sweep"


def cache_eviction_dir() -> Path:
    return workload_output_root() / "cache_eviction"


def window_dir_name(m: dict) -> str:
    return m["window_dir"]


def sessions_parquet(key: str, *, root: Path | None = None) -> Path:
    m = model(key)
    base = root if root is not None else data_root()
    return base / f"model={m['chute_id']}" / m["window_dir"] / "sessions.parquet"


def trace_dir() -> Path:
    """Directory holding the downloaded trace parquet."""
    env = os.environ.get("CHUTES_TRACE_DIR")
    if env:
        return Path(env).resolve()
    return artifact_root() / workload_config()["source"]["trace_dir_default"]


def trace_file() -> Path:
    """The anonymized trace: one parquet, the artifact's only download."""
    return trace_dir() / workload_config()["source"]["file"]


def cohorts_path() -> Path:
    """``user_cohorts.parquet`` — small enough to live in the repository."""
    return artifact_root() / workload_config()["source"]["cohorts"]


def db_path() -> str:
    """The DuckDB view over the anonymized trace.

    Built by ``scripts/build_trace_view.py``; every analysis queries it as the
    table ``all_metrics_user``.
    """
    env = os.environ.get("CHUTES_DB_PATH")
    if env:
        return env
    return str(artifact_root() / workload_config()["source"]["view_default"])
