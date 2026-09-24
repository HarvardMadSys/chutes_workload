"""Experiment manager: sweep a list of policies on one (model, window, cache)
setting and emit per-policy parquets + a global summary table.

Trimmed port of ``simulator/experiment.py``: the raw DuckDB / raw-parquet
TraceReader source paths are dropped — the Figure 20 pipeline always
simulates from a preprocessed ``sessions.parquet``.

Per-policy outputs under ``<root>/<experiment_name>/policy=<label>/``:
  - ``instance_metrics.parquet``        — one row per instance, cumulative
  - ``instance_load_timeline.parquet``  — one row per (instance, 1-min sample)
  - ``policy_summary.parquet``          — one row, global metrics
  - ``run_overview.json``               — manifest of this policy's run

Experiment-level outputs at ``<root>/<experiment_name>/``:
  - ``experiment_config.json``          — manifest of what was run
  - ``summary.csv``  /  ``summary.md``  — one row per policy
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CacheMode, LoadMetric, SimulationConfig
from .policies import POLICY_CLASSES, SchedulerPolicy
from .sessions import load_session_parquet
from .runner import Simulator
from .writer import ResultWriter


@dataclass(frozen=True)
class PolicySpec:
    """One policy variant in an experiment sweep.

    ``label`` is the filesystem-safe name used in output paths and plot
    legends. ``name`` selects the implementation. ``params`` are passed
    straight to the policy constructor.
    """

    label: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> SchedulerPolicy:
        return POLICY_CLASSES[self.name](**self.params)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    # Preprocessed sessions.parquet (with session_id / parent /
    # structural_prefix_tokens) — the simulator's cache key comes from it.
    sessions_parquet_path: str
    # Provenance metadata (recorded in the manifest, not used to select rows —
    # the parquet already IS the selection).
    chute_id: str = ""
    model_name: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    # Simulation.
    num_instances: int = 0
    cache_mode: CacheMode = CacheMode.FINITE_LRU
    cache_size_tokens_per_instance: int | None = None
    # Definition of "load" used by load-aware policies in this experiment.
    load_metric: LoadMetric = LoadMetric.REQUEST_COUNT
    # Seeds the RNG that ``least_loaded`` uses to sample from the bottom-k
    # coolest instances. Same seed → identical runs; bump for replicates.
    random_seed: int = 0
    policies: list[PolicySpec] = field(default_factory=list)
    # Preprocess tag recorded in the manifest (e.g. "naive_chain").
    preprocess_method: str | None = "naive_chain"
    # Output root: experiment files land under ``<output_dir>/<name>/``.
    output_dir: Path = field(default_factory=Path)

    @property
    def root(self) -> Path:
        return self.output_dir / self.name

    def policy_dir(self, label: str) -> Path:
        return self.root / f"policy={label}"


def run_experiment(config: ExperimentConfig) -> Path:
    """Run every policy in the config, write all per-policy outputs and the
    experiment-level summary table. Returns the experiment root directory."""
    print(
        f"Loading preprocessed sessions from {config.sessions_parquet_path} ...",
        file=sys.stderr,
    )
    t0 = time.time()
    requests = load_session_parquet(config.sessions_parquet_path)
    load_seconds = time.time() - t0
    print(f"Loaded {len(requests)} requests in {load_seconds:.1f}s", file=sys.stderr)

    config.root.mkdir(parents=True, exist_ok=True)

    # One Simulator instance, reused across every policy in the sweep —
    # the request stream is held by reference and never reparsed.
    sim = Simulator(requests)

    summary_rows: list[dict[str, Any]] = []
    for spec in config.policies:
        policy = spec.build()
        sim_cfg = SimulationConfig(
            num_instances=config.num_instances,
            cache_mode=config.cache_mode,
            cache_size_tokens_per_instance=config.cache_size_tokens_per_instance,
            policy_name=policy.name,
            load_metric=config.load_metric,
            random_seed=config.random_seed,
        )
        run_dir = config.policy_dir(spec.label)
        t0 = time.time()
        result = sim.run(sim_cfg, policy)
        sim_seconds = time.time() - t0

        writer = ResultWriter(output_dir=run_dir)
        writer.write_instances(result.instance_records)
        writer.write_summary([result.summary])
        writer.write_timeline(result.timeline_records)
        writer.write_overview(
            {
                "policy_label": spec.label,
                "policy_name": spec.name,
                "policy_params": spec.params,
                "num_instances": config.num_instances,
                "cache_mode": config.cache_mode.value,
                "cache_size_tokens_per_instance": config.cache_size_tokens_per_instance,
                "load_metric": config.load_metric.value,
                "request_count_loaded": len(requests),
                "elapsed_sec": sim_seconds,
                "summary": asdict(result.summary),
            }
        )

        summary_rows.append(
            {"policy_label": spec.label, **asdict(result.summary), "elapsed_sec": sim_seconds}
        )
        s = result.summary
        print(
            f"{spec.label:24s} hit={s.token_hit_rate:.3f} "
            f"jain[tok]={s.jain_token_load:.3f} rep={s.effective_replication_ratio:.2f} "
            f"sim={sim_seconds:.1f}s",
            file=sys.stderr,
        )

    manifest = {
        "name": config.name,
        "preprocess_method": config.preprocess_method,
        "source": {
            "sessions_parquet_path": config.sessions_parquet_path,
        },
        "selection": {
            "chute_id": config.chute_id,
            "model_name": config.model_name,
            "window_start": config.window_start,
            "window_end": config.window_end,
            "request_count_loaded": len(requests),
        },
        "config": {
            "num_instances": config.num_instances,
            "cache_mode": config.cache_mode.value,
            "cache_size_tokens_per_instance": config.cache_size_tokens_per_instance,
            "load_metric": config.load_metric.value,
            "random_seed": config.random_seed,
        },
        "policies": [
            {"label": s.label, "name": s.name, "params": s.params}
            for s in config.policies
        ],
    }
    with open(config.root / "experiment_config.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str, sort_keys=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.root / "summary.csv", index=False)
    _write_summary_markdown(summary_df, config.root / "summary.md", config)

    return config.root


# Curated subset shown in the markdown summary. The full table (every
# PolicySummaryRecord field) is always written to summary.csv.
DEFAULT_SUMMARY_COLUMNS: list[str] = [
    "policy_label",
    "request_count",
    "token_hit_rate",
    "request_hit_rate",
    "max_mean_req_count",
    "max_mean_token_load",
    "jain_req_count",
    "jain_token_load",
    "cv_req_count",
    "cv_token_load",
    "total_logical_cache_tokens",
    "total_physical_cache_tokens",
    "effective_replication_ratio",
    "effective_cache_size_ratio",
]


def _write_summary_markdown(
    df: pd.DataFrame,
    path: Path,
    config: ExperimentConfig,
    columns: list[str] | None = None,
) -> None:
    cols = [c for c in (columns or DEFAULT_SUMMARY_COLUMNS) if c in df.columns]
    sub = df[cols].copy()

    lines: list[str] = []
    lines.append(f"# Experiment: {config.name}")
    lines.append("")
    lines.append(f"- model: `{config.model_name or config.chute_id}`")
    lines.append(f"- window: {config.window_start} → {config.window_end}")
    lines.append(f"- num_instances: {config.num_instances}")
    cache_line = f"- cache: {config.cache_mode.value}"
    if config.cache_size_tokens_per_instance is not None:
        cache_line += f" (cap = {config.cache_size_tokens_per_instance:,} tokens)"
    lines.append(cache_line)
    lines.append("")
    lines.append(_dataframe_to_markdown(sub))
    path.write_text("\n".join(lines) + "\n")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if abs(v) >= 1000 or (v != 0 and abs(v) < 0.001):
                return f"{v:.3e}"
            return f"{v:.3f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    headers = list(df.columns)
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])
