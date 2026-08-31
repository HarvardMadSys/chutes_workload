from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from .models import (
    InstanceMetricsRecord,
    InstanceTimelineRecord,
    PolicySummaryRecord,
)


class ResultWriter:
    """Writes simulation outputs to Parquet under output_dir."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _to_df(self, records: list[Any]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in records])

    def _write(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / name
        df.to_parquet(path, index=False)
        return path

    def write_instances(self, records: list[InstanceMetricsRecord]) -> Path:
        return self._write(self._to_df(records), "instance_metrics.parquet")

    def write_summary(self, records: list[PolicySummaryRecord]) -> Path:
        return self._write(self._to_df(records), "policy_summary.parquet")

    def write_timeline(self, records: list[InstanceTimelineRecord]) -> Path:
        return self._write(self._to_df(records), "instance_load_timeline.parquet")

    def write_overview(self, overview: dict[str, Any], filename: str = "run_overview.json") -> Path:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(overview, f, indent=2, default=_json_default, sort_keys=True)
        return path


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
