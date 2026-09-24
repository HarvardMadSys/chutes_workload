"""Helpers shared by the paper/<section>/reproduce.py scripts."""
from __future__ import annotations

import duckdb
import pandas as pd

#: Trace timestamps count from the first request, which is 1970-01-01 00:00:00.
TRACE_START = pd.Timestamp("1970-01-01")


class LazyDB:
    """The DuckDB trace view, opened on first use.

    A section whose query results are all cached never opens the trace, so its
    figures can be redrawn without the 91 GB download.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._con = None

    def __getattr__(self, name):
        if self._con is None:
            self._con = duckdb.connect(self.path, read_only=True)
        return getattr(self._con, name)


def trace_day(ts):
    """Whole days since the start of the trace (the first day is day 0).

    Works on a timestamp, a Series or an Index."""
    return (pd.to_datetime(ts) - TRACE_START) // pd.Timedelta(days=1)


def trace_month(ts):
    """Month of the trace, starting at 1, the numbering release_cohort uses.

    Works on a timestamp, a Series or an Index."""
    ts = pd.to_datetime(ts)
    year, month = (ts.dt.year, ts.dt.month) if isinstance(ts, pd.Series) else (ts.year, ts.month)
    return (year - 1970) * 12 + month
