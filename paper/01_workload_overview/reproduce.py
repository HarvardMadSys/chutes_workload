#!/usr/bin/env python3
"""§2 Background: the five panels of the workload-overview figure.

Draws the panels of fig:workload_temporal_structure into
figures/paper/01_workload_overview/:

  (a) fig_request_rate_day.pdf              requests per day
  (b) fig_token_rate_day.pdf                input + output tokens per day
  (c) fig_daily_unique_models.pdf           distinct models with requests, per day
  (d) fig_function_name_daily_requests.pdf  requests per day by API function
  (e) fig_seasonality.pdf                   average requests per hour of the week

The x-axis of (a) to (d) is the trace day: whole days since the trace's first
request, which is day 0. The trace's timestamps count from that request, so
they are not calendar dates.

Each panel is one DuckDB aggregate over the trace, cached as a parquet file in
output/paper/cache/. A cached result is reused and only the plot is redrawn;
--recompute reruns every query. The trace is opened on the first query, so a
run with every cache present never touches it.

Usage:
    python paper/01_workload_overview/reproduce.py [--db PATH] [--recompute]
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import LazyDB, trace_day  # noqa: E402
from _paths import DB_DEFAULT, section_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the anonymized trace "
                         "(scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true", help="rerun the DuckDB queries even if cache files exist")
args = parser.parse_args()
DB_PATH = args.db
CACHED = not args.recompute   # True: reuse cache files and only replot

PDF_DIR, CACHE_DIR = section_paths("01_workload_overview")
con = LazyDB(DB_PATH)

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'lines.linewidth': 1.5,
})


def cached_query(name, sql):
    """The result of `sql` on the trace, read from CACHE_DIR/name when cached."""
    path = CACHE_DIR / name
    if CACHED and path.exists():
        df = pd.read_parquet(path)
        print(f"Loaded {len(df):,} rows from {path}")
    else:
        df = con.sql(sql).fetchdf()
        df.to_parquet(path)
        print(f"Saved {len(df):,} rows to {path}")
    return df


def by_day(df):
    """A daily query result, indexed by its `day` column."""
    df['day'] = pd.to_datetime(df['day'])
    return df.set_index('day').sort_index()


def style_grid(ax):
    """Dashed grid on both major and minor axes, with major more visible than minor."""
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.12, linestyle='--', linewidth=0.4)


def save(fig, name):
    fig.savefig(PDF_DIR / name, bbox_inches='tight', dpi=300)
    print(f"saved {PDF_DIR / name}")
    plt.close(fig)


def plot_daily(series, ylabel, fmt, name):
    """A daily series as one line: the layout of panels (a) to (c)."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(trace_day(series.index), series, color='#1f77b4', rasterized=True)
    ax.set_xlabel('Day')
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    style_grid(ax)
    fig.tight_layout(pad=0.4)
    save(fig, name)


# ── (a) fig_request_rate_day.pdf: requests per day ──────────────────────────
requests = by_day(cached_query('daily_requests.parquet', """
    SELECT DATE_TRUNC('day', started_at)::DATE AS day, COUNT(*) AS cnt
    FROM all_metrics_user
    WHERE started_at IS NOT NULL
    GROUP BY 1
    ORDER BY 1
"""))
plot_daily(requests['cnt'], 'Requests', lambda x, _: f'{x/1e6:.0f}M', 'fig_request_rate_day.pdf')


# ── (b) fig_token_rate_day.pdf: input + output tokens per day ───────────────
# Unlike (a), counts only requests that completed (completed_at is set).
tokens = by_day(cached_query('daily_tokens.parquet', """
    SELECT
        DATE_TRUNC('day', started_at)::DATE AS day,
        SUM(COALESCE(it, 0) + COALESCE(ot, 0)) AS tokens
    FROM all_metrics_user
    WHERE started_at IS NOT NULL
      AND completed_at IS NOT NULL
    GROUP BY 1
    ORDER BY 1
"""))
plot_daily(tokens['tokens'], 'Total Tokens', lambda x, _: f'{x/1e9:.0f}B', 'fig_token_rate_day.pdf')


# ── (c) fig_daily_unique_models.pdf: distinct models with requests, per day ─
# The query also counts distinct users per day; the paper plots only the models.
unique = by_day(cached_query('daily_unique_users_models.parquet', """
    SELECT
        DATE_TRUNC('day', started_at)::DATE AS day,
        COUNT(DISTINCT user_id) AS unique_users,
        COUNT(DISTINCT chute_id) AS unique_models
    FROM all_metrics_user
    WHERE started_at IS NOT NULL
    GROUP BY 1
    ORDER BY 1
"""))
plot_daily(unique['unique_models'], 'Unique models', lambda x, _: f'{x:.0f}',
           'fig_daily_unique_models.pdf')


# ── (d) fig_function_name_daily_requests.pdf: requests per day by API function
# The four generation functions; every other function_name counts as Other.
FUNCTIONS = {   # function group: (label, color), in plot and legend order
    'chat':              ('Chat', '#2ca02c'),
    'chat_stream':       ('Chat Stream', '#ff7f0e'),
    'completion':        ('Completion', '#d62728'),
    'completion_stream': ('Completion Stream', '#1f77b4'),
    'other':             ('Other', '#7f7f7f'),
}

by_function = cached_query('function_name_daily_requests.parquet', """
    SELECT
        DATE_TRUNC('day', started_at)::DATE AS day,
        CASE
            WHEN function_name IN ('completion_stream', 'chat_stream', 'chat', 'completion')
                THEN function_name
            ELSE 'other'
        END AS function_group,
        COUNT(*) AS requests
    FROM all_metrics_user
    WHERE started_at IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
""")
by_function['day'] = pd.to_datetime(by_function['day'])
per_day = (
    by_function
    .pivot_table(index='day', columns='function_group', values='requests', fill_value=0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
for func, (label, color) in FUNCTIONS.items():
    if func not in per_day.columns:
        continue
    ax.plot(
        trace_day(per_day.index),
        per_day[func],
        linewidth=1.4 if func != 'other' else 1.1,
        alpha=0.95 if func != 'other' else 0.75,
        color=color,
        label=label,
        rasterized=True,
    )
ax.set_xlabel('Day')
ax.set_ylabel('Requests')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x < 1e7 else f'{x/1e6:.0f}M'))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='upper left', frameon=False, ncol=2, columnspacing=1.0, handlelength=1.5, fontsize=8)
ax.tick_params(axis='both', which='major', length=4.5, width=0.8)
ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)
ax.set_ylim(0, 23e6)
style_grid(ax)
fig.tight_layout(pad=0.4)
save(fig, 'fig_function_name_daily_requests.pdf')


# ── (e) fig_seasonality.pdf: average requests per hour of the week ──────────
hour_of_week = cached_query('hour_of_week_requests.parquet', """
    SELECT
        dayofweek(started_at)         AS dow,
        EXTRACT(HOUR FROM started_at) AS hour,
        COUNT(*)                      AS cnt
    FROM all_metrics_user
    WHERE started_at IS NOT NULL
    GROUP BY 1, 2
""")
heatmap = hour_of_week.pivot_table(index='dow', columns='hour', values='cnt', fill_value=0)
heatmap = heatmap.reindex(range(7)).reindex(range(24), axis=1)

# A year holds 53 of some weekdays and 52 of the others, so divide each
# weekday's row by its number of days in panel (a) to get a per-week average.
days_per_dow = requests.index.to_series().dt.dayofweek.value_counts().reindex(range(7)).astype(int)
heatmap_avg = heatmap.div(days_per_dow, axis=0)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.imshow(
    heatmap_avg.values / 1e6, aspect='auto', cmap='YlOrRd',
    interpolation='nearest',
)
ax.set_yticks(range(7))
ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_xlabel('Hour of Day (UTC)', size=10)
ax.set_xticks(range(0, 24, 4))
ax.set_xticklabels(range(0, 24, 4))

cbar = fig.colorbar(im, ax=ax, shrink=1, pad=0.1, location='top', orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Avg requests / week (M)', size=11)

fig.tight_layout()
save(fig, 'fig_seasonality.pdf')
