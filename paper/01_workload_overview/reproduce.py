#!/usr/bin/env python3
"""Section 2 — Workload Overview: figure reproduction.

Reproduces the five panels of the "Temporal structure of aggregate demand"
figure (fig:workload_temporal_structure) in Section 2 (Background / Workload
Overview) of the paper. The code is the paper's own figure script; only the
paths were adapted.

Figures produced (written to figures/paper/01_workload_overview/):

1. fig_request_rate_day.pdf              — daily request rate over the one-year trace (panel a)
2. fig_token_rate_day.pdf                — daily total token volume, input + output (panel b)
3. fig_daily_unique_models.pdf           — number of distinct models active per day (panel c)
   fig_daily_unique_users.pdf            — byproduct of the same query
4. fig_function_name_daily_requests.pdf  — daily request rate by API function name (panel d)
   fig_function_name_daily_requests_legend.pdf — standalone legend (appendix)
5. fig_seasonality.pdf                   — hour-of-day x day-of-week request heatmap (panel e)

Requirements: the anonymized one-year DuckDB trace (see --db), plus Python
packages duckdb, pandas, numpy, matplotlib, seaborn, and fastparquet.
Heavy queries cache their results as parquet files under output/paper/cache/.

Usage:
    python 01_section2_workload_overview.py [--db PATH] [--recompute]

By default, cached parquet files are reused when present (only plotting is
redone). Pass --recompute to rerun the heavy DuckDB queries from scratch.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator, NullLocator, FuncFormatter
import seaborn as sns

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DB_DEFAULT, section_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the anonymized trace "
                         "(scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true", help="rerun heavy DuckDB queries even if cache files exist")
args = parser.parse_args()
DB_PATH = args.db
CACHED = not args.recompute   # True: reuse cache files and just replot

# ────────────────────────────────────────────────────────────────────────────
# Setup: plotting style, output dirs, DB connection
# ────────────────────────────────────────────────────────────────────────────

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
    # line width 1.5
    'lines.linewidth': 1.5,
})
sns.set_palette('tab10')


def style_grid(ax):
    """Dashed grid on both major and minor axes, with major more visible than minor."""
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.12, linestyle='--', linewidth=0.4)


def save_legend(handles, labels, path, ncol=None, figsize=None):
    """Render a standalone legend as its own PDF."""
    if ncol is None:
        ncol = len(labels)
    if figsize is None:
        figsize = (max(1.5, 1.1 * len(labels)) + 0.5, 0.5)
    fig_l = plt.figure(figsize=figsize)
    fig_l.legend(
        handles, labels,
        loc='center', ncol=ncol, frameon=False,
        columnspacing=1.2, handlelength=1.8,
    )
    fig_l.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig_l)


PDF_DIR, CACHE_DIR = section_paths("01_workload_overview")
PDF_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(exist_ok=True)
con = duckdb.connect(DB_PATH, read_only=True)

con.sql("PRAGMA threads=48;")
con.sql("PRAGMA memory_limit='1000GB'")

TABLE_NAME = "all_metrics_user"
# Quick sanity check
row_count = con.sql(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}").fetchone()[0]
print(f"Connected to DuckDB — {row_count:,} rows in {TABLE_NAME}")


# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_request_rate_day.pdf
# Daily request rate over the full one-year trace — panel (a), "Daily request
# rate". "Consistent" version: uses the same `WHERE started_at IS NOT NULL`
# population as the seasonality heatmap below. The seasonality block reuses
# `df_rate_c` from this block to normalize by day-of-week counts.
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: Global request rate (daily) — consistent with weekly heatmap ──
# Uses the same WHERE clause as the seasonality heatmap below so both
# figures describe the same population of requests.

cached = CACHED
cache_file = CACHE_DIR / 'request_rate_daily_consistent.parquet'
out_dir = PDF_DIR

if not cached or not cache_file.exists():
    df_rate_c = con.sql(f"""
        SELECT DATE_TRUNC('day', started_at)::DATE AS day, COUNT(*) AS cnt
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    df_rate_c.to_parquet(cache_file, engine="fastparquet")
    print(f"Saved {len(df_rate_c)} rows to {cache_file}")
else:
    df_rate_c = pd.read_parquet(cache_file)
    print(f"Loaded {len(df_rate_c)} rows from cache")

df_rate_c['day'] = pd.to_datetime(df_rate_c['day'])
df_rate_c = df_rate_c.set_index('day').sort_index()

fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot(
    df_rate_c.index,
    df_rate_c['cnt'],
    color='#1f77b4',
    linewidth=1.5,
    alpha=1.0,
    rasterized=True,
)

ax.set_ylabel('Requests')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n'%y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

style_grid(ax)

fig.tight_layout(pad=0.4)
plt.savefig(out_dir / 'fig_request_rate_day.pdf', bbox_inches='tight', dpi=300)
print(f"saved {out_dir / 'fig_request_rate_day.pdf'}")
plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_token_rate_day.pdf
# Daily total token volume (input + output tokens summed per day) — panel (b),
# "Daily Token Rate". Only requests with both `started_at` and `completed_at`
# are counted, since token counts are recorded at completion.
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: Global token rate (daily, input + output) ──

cached = CACHED
cache_file = CACHE_DIR / 'token_rate_daily.parquet'
out_dir = PDF_DIR

if not cached or not cache_file.exists():
    df_tok_rate = con.sql(f"""
        SELECT
            DATE_TRUNC('day', started_at)::DATE AS day,
            SUM(COALESCE(it, 0) + COALESCE(ot, 0)) AS tokens
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL
          AND completed_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    df_tok_rate.to_parquet(cache_file, engine="fastparquet")
    print(f"Saved {len(df_tok_rate)} rows to {cache_file}")
else:
    df_tok_rate = pd.read_parquet(cache_file)
    print(f"Loaded {len(df_tok_rate)} rows from cache")

df_tok_rate['day'] = pd.to_datetime(df_tok_rate['day'])
df_tok_rate = df_tok_rate.set_index('day').sort_index()

# ── Plot ──
fig, ax = plt.subplots(figsize=(3.5, 3.0))

ax.plot(
    df_tok_rate.index,
    df_tok_rate['tokens'],
    color='#1f77b4',
    linewidth=1.5,
    alpha=1.0,
    rasterized=True,
)

ax.set_ylabel('Total Tokens')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e9:.0f}B'))

# X-axis formatting
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n'%y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

style_grid(ax)

fig.tight_layout(pad=0.4)
plt.savefig(out_dir / 'fig_token_rate_day.pdf', bbox_inches='tight', dpi=300)
print(f"saved {out_dir / 'fig_token_rate_day.pdf'}")
plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_daily_unique_models.pdf
# Number of distinct models (identified by `chute_id`) that received traffic
# each day — panel (c), "Active models per day". The same daily aggregation
# query also counts distinct users, so this block additionally saves
# fig_daily_unique_users.pdf as a byproduct (kept faithful to the source
# notebook).
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: Daily unique users and models over time ──

cached = CACHED

cache_file = CACHE_DIR / "arrival_daily_unique_users_models.parquet"
out_dir = PDF_DIR

MODEL_COL = "chute_id"   # change to "model_name" if your table has model_name

if not cached or not cache_file.exists():
    df_daily_unique = con.sql(f"""
        SELECT
            DATE_TRUNC('day', started_at)::DATE AS day,
            COUNT(DISTINCT user_id) AS unique_users,
            COUNT(DISTINCT {MODEL_COL}) AS unique_models
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    df_daily_unique.to_parquet(cache_file, engine="fastparquet")
    print(f"Saved {len(df_daily_unique):,} rows to {cache_file}")
else:
    df_daily_unique = pd.read_parquet(cache_file)
    print(f"Loaded {len(df_daily_unique):,} rows from {cache_file}")

df_daily_unique["day"] = pd.to_datetime(df_daily_unique["day"])
df_daily_unique = df_daily_unique.set_index("day").sort_index()

# ── Plot: Daily unique users ──
fig, ax = plt.subplots(figsize=(3.5, 3.0))

ax.plot(
    df_daily_unique.index,
    df_daily_unique["unique_users"],
    linewidth=1.5,
    alpha=1.0,
    rasterized=True,
)

ax.set_ylabel("Unique users")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n'%y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

style_grid(ax)

fig.tight_layout(pad=0.4)
fig.savefig(out_dir / "fig_daily_unique_users.pdf", bbox_inches="tight", dpi=300)
print(f"saved {out_dir / 'fig_daily_unique_users.pdf'}")
plt.close(fig)


# ── Plot: Daily unique models ──
fig, ax = plt.subplots(figsize=(3.5, 3.0))

ax.plot(
    df_daily_unique.index,
    df_daily_unique["unique_models"],
    linewidth=1.5,
    alpha=1.0,
    rasterized=True,
)

ax.set_ylabel("Unique models")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n'%y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

style_grid(ax)

fig.tight_layout(pad=0.4)
fig.savefig(out_dir / "fig_daily_unique_models.pdf", bbox_inches="tight", dpi=300)
print(f"saved {out_dir / 'fig_daily_unique_models.pdf'}")
plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_function_name_daily_requests.pdf
# Daily request rate broken down by API function name (Chat, Chat Stream,
# Completion, Completion Stream, Other) — panel (d), "Daily API calls".
# Also renders the standalone legend fig_function_name_daily_requests_legend.pdf
# (used in the appendix). First the shared function-name ordering/labels/colors
# (copied from the source notebook's helper cell), then the figure.
# ────────────────────────────────────────────────────────────────────────────

# ── Helper constants: function-name categories (shared ordering/labels/colors) ──
# Full 5-category set (used for the request-rate plot only)
FUNCTION_ORDER = ['chat','chat_stream', 'completion', 'completion_stream', 'other']
FUNCTION_LABELS = {
    'completion_stream': 'Completion Stream',
    'chat_stream': 'Chat Stream',
    'chat': 'Chat',
    'completion': 'Completion',
    'other': 'Other',
}
FUNCTION_COLORS = {
    'completion_stream': '#1f77b4',
    'chat_stream': '#ff7f0e',
    'chat': '#2ca02c',
    'completion': '#d62728',
    'other': '#7f7f7f',
}

# ── Figure: Daily request rate by function name (5 categories, request rate keeps Other) ──

cached = CACHED
cache_file = CACHE_DIR / 'function_name_daily_requests.parquet'
out_dir = PDF_DIR

if not cached or not cache_file.exists():
    df_func_daily = con.sql(f"""
        SELECT
            DATE_TRUNC('day', started_at)::DATE AS day,
            CASE
                WHEN function_name IN ('completion_stream', 'chat_stream', 'chat', 'completion')
                    THEN function_name
                ELSE 'other'
            END AS function_group,
            COUNT(*) AS requests
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).fetchdf()
    df_func_daily.to_parquet(cache_file)
    print(f"Saved {len(df_func_daily):,} rows to {cache_file}")
else:
    df_func_daily = pd.read_parquet(cache_file)
    print(f"Loaded {len(df_func_daily):,} rows from {cache_file}")

df_func_daily['day'] = pd.to_datetime(df_func_daily['day'])
pivot = (
    df_func_daily
    .pivot_table(index='day', columns='function_group', values='requests', fill_value=0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(3.5, 3.0))

legend_handles = []
legend_labels = []
for func in FUNCTION_ORDER:
    if func not in pivot.columns:
        continue
    line, = ax.plot(
        pivot.index,
        pivot[func],
        linewidth=1.4 if func != 'other' else 1.1,
        alpha=0.95 if func != 'other' else 0.75,
        color=FUNCTION_COLORS[func],
        label=FUNCTION_LABELS[func],
        rasterized=True,
    )
    legend_handles.append(line)
    legend_labels.append(FUNCTION_LABELS[func])
# legend handles start from chat, chat_stream, completion, completion_stream, other (in FUNCTION_ORDER)

ax.set_ylabel('Requests')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x < 1e7 else f'{x/1e6:.0f}M'))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n'%y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
# set legend
ax.legend(loc='upper left', frameon=False, ncol=2, columnspacing=1.0, handlelength=1.5, fontsize=8)
ax.tick_params(axis='both', which='major', length=4.5, width=0.8)
ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)
ax.set_ylim(0, 23e6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

style_grid(ax)

fig.tight_layout(pad=0.4)
fig.savefig(out_dir / 'fig_function_name_daily_requests.pdf', bbox_inches='tight', dpi=300)
print(f"saved {out_dir / 'fig_function_name_daily_requests.pdf'}")
plt.close(fig)

# Save legend separately
save_legend(
    legend_handles, legend_labels,
    out_dir / 'fig_function_name_daily_requests_legend.pdf',
    ncol=len(legend_labels),
)
print(f"saved {out_dir / 'fig_function_name_daily_requests_legend.pdf'}")


# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_seasonality.pdf
# Hour-of-day x day-of-week heatmap of average request volume — panel (e),
# "Request heatmap". Each (day-of-week, hour) cell is normalized by how many
# times that day-of-week occurs in the trace's date range, so uneven weekday
# counts do not inflate rows. Depends on `df_rate_c` from the
# fig_request_rate_day.pdf block above (per-day-of-week normalization).
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: Seasonality heatmap — consistent with daily request rate ──
# Two fixes for consistency with the daily request-rate figure above:
#   1) Drops the `it IS NOT NULL` filter so both figures count the same
#      population of requests.
#   2) Normalizes each (dow, hour) cell by the number of times that
#      day-of-week appears in the date range, so an unequal number of
#      weekdays (e.g. 53 Fri vs 52 Mon when the span is 52w + 3d) does
#      not visually inflate certain rows.

cached = CACHED
cache_file = CACHE_DIR / 'seasonality_consistent.parquet'
out_dir = PDF_DIR

if not cached or not cache_file.exists():
    df_season_c = con.sql(f"""
        SELECT
            dayofweek(started_at)         AS dow,
            EXTRACT(HOUR FROM started_at) AS hour,
            COUNT(*)                      AS cnt
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL
        GROUP BY 1, 2
    """).fetchdf()
    df_season_c.to_parquet(cache_file)
    print(f"Saved {len(df_season_c)} rows to {cache_file}")
else:
    df_season_c = pd.read_parquet(cache_file)
    print(f"Loaded {len(df_season_c)} rows from cache")

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df_season_c['dow'] = df_season_c['dow'].astype(int)
df_season_c['hour'] = df_season_c['hour'].astype(int)

# Count occurrences of each dow in the daily series (from the block above)
# so we can convert raw totals into an average per-day rate.
days_per_dow = (
    df_rate_c.index.to_series().dt.dayofweek.value_counts().reindex(range(7)).astype(int)
)
print('Days per day-of-week in range:', days_per_dow.to_dict())

heatmap = df_season_c.pivot_table(
    index='dow', columns='hour', values='cnt', fill_value=0
)
heatmap = heatmap.reindex(range(7)).reindex(range(24), axis=1)

# Average requests per (dow, hour) — i.e., divide by number of times
# that dow appears in the range.
heatmap_avg = heatmap.div(days_per_dow, axis=0)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.imshow(
    heatmap_avg.values / 1e6, aspect='auto', cmap='YlOrRd',
    interpolation='nearest',
)
ax.set_yticks(range(7))
ax.set_yticklabels(day_labels)
ax.set_xlabel('Hour of Day (UTC)', size=10)
ax.set_xticks(range(0, 24, 4))
ax.set_xticklabels(range(0, 24, 4))

cbar = fig.colorbar(im, ax=ax, shrink=1, pad=0.1, location='top', orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Avg requests / week (M)', size=11)

fig.tight_layout()
plt.savefig(out_dir / 'fig_seasonality.pdf', bbox_inches='tight', dpi=300)
print(f"saved {out_dir / 'fig_seasonality.pdf'}")
plt.close(fig)

# Sanity check: total of (heatmap_avg * days_per_dow) should equal the
# daily-rate total.
recon_total = heatmap.values.sum()
daily_total = df_rate_c['cnt'].sum()
print(f"heatmap total      = {recon_total:,}")
print(f"daily-rate total   = {daily_total:,}")
print(f"diff               = {recon_total - daily_total:,}")
