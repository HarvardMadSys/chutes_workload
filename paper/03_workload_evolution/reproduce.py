#!/usr/bin/env python3
"""§4–§5 Workload evolution: how the model mix and the request shape change over the year.

Figures (written to figures/paper/03_workload_evolution/):
  fig_top10_model_monthly_request_share.pdf  Monthly share of requests of the 10 most-requested
                                             models (X and X-TEE merged); the rest is "Others".
  fig_top10_model_monthly_token_share.pdf    The same models' share of input + output tokens.
  fig_top10_model_monthly_share_legend.pdf   The legend of the two share figures.
  fig_monthly_input_token_intervals.pdf      Input tokens per request, by month: median,
                                             P25-P75 and P10-P90.
  fig_monthly_output_token_intervals.pdf     The same for output tokens.
  fig_user_release_cohort_input_band.pdf     Each user's median input tokens, grouped by the
                                             user's release cohort (first month): median,
                                             P25-P75 and P10-P90 over the users.
  fig_user_release_cohort_output_band.pdf    The same for output tokens.

Each trace query's result is cached as a parquet file in output/paper/cache/.
When every cache is there, the script only replots and never opens the trace;
--recompute reruns the queries. data/paper/chute_models.csv maps chute_id to
the model name.

Usage:
    python paper/03_workload_evolution/reproduce.py [--db PATH] [--recompute]
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import (AutoMinorLocator, LogFormatterMathtext, LogLocator, MultipleLocator,
                               NullFormatter)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import LazyDB, trace_month  # noqa: E402
from _paths import CHUTE_MODELS_CSV, DB_DEFAULT, section_paths  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the anonymized trace "
                         "(scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true", help="rerun heavy DuckDB queries even if cache files exist")
args = parser.parse_args()
DB_PATH = args.db
CACHED = not args.recompute   # True: reuse cache files and just replot

PDF_DIR, CACHE_DIR = section_paths("03_workload_evolution")
TABLE_NAME = 'all_metrics_user'
con = LazyDB(DB_PATH)   # opens the trace on first use, so a replot from cache never does

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
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

GRID_MAJOR_KW = dict(linestyle='--', linewidth=0.6, alpha=0.35)
GRID_MINOR_KW = dict(linestyle='--', linewidth=0.4, alpha=0.12)


def cached(name, compute):
    """compute() a DataFrame, or read the copy a previous run saved as CACHE_DIR/name."""
    path = CACHE_DIR / name
    if CACHED and path.exists():
        df = pd.read_parquet(path)
        print(f'Loaded {len(df):,} rows from {path}')
    else:
        df = compute()
        df.to_parquet(path)
        print(f'Saved {len(df):,} rows to {path}')
    return df


def cached_query(name, sql):
    """Run sql on the trace, or read the result a previous run saved as CACHE_DIR/name."""
    return cached(name, lambda: con.sql(sql).fetchdf())


def add_log_y_grid(ax, numticks=10):
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=numticks))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', **GRID_MINOR_KW)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)


def finish_fig(fig, filename):
    fig.tight_layout(pad=0.4)
    fig.savefig(PDF_DIR / filename, bbox_inches='tight', dpi=300)
    print(f'Saved {PDF_DIR / filename}')
    plt.close(fig)


def month_axis(ax, xlabel):
    """An x axis in months of the trace: labels at months 1, 4, 7, ..., a minor tick every month."""
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MultipleLocator(3, offset=1))
    ax.xaxis.set_minor_locator(MultipleLocator(1))


def plot_band(df, x, xlabel, prefix, ylabel, color, filename, log_y=False):
    """Median line df[prefix_p50] over df[x], with shaded P25-P75 and P10-P90 bands."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.fill_between(df[x], df[f'{prefix}_p10'], df[f'{prefix}_p90'],
                    color=color, alpha=0.10, linewidth=0, label='P10-P90')
    ax.fill_between(df[x], df[f'{prefix}_p25'], df[f'{prefix}_p75'],
                    color=color, alpha=0.22, linewidth=0, label='P25-P75')
    ax.plot(df[x], df[f'{prefix}_p50'], color=color, linewidth=1.6, label='Median')
    month_axis(ax, xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale('log')
        add_log_y_grid(ax)
    ax.grid(True, which='major', axis='x', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', axis='x', **GRID_MINOR_KW)
    ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=3)
    finish_fig(fig, filename)


# ────────────────────────────────────────────────────────────────────────────
# fig_top10_model_monthly_request_share.pdf, fig_top10_model_monthly_token_share.pdf,
# fig_top10_model_monthly_share_legend.pdf
# How each month's requests (and tokens) split across the 10 most-requested
# models and "Others", as stacked bars; the legend is a PDF of its own.
# ────────────────────────────────────────────────────────────────────────────

df_month_model = cached_query('monthly_requests_tokens_by_model.parquet', f"""
    SELECT
        date_trunc('month', started_at)::TIMESTAMP AS month,
        chute_id,
        COUNT(*) AS model_requests,
        SUM(COALESCE(it, 0) + COALESCE(ot, 0)) AS model_tokens
    FROM {TABLE_NAME}
    WHERE chute_id IS NOT NULL AND started_at IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 3 DESC
""")
df_month_model['month'] = trace_month(df_month_model['month'])   # month of the trace, 1..13

chute_models = pd.read_csv(CHUTE_MODELS_CSV)
CHUTE_TO_MODEL = dict(zip(chute_models['chute_id'], chute_models['name']))


def top10_share(df_month_model):
    """Each month's request and token share of the 10 most-requested models and of "Others".

    A model's -TEE chute counts as the model itself. A month's totals include
    every request, so chutes without a model name fall into "Others".
    Returns the table (one row per month and model) and the 10 names, most
    requested first.
    """
    df = df_month_model.assign(model_name=df_month_model['chute_id'].map(CHUTE_TO_MODEL)
                               .str.replace(r'-TEE$', '', regex=True, case=False))
    totals = df.groupby('month', as_index=False).agg(
        total_requests=('model_requests', 'sum'), total_tokens=('model_tokens', 'sum'))
    per_model = df[df['model_name'].notna()].groupby(['month', 'model_name'], as_index=False).agg(
        model_requests=('model_requests', 'sum'), model_tokens=('model_tokens', 'sum'))
    top10 = (per_model.groupby('model_name')['model_requests'].sum()
             .sort_values(ascending=False).index[:10].tolist())

    top = per_model[per_model['model_name'].isin(top10)].merge(totals, on='month', how='left')
    top_sum = top.groupby('month', as_index=False)[['model_requests', 'model_tokens']].sum()
    others = totals.merge(top_sum, on='month', how='left').fillna({'model_requests': 0, 'model_tokens': 0})
    others['model_requests'] = others['total_requests'] - others['model_requests']
    others['model_tokens'] = others['total_tokens'] - others['model_tokens']
    others['model_name'] = 'Others'

    share = pd.concat([top, others], ignore_index=True)
    share['request_share'] = share['model_requests'] / share['total_requests']
    share['token_share'] = share['model_tokens'] / share['total_tokens']
    return share, top10


share, model_order = top10_share(df_month_model)
print('Top 10 models by requests (X and X-TEE merged):\n  ' + '\n  '.join(model_order))
plot_order = model_order + ['Others']
labels = {name: name.split('/')[-1] for name in plot_order}
# tab20 without its two grays (saturation below 0.22): gray is for "Others"
palette = [c for c in plt.get_cmap('tab20').colors if colorsys.rgb_to_hls(*c)[2] >= 0.22]
colors = dict(zip(model_order, palette)) | {'Others': '#d9d9d9'}
months = np.sort(share['month'].unique())


def plot_share(share_col, ylabel, filename):
    """Stacked monthly bars of share[share_col], one segment per model in plot_order."""
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    bottom = np.zeros(len(months))
    for name in plot_order:
        values = (share[share['model_name'] == name].set_index('month')[share_col]
                  .reindex(months).fillna(0).to_numpy(dtype=float))
        ax.bar(months, values, bottom=bottom, width=0.8, color=colors[name], linewidth=0)
        bottom += values
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    month_axis(ax, 'Month')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', axis='y', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', axis='y', **GRID_MINOR_KW)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)
    ax.set_xlim(months.min() - 0.5, months.max() + 0.5)
    finish_fig(fig, filename)


def plot_share_legend(filename):
    handles = [Patch(facecolor=colors[name], label=labels[name]) for name in plot_order]
    ncol = min(len(handles), 3)
    nrow = int(np.ceil(len(handles) / ncol))
    fig = plt.figure(figsize=(1.6 * ncol + 0.4, 0.35 * nrow + 0.4))
    fig.legend(handles=handles, loc='center', ncol=ncol, frameon=True,
               columnspacing=1.4, handlelength=1.6, handletextpad=0.6, fontsize=9)
    fig.savefig(PDF_DIR / filename, bbox_inches='tight', dpi=300)
    print(f'Saved {PDF_DIR / filename}')
    plt.close(fig)


plot_share('request_share', 'Share of requests', 'fig_top10_model_monthly_request_share.pdf')
plot_share('token_share', 'Share of total tokens', 'fig_top10_model_monthly_token_share.pdf')
plot_share_legend('fig_top10_model_monthly_share_legend.pdf')

# ────────────────────────────────────────────────────────────────────────────
# fig_monthly_input_token_intervals.pdf, fig_monthly_output_token_intervals.pdf
# Input and output tokens per request (requests with both > 0), by month:
# median, P25-P75 and P10-P90, on a log axis.
# ────────────────────────────────────────────────────────────────────────────

df_monthly_tokens = cached_query('monthly_token_percentiles.parquet', f"""
    SELECT
        date_trunc('month', started_at)::TIMESTAMP AS month,
        COUNT(*) AS n_requests,
        approx_quantile(it, 0.10) AS it_p10,
        approx_quantile(it, 0.25) AS it_p25,
        approx_quantile(it, 0.50) AS it_p50,
        approx_quantile(it, 0.75) AS it_p75,
        approx_quantile(it, 0.90) AS it_p90,
        approx_quantile(ot, 0.10) AS ot_p10,
        approx_quantile(ot, 0.25) AS ot_p25,
        approx_quantile(ot, 0.50) AS ot_p50,
        approx_quantile(ot, 0.75) AS ot_p75,
        approx_quantile(ot, 0.90) AS ot_p90
    FROM {TABLE_NAME}
    WHERE started_at IS NOT NULL
      AND it IS NOT NULL AND it > 0
      AND ot IS NOT NULL AND ot > 0
    GROUP BY 1
    ORDER BY 1
""")
df_monthly_tokens['month'] = trace_month(df_monthly_tokens['month'])   # month of the trace, 1..13

plot_band(df_monthly_tokens, 'month', 'Month', 'it', 'Input tokens', '#1f77b4',
          'fig_monthly_input_token_intervals.pdf', log_y=True)
plot_band(df_monthly_tokens, 'month', 'Month', 'ot', 'Output tokens', '#ff7f0e',
          'fig_monthly_output_token_intervals.pdf', log_y=True)

# ────────────────────────────────────────────────────────────────────────────
# fig_user_release_cohort_input_band.pdf, fig_user_release_cohort_output_band.pdf
# Each user's median input (output) tokens, grouped by the user's release
# cohort: median, P25-P75 and P10-P90 over the users of each cohort. Users
# need more than 10 requests (with input and output tokens > 0) and medians
# above 1.
# ────────────────────────────────────────────────────────────────────────────

# release_cohort is the month of the trace (1..13) in which the person behind a
# user_id first appeared. The trace rotates user_id every 3 months, so
# MIN(started_at) per user_id would give the month the id appeared, not the
# person; the cohort therefore ships as a column of its own.
COHORT_EXPR = "MIN(m.release_cohort)"
USER_TOP_K = 10           # the heaviest users are queried one at a time,
USER_BATCH_SIZE = 20000   # the rest in batches of this many users


def user_token_medians():
    """Per user: release cohort, request count, median input and output tokens.

    One GROUP BY user_id over the whole trace is too heavy, so the query runs
    on the USER_TOP_K heaviest users one at a time, then on the others in
    batches of USER_BATCH_SIZE.
    """
    user_rank = cached_query('user_cohort_user_rank.parquet', f"""
        SELECT user_id, COUNT(*) AS n_requests
        FROM {TABLE_NAME}
        WHERE user_id IS NOT NULL
          AND started_at IS NOT NULL
          AND it IS NOT NULL AND it > 0
          AND ot IS NOT NULL AND ot > 0
        GROUP BY user_id
        ORDER BY n_requests DESC
    """)
    users = user_rank['user_id'].tolist()
    batches = ([[user] for user in users[:USER_TOP_K]]
               + [users[i:i + USER_BATCH_SIZE] for i in range(USER_TOP_K, len(users), USER_BATCH_SIZE)])
    parts = []
    for i, batch in enumerate(batches, 1):
        con.register('user_subset', pd.DataFrame({'user_id': batch}))
        try:
            parts.append(con.sql(f"""
                SELECT
                    m.user_id,
                    {COHORT_EXPR} AS release_month,
                    COUNT(*) AS n_requests,
                    approx_quantile(m.it, 0.5) AS input_median,
                    approx_quantile(m.ot, 0.5) AS output_median
                FROM {TABLE_NAME} m
                JOIN user_subset s USING (user_id)
                WHERE m.user_id IS NOT NULL
                  AND m.started_at IS NOT NULL
                  AND m.it IS NOT NULL AND m.it > 0
                  AND m.ot IS NOT NULL AND m.ot > 0
                GROUP BY m.user_id
            """).fetchdf())
        finally:
            con.unregister('user_subset')
        print(f'  batch {i}/{len(batches)}: {len(batch):,} users')
    df = pd.concat(parts, ignore_index=True)
    keep = (df['n_requests'] > 10) & (df['input_median'] > 1) & (df['output_median'] > 1)
    return df[keep].reset_index(drop=True)


def cohort_bands(df_user):
    """Per release cohort: P10, P25, P50, P75 and P90 over users of each user's median."""
    rows = []
    for cohort, sub in df_user.groupby('release_month'):
        row = {'release_month': cohort}
        for col in ('input_median', 'output_median'):
            for name, q in (('p10', 0.10), ('p25', 0.25), ('p50', 0.50), ('p75', 0.75), ('p90', 0.90)):
                row[f'{col}_{name}'] = np.quantile(sub[col].to_numpy(), q)
        rows.append(row)
    return pd.DataFrame(rows)


df_user = cached('user_cohort_token_medians.parquet', user_token_medians)
bands = cohort_bands(df_user)

plot_band(bands, 'release_month', 'Release cohort (month)', 'input_median', 'Median input', '#1f77b4',
          'fig_user_release_cohort_input_band.pdf')
plot_band(bands, 'release_month', 'Release cohort (month)', 'output_median', 'Median output', '#ff7f0e',
          'fig_user_release_cohort_output_band.pdf')
