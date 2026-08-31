#!/usr/bin/env python3
"""Section 5 — Workload Evolution: Figure Reproduction.

Regenerates every figure included in the paper's Section 5 ("Workload
Evolution") from the anonymized one-year trace (DuckDB).

Figures produced (written to figures/paper/03_workload_evolution/):
  fig_top10_model_monthly_request_share.pdf  Stacked monthly share of requests for the
                                             top-10 (TEE-merged) models vs. Others.
  fig_top10_model_monthly_token_share.pdf    Stacked monthly share of total tokens
                                             (input + output) for the same top-10 models.
  fig_top10_model_monthly_share_legend.pdf   Standalone legend shared by the two share figures.
  fig_monthly_input_token_intervals.pdf      Monthly input-token percentile bands
                                             (P10-P90, P25-P75, median) across all requests.
  fig_monthly_output_token_intervals.pdf     Monthly output-token percentile bands
                                             across all requests.
  fig_user_release_cohort_input_band.pdf     Per-user median input tokens, bucketed by the
                                             user's first-active month (cohort), with
                                             cross-user percentile bands.
  fig_user_release_cohort_output_band.pdf    Per-user median output tokens for the same
                                             user cohorts.

Each heavy DuckDB query is guarded by a cache toggle: the first run computes
the aggregate and writes a parquet file under cache/; subsequent runs reuse
the cache and just replot (pass --recompute to force a rerun of the queries).
Requirements: the trace database (--db) and the bundled chute_models.csv
(chute_id -> model name mapping) at the repo root.

Usage:
    python 03_section5_workload_evolution.py [--db PATH] [--recompute]
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import colorsys
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import (
    LogLocator,
    LogFormatterMathtext,
    NullFormatter,
    AutoMinorLocator,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import CHUTE_MODELS_CSV, DB_DEFAULT, section_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the anonymized trace "
                         "(scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true", help="rerun heavy DuckDB queries even if cache files exist")
args = parser.parse_args()
DB_PATH = args.db
CACHED = not args.recompute   # True: reuse cache files and just replot

# ────────────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings('ignore')

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
sns.set_palette('tab10')

GRID_MAJOR_KW = dict(linestyle='--', linewidth=0.6, alpha=0.35)
GRID_MINOR_KW = dict(linestyle='--', linewidth=0.4, alpha=0.12)


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


PDF_DIR, CACHE_DIR = section_paths("03_workload_evolution")
PDF_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(exist_ok=True)
con = duckdb.connect(DB_PATH, read_only=True)

TABLE_NAME = 'all_metrics_user'
row_count = con.sql(f'SELECT COUNT(*) AS n FROM {TABLE_NAME}').fetchone()[0]
print(f'Connected to DuckDB — {row_count:,} rows in {TABLE_NAME}')

# chute_id → human-readable model name mapping (ships alongside this script)
chute_df = pd.read_csv(CHUTE_MODELS_CSV)
CHUTE_TO_MODEL = dict(zip(chute_df['chute_id'], chute_df['name']))
print(f'Loaded {len(CHUTE_TO_MODEL):,} chute_id → model name mappings')

# ────────────────────────────────────────────────────────────────────────────
# Figures: fig_top10_model_monthly_request_share.pdf,
#          fig_top10_model_monthly_token_share.pdf,
#          fig_top10_model_monthly_share_legend.pdf
#
# Stacked monthly bars showing what fraction of all requests (and of all
# input + output tokens) each of the globally top-10 models accounts for,
# with every remaining model folded into a gray "Others" bar. Models that
# have a -TEE variant are merged into their non-TEE counterpart so each
# model family occupies a single bar segment. The legend is emitted as a
# standalone PDF shared by both share figures.
# ────────────────────────────────────────────────────────────────────────────

# ── Variant: Top 10 models (TEE-merged) — stacked monthly share of requests AND tokens ──
# Models that have a `-TEE` variant are folded into their non-TEE counterpart so
# the family occupies a single bar — frees up space in the top-10 for other models.
# Colors are derived from a prefix-match clustering (transitive closure on
# pairwise common-prefix length) so same-family models share a hue ramp.

# ── X-axis tick configuration ──
TICK_INTERVAL_MONTHS = 3
TICK_FORMAT = "%b\n'%y"   # e.g., "Aug\n'25"
FAMILY_PREFIX_MIN = 3   # shared display-label prefix > FAMILY_PREFIX_MIN ⇒ same family

cached = CACHED
raw_cache_file = CACHE_DIR / 'section3_model_monthly_requests_tokens_for_share.parquet'
top10_cache_file = CACHE_DIR / 'section3_top10_named_model_tee_merged_share.parquet'
if cached and top10_cache_file.exists():
    df_top10_share = pd.read_parquet(top10_cache_file)
    print(f'Loaded {len(df_top10_share):,} model-month rows from {top10_cache_file}')
else:
    if cached and raw_cache_file.exists():
        df_month_model = pd.read_parquet(raw_cache_file)
        print(f'Loaded {len(df_month_model):,} raw model-month rows from {raw_cache_file}')
    else:
        df_month_model = con.sql(f"""
            SELECT
                date_trunc('month', started_at)::TIMESTAMP AS month,
                chute_id,
                COUNT(*) AS model_requests,
                SUM(COALESCE(it, 0) + COALESCE(ot, 0)) AS model_tokens
            FROM {TABLE_NAME}
            WHERE chute_id IS NOT NULL AND started_at IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 3 DESC
        """).fetchdf()
        df_month_model.to_parquet(raw_cache_file)
        print(f'Saved {len(df_month_model):,} raw model-month rows to {raw_cache_file}')

    df_month_model['model_name_raw'] = df_month_model['chute_id'].map(CHUTE_TO_MODEL)
    # Strip trailing -TEE (case-insensitive) so deepseek-ai/X and deepseek-ai/X-TEE collapse
    df_month_model['model_name'] = (
        df_month_model['model_name_raw'].str.replace(r'-TEE$', '', regex=True, case=False)
    )

    # Monthly totals (denominator) come from the full trace, named or not
    monthly_total = (
        df_month_model.groupby('month', as_index=False)
        .agg(total_requests=('model_requests', 'sum'),
             total_tokens=('model_tokens', 'sum'))
    )

    # Re-aggregate after TEE-merge across (month, model_name)
    df_month_model_merged = (
        df_month_model[df_month_model['model_name'].notna()]
        .groupby(['month', 'model_name'], as_index=False)
        .agg(model_requests=('model_requests', 'sum'),
             model_tokens=('model_tokens', 'sum'))
    )

    global_named_totals = (
        df_month_model_merged
        .groupby('model_name', as_index=False)
        .agg(model_requests=('model_requests', 'sum'),
             model_tokens=('model_tokens', 'sum'))
        .sort_values('model_requests', ascending=False)
    )

    top10_names = global_named_totals['model_name'].head(10).tolist()
    print('Top 10 (TEE-merged) models globally by requests:\n  '
          + '\n  '.join(top10_names))

    top10_rows = df_month_model_merged[
        df_month_model_merged['model_name'].isin(top10_names)
    ].copy()
    top10_rows = top10_rows.merge(monthly_total, on='month', how='left')
    top10_rows['request_share'] = top10_rows['model_requests'] / top10_rows['total_requests']
    top10_rows['token_share']   = top10_rows['model_tokens']   / top10_rows['total_tokens']

    top10_sum = (
        top10_rows.groupby('month', as_index=False)
        .agg(top_requests=('model_requests', 'sum'),
             top_tokens=('model_tokens', 'sum'))
    )

    others = monthly_total.merge(top10_sum, on='month', how='left')
    others[['top_requests', 'top_tokens']] = others[['top_requests', 'top_tokens']].fillna(0)
    others['model_requests'] = others['total_requests'] - others['top_requests']
    others['model_tokens']   = others['total_tokens']   - others['top_tokens']
    others['request_share']  = others['model_requests'] / others['total_requests']
    others['token_share']    = others['model_tokens']   / others['total_tokens']
    others['model_name'] = 'Others'

    keep_cols = [
        'month', 'model_name',
        'model_requests', 'model_tokens',
        'total_requests', 'total_tokens',
        'request_share', 'token_share',
    ]
    top10_rows = top10_rows[keep_cols]
    others = others[keep_cols]
    df_top10_share = pd.concat([top10_rows, others], ignore_index=True)

    name_order = top10_names + ['Others']
    df_top10_share['model_order'] = df_top10_share['model_name'].map(
        {n: i for i, n in enumerate(name_order)}
    )
    df_top10_share = (
        df_top10_share.sort_values(['month', 'model_order']).reset_index(drop=True)
    )
    df_top10_share.to_parquet(top10_cache_file)
    print(f'Saved {len(df_top10_share):,} model-month rows to {top10_cache_file}')


# ── Shared plotting setup ──

months = pd.to_datetime(sorted(df_top10_share['month'].unique()))
bar_width = 24

global_totals = (
    df_top10_share[df_top10_share['model_name'] != 'Others']
    .groupby('model_name', as_index=False)['model_requests']
    .sum()
    .sort_values('model_requests', ascending=False)
)
model_order = global_totals['model_name'].tolist()
plot_order = model_order + ['Others']

label_map = {name: str(name).split('/')[-1] for name in model_order}
label_map['Others'] = 'Others'


def assign_family_colors(names, label_map, prefix_min=3):
    """Plain palette: tab20 with grayish entries removed (gray is reserved for
    Others). Colors are assigned in the order `names` is passed in, so callers
    control ordering by sorting upstream."""
    base_colors = list(plt.cm.tab20.colors)

    def is_grayish(color, sat_threshold=0.22):
        r, g, b = color[:3]
        _, _, s = colorsys.rgb_to_hls(r, g, b)
        return s < sat_threshold

    palette = [c for c in base_colors if not is_grayish(c)]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    # fam_members kept for the debug print downstream — one bucket per model
    fam_members = {i: [i] for i in range(len(names))}
    return colors, fam_members


# Order labels (descending by global requests) before assigning colors
model_order = sorted(model_order,
                     key=lambda n: -global_totals.set_index("model_name").loc[n, "model_requests"])
plot_order = model_order + ["Others"]

color_map, fam_members = assign_family_colors(
    model_order, label_map, prefix_min=FAMILY_PREFIX_MIN,
)
color_map['Others'] = '#d9d9d9'

# Quick debug print so you can see how models were grouped
print('Family clusters (display labels):')
for fid, members in fam_members.items():
    print(f'  family {fid}: ' + ', '.join(label_map[model_order[i]] for i in members))


def _plot_share(share_col, ylabel, fname):
    plot_df = (
        df_top10_share
        .groupby(['month', 'model_name'], as_index=False)[share_col]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    bottom = np.zeros(len(months))

    for model_name in plot_order:
        sub = (
            plot_df[plot_df['model_name'] == model_name]
            .set_index('month')
            .reindex(months)
        )
        values = sub[share_col].fillna(0).to_numpy(dtype=float)
        if not np.any(values > 0):
            continue
        ax.bar(
            months, values, bottom=bottom, width=bar_width,
            color=color_map[model_name], label=label_map[model_name], linewidth=0,
        )
        bottom += values

    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=TICK_INTERVAL_MONTHS))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(TICK_FORMAT))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(True, which='major', axis='y', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', axis='y', **GRID_MINOR_KW)

    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)

    ax.set_xlim(
        months.min() - pd.Timedelta(days=15),
        months.max() + pd.Timedelta(days=15),
    )

    finish_fig(fig, fname)


_plot_share('request_share', 'Share of requests',
            'fig_top10_model_monthly_request_share.pdf')
_plot_share('token_share',   'Share of total tokens',
            'fig_top10_model_monthly_token_share.pdf')

# ── Standalone legend (no size constraint) ──

legend_handles = [Patch(facecolor=color_map[name], label=label_map[name])
                  for name in plot_order]

n_items = len(legend_handles)
legend_ncol = min(n_items, 3)
legend_nrow = int(np.ceil(n_items / legend_ncol))

# Roomy size — legend isn't constrained by the bar-figure footprint
fig_l = plt.figure(figsize=(1.6 * legend_ncol + 0.4, 0.35 * legend_nrow + 0.4))
fig_l.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc='center',
    ncol=legend_ncol,
    frameon=True,
    columnspacing=1.4,
    handlelength=1.6,
    handletextpad=0.6,
    fontsize=9,
)
fig_l.savefig(
    PDF_DIR / 'fig_top10_model_monthly_share_legend.pdf',
    bbox_inches='tight', dpi=300,
)
print(f"Saved {PDF_DIR / 'fig_top10_model_monthly_share_legend.pdf'}")
plt.close(fig_l)

# ────────────────────────────────────────────────────────────────────────────
# Figures: fig_monthly_input_token_intervals.pdf,
#          fig_monthly_output_token_intervals.pdf
#
# Monthly percentile bands of per-request input and output token counts
# across the whole trace (requests with positive input and output tokens):
# the median line with shaded P25-P75 and P10-P90 intervals, on a log
# y-axis. These show how request sizes drift over the year.
# ────────────────────────────────────────────────────────────────────────────

# ── Figures: Monthly input and output token shaded interval lineplots ──
cached = CACHED
month_cache_file = CACHE_DIR / 'section3_overall_monthly_input_output_token_intervals.parquet'

# ── Monthly ──
if cached and month_cache_file.exists():
    df_monthly_tokens = pd.read_parquet(month_cache_file)
    print(f'Loaded {len(df_monthly_tokens):,} month rows from {month_cache_file}')
else:
    df_monthly_tokens = con.sql(f"""
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
    """).fetchdf()
    df_monthly_tokens.to_parquet(month_cache_file)
    print(f'Saved {len(df_monthly_tokens):,} month rows to {month_cache_file}')


def plot_monthly_token_interval(prefix, ylabel, color, filename):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.fill_between(df_monthly_tokens['month'], df_monthly_tokens[f'{prefix}_p10'], df_monthly_tokens[f'{prefix}_p90'],
                    color=color, alpha=0.10, linewidth=0, label='P10-P90')
    ax.fill_between(df_monthly_tokens['month'], df_monthly_tokens[f'{prefix}_p25'], df_monthly_tokens[f'{prefix}_p75'],
                    color=color, alpha=0.22, linewidth=0, label='P25-P75')
    ax.plot(df_monthly_tokens['month'], df_monthly_tokens[f'{prefix}_p50'],
            color=color, linewidth=1.6, label='Median')
    ax.set_xlabel('Month')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b \n'%y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    add_log_y_grid(ax)
    ax.grid(True, which='major', axis='x', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', axis='x', **GRID_MINOR_KW)
    # ax.tick_params(axis='x', rotation=45, which='both')
    ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=3)
    finish_fig(fig, filename)


plot_monthly_token_interval('it', 'Input tokens', '#1f77b4', 'fig_monthly_input_token_intervals.pdf')
plot_monthly_token_interval('ot', 'Output tokens', '#ff7f0e', 'fig_monthly_output_token_intervals.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Figures: fig_user_release_cohort_input_band.pdf,
#          fig_user_release_cohort_output_band.pdf
#
# Each user is assigned to a release cohort — the month of their first
# request — and summarized by their median input / output token counts
# (users with more than 10 valid-token requests and medians above 1). Per
# cohort, the plots show the cross-user median with P25-P75 and P10-P90
# bands, revealing how newer user cohorts differ in request shape. The full
# per-user GROUP BY is too heavy in one pass, so the query batches users
# (top 10 individually, the rest in chunks).
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: User release cohort lineplot for median input/output tokens ──
# Each point is one user, bucketed by the month of their first request.
# Filters: user has > 10 valid-token requests, median input > 1, median output > 1.
# Single GROUP BY user is too heavy at full scale, so batch: top 10 users 1 by 1,
# then the rest in chunks of 2000 (same pattern as the model-access query above).
USER_BATCH_SIZE = 20000
USER_TOP_K = 10
# A user's release cohort is a property of the person. On the anonymized public
# trace user_id rotates every 3 months, so MIN(month) per user_id would give the
# month that *id* appeared, not the person; that trace ships the true value as a
# release_month column instead. Falls back to computing it when absent.
_COLS = {row[0] for row in con.sql(f"DESCRIBE SELECT * FROM {TABLE_NAME}").fetchall()}
COHORT_EXPR = ("MIN(m.release_cohort)" if "release_cohort" in _COLS
               else "MIN(date_diff('month', TIMESTAMP '1970-01-01', m.started_at) + 1)")
print(f"release cohort from: {'shipped release_cohort column' if 'release_cohort' in _COLS else 'MIN(started_at) per user'}")
cached = CACHED
cache_file = CACHE_DIR / 'section3_user_release_cohort_token_medians.parquet'
rank_cache_file = CACHE_DIR / 'section3_user_release_cohort_user_rank.parquet'

if cached and cache_file.exists():
    df_user_release_tokens = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_user_release_tokens):,} user rows from {cache_file}')
else:
    if cached and rank_cache_file.exists():
        user_rank = pd.read_parquet(rank_cache_file)
        print(f'Loaded user ranks from {rank_cache_file}')
    else:
        user_rank = con.sql(f"""
            SELECT user_id, COUNT(*) AS n_requests
            FROM {TABLE_NAME}
            WHERE user_id IS NOT NULL
              AND started_at IS NOT NULL
              AND it IS NOT NULL AND it > 0
              AND ot IS NOT NULL AND ot > 0
            GROUP BY user_id
            ORDER BY n_requests DESC
        """).fetchdf()
        user_rank.to_parquet(rank_cache_file)
        print(f'Saved {len(user_rank):,} user ranks to {rank_cache_file}')

    top_users = user_rank['user_id'].head(USER_TOP_K).tolist()
    rest_users = user_rank['user_id'].iloc[USER_TOP_K:].tolist()
    parts = []

    def query_user_subset(user_ids, label):
        if not user_ids:
            return
        rel_name = 'section3_user_subset'
        con.register(rel_name, pd.DataFrame({'user_id': user_ids}))
        try:
            part = con.sql(f"""
                SELECT
                    m.user_id,
                    {COHORT_EXPR} AS release_month,
                    COUNT(*) AS n_requests,
                    approx_quantile(m.it, 0.5) AS input_median,
                    approx_quantile(m.ot, 0.5) AS output_median
                FROM {TABLE_NAME} m
                JOIN {rel_name} s USING (user_id)
                WHERE m.user_id IS NOT NULL
                  AND m.started_at IS NOT NULL
                  AND m.it IS NOT NULL AND m.it > 0
                  AND m.ot IS NOT NULL AND m.ot > 0
                GROUP BY m.user_id
            """).fetchdf()
        finally:
            con.unregister(rel_name)
        parts.append(part)
        print(f'  {label}: {len(user_ids):,} users -> {len(part):,} rows')

    for idx, user_id in enumerate(top_users, 1):
        query_user_subset([user_id], f'top user {idx}/{USER_TOP_K}')

    n_batches = (len(rest_users) + USER_BATCH_SIZE - 1) // USER_BATCH_SIZE
    for b in range(n_batches):
        chunk = rest_users[b * USER_BATCH_SIZE:(b + 1) * USER_BATCH_SIZE]
        query_user_subset(chunk, f'rest batch {b + 1}/{n_batches}')

    df_user_release_tokens = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=['user_id', 'release_month', 'n_requests', 'input_median', 'output_median'])
    )
    df_user_release_tokens = df_user_release_tokens[
        (df_user_release_tokens['n_requests'] > 10)
        & (df_user_release_tokens['input_median'] > 1)
        & (df_user_release_tokens['output_median'] > 1)
    ].reset_index(drop=True)
    df_user_release_tokens.to_parquet(cache_file)
    print(f'Saved {len(df_user_release_tokens):,} user rows to {cache_file}')

df_user_release_tokens['release_month'] = pd.to_datetime(df_user_release_tokens['release_month'])

# Aggregate cross-user percentiles per release cohort.
agg_rows = []
for cohort, sub in df_user_release_tokens.groupby('release_month'):
    rec = {'release_month': cohort, 'n_users': len(sub)}
    for col in ('input_median', 'output_median'):
        vals = sub[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        rec[f'{col}_p10'] = np.quantile(vals, 0.10)
        rec[f'{col}_p25'] = np.quantile(vals, 0.25)
        rec[f'{col}_p50'] = np.quantile(vals, 0.50)
        rec[f'{col}_p75'] = np.quantile(vals, 0.75)
        rec[f'{col}_p90'] = np.quantile(vals, 0.90)
    agg_rows.append(rec)

df_user_cohort_bands = pd.DataFrame(agg_rows).sort_values('release_month').reset_index(drop=True)


def plot_user_cohort_band(prefix, ylabel, color, filename):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    x = df_user_cohort_bands['release_month']
    ax.fill_between(x, df_user_cohort_bands[f'{prefix}_p10'], df_user_cohort_bands[f'{prefix}_p90'],
                    color=color, alpha=0.10, linewidth=0, label='P10-P90')
    ax.fill_between(x, df_user_cohort_bands[f'{prefix}_p25'], df_user_cohort_bands[f'{prefix}_p75'],
                    color=color, alpha=0.22, linewidth=0, label='P25-P75')
    ax.plot(x, df_user_cohort_bands[f'{prefix}_p50'], color=color, linewidth=1.6, label='Median')
    ax.set_xlabel('Release cohort')
    ax.set_ylabel(ylabel)
    # ax.set_yscale('log')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b \n'%y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    # add_log_y_grid(ax)
    ax.grid(True, which='major', axis='x', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', axis='x', **GRID_MINOR_KW)
    # ax.tick_params(axis='x', rotation=45, which='both')
    ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=3)
    finish_fig(fig, filename)


plot_user_cohort_band(
    'input_median',
    'Median input',
    '#1f77b4',
    'fig_user_release_cohort_input_band.pdf',
)
plot_user_cohort_band(
    'output_median',
    'Median output',
    '#ff7f0e',
    'fig_user_release_cohort_output_band.pdf',
)

print('All Section 5 figures written to', PDF_DIR)
