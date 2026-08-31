#!/usr/bin/env python3
"""Section 4 — Users and Models: figure reproduction.

Regenerates every figure the paper's Section 4 ("Users and Models", VLDB
version) includes, from the anonymized one-year trace (DuckDB). The code is
the paper's own figure scripts — the model density heatmap, the user/model
hexbins and per-user access rasters, the popularity x burstiness density, and
the per-model IAT table the heatmap merges in — with only the paths adapted.

Figures (written to figures/paper/06_users_models/):
  requests_users_model_density.pdf              Models: requests vs. users — hexbin density (Fig. "Requests and model usage…", subfig (a), fig:volume_breadth_models)
  fig_user_requests_distinct_models_heatmap.pdf Users: requests vs. distinct models — hexbin density (Fig. "Requests and model usage…", subfig (b), fig:volume_breadth_users)
  fig_user_median_input_output_heatmap.pdf      Users: median input vs. median output tokens (Fig. "Most models and users have longer median inputs…", subfig (b), fig:token_shape_users)
  fig_model_median_input_output_heatmap.pdf     Models: median input vs. median output tokens (Fig. "Most models and users have longer median inputs…", subfig (a), fig:token_shape_models)
  periodic.pdf                                  30-day model-access raster, periodic high-volume user (Fig. "Thirty-day model-access rasters…", subfig (a), fig:user_access_persistent)
  user_raster_explore.pdf                       30-day model-access raster, model-exploring user (Fig. "Thirty-day model-access rasters…", subfig (b), fig:user_access_exploration)
  standalone_model_rank41_Qwen_Qwen3-Next-80B-A3B-Instruct_d166_d227_u10000bp_r10000bp_user_access_raster.pdf
                                                30-day user-access raster, Qwen/Qwen3-Next-80B-A3B-Instruct — periodic user access (Fig. "Thirty-day user-access rasters…", subfig (a), fig:model_access_power_user)
  openai_gpt-oss-20b_user_access_raster.pdf     30-day user-access raster, openai/gpt-oss-20b — correlated access (Fig. "Thirty-day user-access rasters…", subfig (b), fig:model_access_correlated)
  fig_model_heatmap_fixedxy_density.pdf         Model popularity vs. burstiness — hexbin density (Fig. "Model request load and burstiness", subfig (a), fig:model_burstiness_density)
  fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf
                                                Same axes, per-model scatter colored by lag-1 IAT autocorrelation (Fig. "Model request load and burstiness", subfig (b), fig:model_burstiness_autocorr)

Intermediate query results land in output/paper/cache/. Heavy DuckDB queries
are skipped on reruns when their cache file exists; pass --recompute to force
them to rerun against the trace.

Usage:
    python 06_section4_users_models.py [--db PATH] [--recompute]
"""

import matplotlib
matplotlib.use("Agg")  # headless — must precede any pyplot import

import argparse
import sys

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import re
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.ticker import (
    LogLocator,
    LogFormatterMathtext,
    NullFormatter,
    NullLocator,
    AutoMinorLocator,
    FuncFormatter,
)

warnings.filterwarnings('ignore')

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
#
# Imports, output/cache directories, and the read-only DuckDB connection.
# Plot style + grid helpers copied from the users-and-models figure script. chute_models.csv
# (shipped at the repo root) is the optional chute_id → model-name lookup used
# by the raster hover names; the figures still render when it is absent.
# ────────────────────────────────────────────────────────────────────────────

PDF_DIR, CACHE_DIR = section_paths("06_users_models")
PDF_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(exist_ok=True)
con = duckdb.connect(DB_PATH, read_only=True)
con.sql("PRAGMA threads=96;")
con.sql("PRAGMA memory_limit='1000GB'")
TABLE_NAME = "all_metrics_user"

row_count = con.sql(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}").fetchone()[0]
print(f"Connected to DuckDB — {row_count:,} rows in {TABLE_NAME}")

# Optional chute_id → model name lookup (ships at the repo root).
CHUTE_TO_MODEL = {}
try:
    chute_df = pd.read_csv(CHUTE_MODELS_CSV)
    CHUTE_TO_MODEL = dict(zip(chute_df["chute_id"], chute_df["name"]))
    print(f"Loaded {len(CHUTE_TO_MODEL):,} chute_id → model name mappings")
except FileNotFoundError:
    print("chute_models.csv not found — raster model names fall back to chute_ids")

# ── Plot style & grid helpers (from the users-and-models figure script) ──
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


def style_grid(ax):
    """Dashed grid on both major and minor axes, with major more visible than minor."""
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', **GRID_MINOR_KW)


def add_log_x_grid(ax, numticks=10):
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=numticks))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', **GRID_MAJOR_KW)
    ax.grid(True, which='minor', **GRID_MINOR_KW)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)


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
    print(f"Saved {PDF_DIR / filename}")
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure: requests_users_model_density.pdf
#
# Per-model hexbin on (requests × unique users), colored by model density
# (fig:volume_breadth_models — "Models: requests vs. users"). Plot copied from
# the global-characterization figure script cell 5 ("Figure 1: Requests × users, colored by model
# density"). Its two parquet inputs are built here when absent:
#   model_requests_vs_users.parquet  — the global-characterization figure script cell 4
#   iat_per_model.parquet            — the per-model IAT script cell 10 (batched per-model
#                                      IAT stats; ranks from model_popularity_rank)
# ────────────────────────────────────────────────────────────────────────────

# ── Data: per-model request count and unique users (from the global-characterization figure script cell 4) ──
cached = CACHED
cache_file = CACHE_DIR / 'model_requests_vs_users.parquet'

if not cached or not cache_file.exists():
    print('Computing per-chute request count and unique users …')
    df_mu = con.sql(f"""
        SELECT chute_id,
               COUNT(*)                AS n_requests,
               COUNT(DISTINCT user_id) AS n_users
        FROM {TABLE_NAME}
        WHERE chute_id IS NOT NULL AND user_id IS NOT NULL
        GROUP BY chute_id
    """).fetchdf()
    df_mu.to_parquet(cache_file)
    print(f'  saved {len(df_mu)} chutes → {cache_file}')
else:
    df_mu = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_mu)} chutes from cache')

# ── Data: per-model IAT statistics — variable-size batches (user_model cell 10) ──
# Top-ranked models have huge request counts; running their LAG window over the
# full table is memory-heavy, so we process them solo. Long-tail models are
# cheap, so we batch 100 at a time. Schedule:
#   ranks 1–5    → 1 chute / batch
#   ranks 6–20   → 2 chutes / batch
#   ranks 21+    → 100 chutes / batch
cached = CACHED
cache_file = CACHE_DIR / 'iat_per_model.parquet'
pop_cache_file = CACHE_DIR / 'model_popularity_rank.parquet'

if not cached or not cache_file.exists():
    # Rank chutes by total request count (query from user_model cell 7) — build if absent
    if not pop_cache_file.exists():
        print('Computing chute request counts …')
        df_pop = con.sql(f"""
            SELECT chute_id, COUNT(*) AS n_requests
            FROM {TABLE_NAME}
            WHERE chute_id IS NOT NULL
            GROUP BY chute_id
            ORDER BY n_requests DESC
        """).fetchdf()
        df_pop.to_parquet(pop_cache_file)
        print(f'  saved {len(df_pop)} chutes → {pop_cache_file}')
    else:
        df_pop = pd.read_parquet(pop_cache_file)
        print(f'Loaded {len(df_pop)} chutes from cache')

    df_pop = df_pop.sort_values('n_requests', ascending=False).reset_index(drop=True)
    chute_order = df_pop['chute_id'].tolist()

    # Build batch list per the schedule above
    batches = []
    for i in range(min(5, len(chute_order))):
        batches.append([chute_order[i]])
    i = 5
    while i < min(20, len(chute_order)):
        batches.append(chute_order[i:i+2])
        i += 2
    while i < len(chute_order):
        batches.append(chute_order[i:i+100])
        i += 100

    print(f'Computing IAT in {len(batches)} batches across {len(chute_order)} chutes …')

    results = []
    for bi, batch in enumerate(batches):
        ids_sql = "(" + ",".join(f"'{c}'" for c in batch) + ")"
        df_b = con.sql(f"""
            WITH iats AS (
                SELECT chute_id,
                       EXTRACT(EPOCH FROM (started_at - LAG(started_at) OVER (
                           PARTITION BY chute_id ORDER BY started_at
                       ))) AS iat_sec
                FROM {TABLE_NAME}
                WHERE chute_id IN {ids_sql}
                  AND started_at IS NOT NULL
            )
            SELECT chute_id,
                   COUNT(*)                     AS n_iats,
                   AVG(iat_sec)                 AS iat_mean,
                   QUANTILE_CONT(iat_sec, 0.5)  AS iat_median,
                   STDDEV(iat_sec)              AS iat_std
            FROM iats
            WHERE iat_sec IS NOT NULL AND iat_sec > 0
            GROUP BY chute_id
            HAVING COUNT(*) >= 100
        """).fetchdf()
        results.append(df_b)
        if bi < 25 or bi % 10 == 0 or bi == len(batches) - 1:
            print(f"  batch {bi+1}/{len(batches)} ({len(batch)} chutes) → {len(df_b)} rows")

    df_iat = pd.concat(results, ignore_index=True)
    # Burstiness B in [-1, 1]: -1 = perfectly periodic, 0 = Poisson, +1 = extreme bursts
    df_iat['burstiness'] = (df_iat['iat_std'] - df_iat['iat_mean']) / (df_iat['iat_std'] + df_iat['iat_mean'])
    # Coefficient of variation: <1 regular, =1 Poisson, >1 bursty
    df_iat['cv'] = df_iat['iat_std'] / df_iat['iat_mean']
    df_iat.to_parquet(cache_file)
    print(f'  saved {len(df_iat)} chutes → {cache_file}')
else:
    df_iat = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_iat)} chutes from cache')

# ── Figure 1: Requests × users, colored by model density (the global-characterization figure script cell 5) ──
df_mu = pd.read_parquet(CACHE_DIR / 'model_requests_vs_users.parquet')
df_iat = pd.read_parquet(CACHE_DIR / 'iat_per_model.parquet')

df = df_mu.merge(df_iat, on='chute_id')
df = df[
    (df['n_requests'] > 0) &
    (df['n_users'] > 0) &
    (df['iat_median'] > 0)
].copy()

fig, ax = plt.subplots(figsize=(3.8, 3.0))

hb_count = ax.hexbin(
    df['n_requests'],
    df['n_users'],
    xscale='log',
    yscale='log',
    gridsize=22,
    mincnt=1,
    cmap='viridis',
    norm=LogNorm(),
    linewidths=0
)

ax.set_xlabel('Requests per model')
ax.set_ylabel('Distinct users')
ax.set_ylim(1, df['n_users'].max() * 1.5)

cbar = plt.colorbar(hb_count, ax=ax, pad=0.02)
cbar.set_label('Models')

fig.tight_layout(pad=0.4)
plt.savefig(PDF_DIR / 'requests_users_model_density.pdf',
            bbox_inches='tight', dpi=300)
print(f"Saved {PDF_DIR / 'requests_users_model_density.pdf'}")
plt.close(fig)

print(f"Models shown: {len(df)}")

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_user_requests_distinct_models_heatmap.pdf
#
# Per-user hexbin on (requests × distinct models), colored by user count
# (fig:volume_breadth_users — "Users: requests vs. models"). Copied from
# the users-and-models figure script cell 2.
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: Per-user heatmap on (requests × distinct models), colored by user count ──
cached = CACHED
cache_file = CACHE_DIR / 'section3_user_requests_distinct_models_approx.parquet'

if cached and cache_file.exists():
    df_user_mix = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_user_mix):,} user rows from {cache_file}')
else:
    df_user_mix = con.sql(f"""
        SELECT
            user_id,
            COUNT(*) AS n_requests,
            approx_count_distinct(chute_id) AS n_models
        FROM {TABLE_NAME}
        WHERE user_id IS NOT NULL AND chute_id IS NOT NULL
        GROUP BY user_id
        HAVING COUNT(*) > 10 AND approx_count_distinct(chute_id) > 1
    """).fetchdf()
    df_user_mix.to_parquet(cache_file)
    print(f'Saved {len(df_user_mix):,} user rows to {cache_file}')

fig, ax = plt.subplots(figsize=(3.8, 3.0))
hb = ax.hexbin(
    df_user_mix['n_requests'], df_user_mix['n_models'],
    xscale='log', yscale='log', gridsize=24, mincnt=1,
    cmap='viridis', norm=LogNorm(), linewidths=0
)
ax.set_xlabel('Requests per user')
ax.set_ylabel('Distinct models')
add_log_x_grid(ax)
add_log_y_grid(ax)
cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label('Users')
finish_fig(fig, 'fig_user_requests_distinct_models_heatmap.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_user_median_input_output_heatmap.pdf
#
# Per-user hexbin on (median input tokens × median output tokens), colored by
# user count (fig:token_shape_users — "Users: input vs. output"). Copied from
# the users-and-models figure script cell 4; per-user medians are queried in rank-segmented
# batches (top-10 users solo, then 7,500-user batches).
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: Per-user median input × median output, colored by user count ──
cached = CACHED
cache_file = CACHE_DIR / 'section3_user_median_input_output_where_50k.parquet'
rank_cache_file = CACHE_DIR / 'section3_user_rank_for_segmented_queries.parquet'
USER_BATCH_SIZE = 7_500

if cached and cache_file.exists():
    df_user_tokens = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_user_tokens):,} user rows from {cache_file}')
else:
    if cached and rank_cache_file.exists():
        user_rank = pd.read_parquet(rank_cache_file)
        print(f'Loaded user ranks from {rank_cache_file}')
    else:
        user_rank = con.sql(f"""
            SELECT user_id, COUNT(*) AS n_requests
            FROM {TABLE_NAME}
            WHERE user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY n_requests DESC
        """).fetchdf()
        user_rank.to_parquet(rank_cache_file)
        print(f'Saved {len(user_rank):,} user ranks to {rank_cache_file}')

    def sql_in(values):
        return '(' + ', '.join("'" + str(v).replace("'", "''") + "'" for v in values) + ')'

    top_users = user_rank['user_id'].head(10).tolist()
    rest_users = user_rank['user_id'].iloc[10:].tolist()
    parts = []

    def query_user_subset(user_ids, label):
        if not user_ids:
            return
        part = con.sql(f"""
            SELECT
                user_id,
                COUNT(*) AS n_requests,
                approx_quantile(it, 0.5) AS input_median,
                approx_quantile(ot, 0.5) AS output_median
            FROM {TABLE_NAME}
            WHERE user_id IN {sql_in(user_ids)}
              AND it IS NOT NULL AND it > 0
              AND ot IS NOT NULL AND ot > 0
            GROUP BY user_id
            HAVING COUNT(*) > 10
        """).fetchdf()
        parts.append(part)
        print(f'  {label}: {len(user_ids):,} users -> {len(part):,} rows')

    for idx, user_id in enumerate(top_users, 1):
        query_user_subset([user_id], f'top user {idx}/10')

    n_batches = int(np.ceil(len(rest_users) / USER_BATCH_SIZE)) if rest_users else 0
    for start in range(0, len(rest_users), USER_BATCH_SIZE):
        query_user_subset(
            rest_users[start:start + USER_BATCH_SIZE],
            f'50K user batch {start // USER_BATCH_SIZE + 1}/{n_batches}'
        )

    df_user_tokens = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['user_id', 'n_requests', 'input_median', 'output_median']
    )

    df_user_tokens.to_parquet(cache_file)
    print(f'Saved {len(df_user_tokens):,} user rows to {cache_file}')

fig, ax = plt.subplots(figsize=(3.8, 3.0))

hb = ax.hexbin(
    df_user_tokens['input_median'],
    df_user_tokens['output_median'],
    xscale='log',
    yscale='log',
    gridsize=24,
    mincnt=1,
    cmap='viridis',
    norm=LogNorm(),
    linewidths=0,
)

ax.set_xlabel('Median user input')
ax.set_ylabel('Median user output')

add_log_x_grid(ax)
add_log_y_grid(ax)
# 1-^6 both
ax.set_xlim(1, 1_0000_00)
ax.set_ylim(1, 1_000_00)
# add correlation line
ax.plot([1, 100_000], [1, 100_000], color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='y = x')
cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label('Users')

finish_fig(fig, 'fig_user_median_input_output_heatmap.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_model_median_input_output_heatmap.pdf
#
# Per-model hexbin on (median input tokens × median output tokens), colored by
# model count (fig:token_shape_models — "Models: input vs. output"). Copied
# from the users-and-models figure script cell 5; per-model medians queried in rank-segmented
# batches (top-10 models solo, then the rest in one registered-relation join).
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: Per-model median input × median output, colored by model count ──
cached = CACHED
cache_file = CACHE_DIR / 'section3_model_median_input_output_segmented.parquet'
rank_cache_file = CACHE_DIR / 'section3_model_rank_for_segmented_queries.parquet'

if cached and cache_file.exists():
    df_model_tokens = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_model_tokens):,} model rows from {cache_file}')
else:
    if cached and rank_cache_file.exists():
        model_rank = pd.read_parquet(rank_cache_file)
        print(f'Loaded model ranks from {rank_cache_file}')
    else:
        model_rank = con.sql(f"""
            SELECT chute_id, COUNT(*) AS n_requests
            FROM {TABLE_NAME}
            WHERE chute_id IS NOT NULL
            GROUP BY chute_id
            ORDER BY n_requests DESC
        """).fetchdf()
        model_rank.to_parquet(rank_cache_file)
        print(f'Saved {len(model_rank):,} model ranks to {rank_cache_file}')

    top_models = model_rank['chute_id'].head(10).tolist()
    rest_models = model_rank['chute_id'].iloc[10:].tolist()
    parts = []

    def query_model_subset(model_ids, label):
        if not model_ids:
            return
        rel_name = 'section3_model_subset'
        con.register(rel_name, pd.DataFrame({'chute_id': model_ids}))
        try:
            part = con.sql(f"""
                SELECT
                    m.chute_id,
                    COUNT(*) AS n_requests,
                    approx_quantile(m.it, 0.5) AS input_median,
                    approx_quantile(m.ot, 0.5) AS output_median
                FROM {TABLE_NAME} m
                JOIN {rel_name} s USING (chute_id)
                WHERE m.chute_id IS NOT NULL
                  AND m.it IS NOT NULL AND m.it > 0
                  AND m.ot IS NOT NULL AND m.ot > 0
                GROUP BY m.chute_id
                HAVING COUNT(*) > 0
            """).fetchdf()
        finally:
            con.unregister(rel_name)
        parts.append(part)
        print(f'  {label}: {len(model_ids):,} models -> {len(part):,} rows')

    for idx, model_id in enumerate(top_models, 1):
        query_model_subset([model_id], f'top model {idx}/10')
    query_model_subset(rest_models, 'rest models')

    df_model_tokens = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['chute_id', 'n_requests', 'input_median', 'output_median']
    )
    df_model_tokens.to_parquet(cache_file)
    print(f'Saved {len(df_model_tokens):,} model rows to {cache_file}')

fig, ax = plt.subplots(figsize=(3.8, 3.0))
hb = ax.hexbin(
    df_model_tokens['input_median'], df_model_tokens['output_median'],
    xscale='log', yscale='log', gridsize=24, mincnt=1,
    cmap='viridis', norm=LogNorm(), linewidths=0
)
ax.set_xlabel('Median model input')
ax.set_ylabel('Median model output')
add_log_x_grid(ax)
add_log_y_grid(ax)
# make x and y limits the same to ensure y=x line is diagonal
max_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.set_xlim(1, max_limit)
ax.set_ylim(1, max_limit)
# add y = x reference line
ax.plot([1, 100000], [1, 100000], color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='y = x')
cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label('Models')
finish_fig(fig, 'fig_model_median_input_output_heatmap.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Access rasters: user → sampled models over time
#
# Thirty-day model-access rasters for two high-volume users
# (fig:user_access_persistent "Periodic user usage" and
# fig:user_access_exploration "Model exploration"). Copied from
# the users-and-models figure script cell 11 ("Standalone raster: user → sampled models over
# time"), which also builds the top-200 ranking cache it reads. The pipeline is
# RESTRICTED to the two users whose rasters the paper embeds; everything else
# (window ranking, model sampling, request sampling, cache naming, SQL-IN query
# mechanism) is kept as in the source cell.
# ────────────────────────────────────────────────────────────────────────────

# ── Standalone raster: user → sampled models over time ──
RASTER_CACHE_DIR = CACHE_DIR / "user_usage_rasters"
RASTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Required params.
USER_RASTER_CACHED = CACHED
USER_RASTER_TOP_N = 200
USER_RASTER_START_RANK = 25
USER_RASTER_MAX_FIGURES = 200
USER_RASTER_MIN_MODELS = 5
USER_RASTER_MAX_MODELS = 250

USER_RASTER_MODEL_SAMPLE_FRAC = 1.0
USER_RASTER_MODEL_SAMPLE_SEED = 20260504
USER_RASTER_REQUEST_SAMPLE_FRAC = 1.0
# Elapsed-time window (origin 1970-01-01): trace day 166 -> day 203.
RASTER_DATE_START = "1970-06-16"
RASTER_DATE_END = "1970-07-23"
PLOT_WINDOW_DAYS = 31

# The paper uses exactly two of the per-user rasters the source cell emits.
# They are addressed by their rank in the day 166 → 203 window,
# because user_id is remapped in the released trace and the original ids do
# not occur there. Rank is the stable handle: the window sits inside one
# id-rotation round, so the ranking is over the same people either way, and
# it was checked rank for rank on n_requests and n_models for all 200 users.
PAPER_RASTER_USER_RANKS = {190: "periodic.pdf", 134: "user_raster_explore.pdf"}


def _raster_out_name(row):
    """Output name for a selected user, by its rank in the window."""
    return PAPER_RASTER_USER_RANKS[int(row["rank"])]


def _pick_raster_user(df, rank):
    """The paper's user, addressed by rank within the window."""
    return df[df["rank"] == rank]


def _sql_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(values):
    values = [v for v in values if pd.notna(v)]
    if not values:
        return "('')"
    return "(" + ",".join(_sql_quote(v) for v in values) + ")"


def _safe_slug(value, max_len=90):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value[:max_len] if value else "unknown"


def _date_tag(start=RASTER_DATE_START, end=RASTER_DATE_END):
    return f"{start.replace('-', '')}_{end.replace('-', '')}"


window_tag = _date_tag()
rank_cache_file = CACHE_DIR / f"standalone_user_raster_top{USER_RASTER_TOP_N}_users_{window_tag}.parquet"
if USER_RASTER_CACHED and rank_cache_file.exists():
    top_users = pd.read_parquet(rank_cache_file)
    print(f"Loaded {len(top_users):,} ranked users from {rank_cache_file}")
else:
    top_users = con.sql(f"""
        SELECT
            user_id,
            COUNT(*) AS n_requests,
            approx_count_distinct(chute_id) AS n_models,
            MIN(started_at) AS first_seen
        FROM {TABLE_NAME}
        WHERE user_id IS NOT NULL
          AND chute_id IS NOT NULL
          AND started_at >= {_sql_quote(RASTER_DATE_START)}
          AND started_at < {_sql_quote(RASTER_DATE_END)}
        GROUP BY user_id
        ORDER BY n_requests DESC
        LIMIT {USER_RASTER_TOP_N}
    """).fetchdf()
    top_users["rank"] = np.arange(1, len(top_users) + 1)
    print(top_users.tail())
    print(f"Queried top {len(top_users):,} users by request count from {RASTER_DATE_START} to {RASTER_DATE_END}")
    top_users.to_parquet(rank_cache_file)
    print(f"Saved {len(top_users):,} ranked users to {rank_cache_file}")

# Only keep users whose first activity leaves at least PLOT_WINDOW_DAYS of data before RASTER_DATE_END.
top_users["first_seen"] = pd.to_datetime(top_users["first_seen"])
first_seen_cutoff = pd.to_datetime(RASTER_DATE_END) - pd.Timedelta(days=PLOT_WINDOW_DAYS)


def fetch_standalone_user_raster(user_id, rank):
    model_bp = max(1, int(round(USER_RASTER_MODEL_SAMPLE_FRAC * 10000)))
    req_bp = max(1, int(round(USER_RASTER_REQUEST_SAMPLE_FRAC * 10000)))
    cache_file = (
        RASTER_CACHE_DIR
        / f"standalone_user_rank{rank:02d}_{_safe_slug(user_id)}_{window_tag}_m{model_bp:05d}bp_r{req_bp:05d}bp_model_access_raster.parquet"
    )
    if USER_RASTER_CACHED and cache_file.exists():
        return pd.read_parquet(cache_file), cache_file

    user_sql = _sql_quote(user_id)
    start_sql = _sql_quote(RASTER_DATE_START)
    end_sql = _sql_quote(RASTER_DATE_END)
    models_for_user = con.sql(f"""
        SELECT chute_id, MIN(started_at) AS first_seen, COUNT(*) AS model_requests
        FROM {TABLE_NAME}
        WHERE user_id = {user_sql}
          AND chute_id IS NOT NULL
          AND started_at >= {start_sql}
          AND started_at < {end_sql}
        GROUP BY chute_id
        ORDER BY first_seen, chute_id
    """).fetchdf()
    models_for_user = models_for_user.sort_values(["first_seen", "chute_id"]).reset_index(drop=True)
    if models_for_user.empty:
        df = pd.DataFrame(columns=["started_at", "model_idx", "chute_id", "model_requests", "model_name"])
        df.to_parquet(cache_file)
        return df, cache_file

    n_available = len(models_for_user)
    n_sample = min(n_available, max(1, int(np.ceil(n_available * USER_RASTER_MODEL_SAMPLE_FRAC))))
    sampled_models = (
        models_for_user
        .sample(n=n_sample, random_state=USER_RASTER_MODEL_SAMPLE_SEED + int(rank))
        .sort_values(["first_seen", "chute_id"])
        .reset_index(drop=True)
    )
    sampled_models["model_idx"] = np.arange(len(sampled_models))
    model_idx = sampled_models[["chute_id", "model_idx", "model_requests"]]

    raw = con.sql(f"""
        SELECT started_at, chute_id
        FROM {TABLE_NAME}
        WHERE user_id = {user_sql}
          AND chute_id IN {_sql_in(model_idx["chute_id"])}
          AND started_at >= {start_sql}
          AND started_at < {end_sql}
          AND random() < {USER_RASTER_REQUEST_SAMPLE_FRAC}
        ORDER BY started_at, invocation_id
    """).fetchdf()
    df = raw.merge(model_idx, on="chute_id", how="left")
    df["model_name"] = df["chute_id"].map(CHUTE_TO_MODEL).fillna(df["chute_id"])
    df.to_parquet(cache_file)
    print(f"  sampled {len(model_idx):,}/{n_available:,} models; saved {len(df):,} requests")
    return df, cache_file


def plot_standalone_user_raster(df, user_id, rank, out_name):
    df = df.copy()
    df["started_at"] = pd.to_datetime(df["started_at"])
    fig, ax = plt.subplots(figsize=(3.5, 3))
    x_days = (df["started_at"] - pd.to_datetime(RASTER_DATE_START)).dt.total_seconds() / 86400
    # Normalize so each user's first request lands at day 0 (users start at different times within the query window).
    if not x_days.empty:
        x_days = x_days - x_days.min()
    # Keep only the first PLOT_WINDOW_DAYS days after each user's first request.
    mask = (x_days < PLOT_WINDOW_DAYS).values
    df = df.loc[mask].reset_index(drop=True)
    x_days = x_days.loc[mask].reset_index(drop=True)
    colors = plt.cm.tab20(df["model_idx"].astype(int) % 20) if len(df) else None
    ax.scatter(x_days, df["model_idx"], s=4, alpha=0.6, linewidths=0, c=colors, rasterized=True)
    ax.set_xlim(0, PLOT_WINDOW_DAYS)
    ax.set_xticks(np.arange(0, PLOT_WINDOW_DAYS + 1, 6))
    ax.set_xlabel("Days")
    ax.set_ylabel("Model index")
    ax.grid(True, alpha=0.22)
    n_requests = len(df)
    n_models = int(df["model_idx"].nunique()) if n_requests else 0
    ax.set_title(f"User rank {rank}\n{n_requests:,} requests, {n_models:,} models")
    out_path = PDF_DIR / out_name
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)
    return out_path


def render_user_rasters(selected):
    """Source cell's plotting loop, over whichever candidate rows are passed in."""
    outputs = []
    for idx, row in selected.iterrows():
        if row["n_models"] < USER_RASTER_MIN_MODELS:
            print(f"Skipping user rank {int(row['rank'])}: only {int(row['n_models'])} models")
            continue
        if row["n_models"] > USER_RASTER_MAX_MODELS:
            print(f"Skipping user rank {int(row['rank'])}: {int(row['n_models'])} models")
            continue
        if row["first_seen"] > first_seen_cutoff:
            print(f"Skipping user rank {int(row['rank'])}: first seen {row['first_seen'].date()} leaves <{PLOT_WINDOW_DAYS}d of history")
            continue
        df_raster, cache_file = fetch_standalone_user_raster(row["user_id"], int(row["rank"]))
        out_path = plot_standalone_user_raster(
            df_raster, row["user_id"], int(row["rank"]), _raster_out_name(row)
        )
        outputs.append({"rank": int(row["rank"]), "path": str(out_path), "n_points": len(df_raster)})
    return pd.DataFrame(outputs)


selected_users = top_users[top_users["rank"] >= USER_RASTER_START_RANK].head(USER_RASTER_MAX_FIGURES)
# Restrict to the two paper users (the source cell loops over all candidates;
# both paper users sit past the `idx <= 100` browse-skip the cell carried).
selected_users = selected_users[selected_users["rank"].isin(PAPER_RASTER_USER_RANKS)]
print(f"Selected users for raster plots: {selected_users.shape[0]} (by rank)")

# ────────────────────────────────────────────────────────────────────────────
# Figure: periodic.pdf
#
# Rank 190 in the day 166 → 203 window: a stable set of models
# accessed repeatedly across the month (fig:user_access_persistent,
# "Periodic user usage").
# ────────────────────────────────────────────────────────────────────────────

print(render_user_rasters(_pick_raster_user(selected_users, 190)))

# ────────────────────────────────────────────────────────────────────────────
# Figure: user_raster_explore.pdf
#
# Rank 134 in the same window: a few steady models plus a short burst of
# exploration across many others (fig:user_access_exploration,
# "Model exploration").
# ────────────────────────────────────────────────────────────────────────────

print(render_user_rasters(_pick_raster_user(selected_users, 134)))

# ────────────────────────────────────────────────────────────────────────────
# Access rasters: model → sampled users over time
#
# Thirty-day user-access rasters for two popular models
# (fig:model_access_power_user "Periodic user access" and
# fig:model_access_correlated "Correlated access"). Copied from
# the users-and-models figure script cell 12 ("Standalone raster: model → sampled users over
# time"), which also builds the top-200 model ranking cache it reads. The
# pipeline is RESTRICTED to the two models whose rasters the paper embeds;
# everything else (window ranking, user sampling, request sampling, cache
# naming, SQL-IN query mechanism) is kept as in the source cell.
#
# NOTE: this raster pair uses a LONGER window than the user rasters above —
# day 166 → 227 instead of day 166 → 203 — so the source
# cell's RASTER_DATE_START/RASTER_DATE_END/PLOT_WINDOW_DAYS are renamed with a
# MODEL_ prefix here to avoid clobbering the user-raster window.
# ────────────────────────────────────────────────────────────────────────────

# ── Standalone raster: model → sampled users over time ──
# _sql_quote / _sql_in / _safe_slug are identical in the two source cells and
# are reused from the user-raster block above.
MODEL_RASTER_CACHED = CACHED
MODEL_RASTER_TOP_N = 200
MODEL_RASTER_START_RANK = 25
MODEL_RASTER_MAX_FIGURES = 100
MODEL_RASTER_MIN_USERS = 5
MODEL_RASTER_CAP = 800
MODEL_RASTER_USER_SAMPLE_FRAC = 1.0
MODEL_RASTER_USER_SAMPLE_SEED = 20260504
MODEL_RASTER_REQUEST_SAMPLE_FRAC = 1.0
# Elapsed-time window (origin 1970-01-01): trace day 166 -> day 227.
MODEL_RASTER_DATE_START = "1970-06-16"
MODEL_RASTER_DATE_END = "1970-08-16"
MODEL_PLOT_WINDOW_DAYS = 31


def _model_name(chute_id):
    return CHUTE_TO_MODEL.get(chute_id, str(chute_id))


def _model_date_tag(start=MODEL_RASTER_DATE_START, end=MODEL_RASTER_DATE_END):
    return f"{start.replace('-', '')}_{end.replace('-', '')}"


# The paper uses exactly two of the per-model rasters the source cell emits.
# These (rank, model name) pairs — ranks within the day 166 → day 227
# window — were identified by md5-matching the paper's embedded PDFs against
# the notebook's auto-named outputs under paper/sections/pdf/final/section3/.
# Figure (a) keeps the notebook's auto-generated file name, which is what the
# tex \includegraphics call names; figure (b) is the same auto-named output
# (standalone_model_rank36_openai_gpt-oss-20b_d166_d227_u10000bp_
# r10000bp_user_access_raster.pdf) renamed by the paper.
PAPER_RASTER_MODELS = {
    # (a) — the notebook's own auto-generated name, kept verbatim
    (41, "Qwen/Qwen3-Next-80B-A3B-Instruct"):
        "standalone_model_rank41_Qwen_Qwen3-Next-80B-A3B-Instruct_d166_d227"
        "_u10000bp_r10000bp_user_access_raster.pdf",
    # (b) — renamed by the paper from standalone_model_rank36_openai_gpt-oss-20b_
    #       d166_d227_u10000bp_r10000bp_user_access_raster.pdf
    (36, "openai/gpt-oss-20b"): "openai_gpt-oss-20b_user_access_raster.pdf",
}

model_window_tag = _model_date_tag()
model_rank_cache_file = (
    CACHE_DIR / f"standalone_model_raster_top{MODEL_RASTER_TOP_N}_models_{model_window_tag}.parquet"
)
if MODEL_RASTER_CACHED and model_rank_cache_file.exists():
    top_models = pd.read_parquet(model_rank_cache_file)
    print(f"Loaded {len(top_models):,} ranked models from {model_rank_cache_file}")
else:
    top_models = con.sql(f"""
        SELECT
            chute_id,
            COUNT(*) AS n_requests,
            approx_count_distinct(user_id) AS n_users
        FROM {TABLE_NAME}
        WHERE chute_id IS NOT NULL
          AND user_id IS NOT NULL
          AND started_at >= {_sql_quote(MODEL_RASTER_DATE_START)}
          AND started_at < {_sql_quote(MODEL_RASTER_DATE_END)}
        GROUP BY chute_id
        ORDER BY n_requests DESC
        LIMIT {MODEL_RASTER_TOP_N}
    """).fetchdf()
    top_models["model_name"] = top_models["chute_id"].map(CHUTE_TO_MODEL).fillna(top_models["chute_id"])
    top_models["rank"] = np.arange(1, len(top_models) + 1)
    top_models.to_parquet(model_rank_cache_file)
    print(f"Saved {len(top_models):,} ranked models to {model_rank_cache_file}")


def fetch_standalone_model_raster(chute_id, rank):
    label = _safe_slug(_model_name(chute_id))
    user_bp = max(1, int(round(MODEL_RASTER_USER_SAMPLE_FRAC * 10000)))
    req_bp = max(1, int(round(MODEL_RASTER_REQUEST_SAMPLE_FRAC * 10000)))
    cache_file = (
        RASTER_CACHE_DIR
        / f"standalone_model_rank{rank:02d}_{label}_{model_window_tag}_u{user_bp:05d}bp_r{req_bp:05d}bp_user_access_raster.parquet"
    )
    if MODEL_RASTER_CACHED and cache_file.exists():
        return pd.read_parquet(cache_file), cache_file

    chute_sql = _sql_quote(chute_id)
    start_sql = _sql_quote(MODEL_RASTER_DATE_START)
    end_sql = _sql_quote(MODEL_RASTER_DATE_END)
    users_for_model = con.sql(f"""
        SELECT user_id, MIN(started_at) AS first_seen, COUNT(*) AS user_requests
        FROM {TABLE_NAME}
        WHERE chute_id = {chute_sql}
          AND user_id IS NOT NULL
          AND started_at >= {start_sql}
          AND started_at < {end_sql}
        GROUP BY user_id
        ORDER BY first_seen, user_id
    """).fetchdf()
    users_for_model = users_for_model.sort_values(["first_seen", "user_id"]).reset_index(drop=True)
    if users_for_model.empty:
        df = pd.DataFrame(columns=["started_at", "user_idx", "user_requests"])
        df.to_parquet(cache_file)
        return df, cache_file

    n_available = len(users_for_model)
    n_sample = min(n_available, max(1, int(np.ceil(n_available * MODEL_RASTER_USER_SAMPLE_FRAC))))
    sampled_users = (
        users_for_model
        .sample(n=n_sample, random_state=MODEL_RASTER_USER_SAMPLE_SEED + int(rank))
        .head(MODEL_RASTER_CAP)
        .sort_values(["first_seen", "user_id"])
        .reset_index(drop=True)
    )
    sampled_users["user_idx"] = np.arange(len(sampled_users))
    user_idx = sampled_users[["user_id", "user_idx", "user_requests"]]

    raw = con.sql(f"""
        SELECT started_at, user_id
        FROM {TABLE_NAME}
        WHERE chute_id = {chute_sql}
          AND user_id IN {_sql_in(sampled_users["user_id"])}
          AND started_at >= {start_sql}
          AND started_at < {end_sql}
          AND random() < {MODEL_RASTER_REQUEST_SAMPLE_FRAC}
        ORDER BY started_at, invocation_id
    """).fetchdf()
    df = raw.merge(user_idx, on="user_id", how="left").drop(columns=["user_id"])
    df.to_parquet(cache_file)
    print(f"  sampled {len(user_idx):,}/{n_available:,} users; saved {len(df):,} requests")
    return df, cache_file


def plot_standalone_model_raster(df, chute_id, rank, out_name):
    df = df.copy()
    df["started_at"] = pd.to_datetime(df["started_at"])
    fig, ax = plt.subplots(figsize=(3.5, 3))
    x_days = (df["started_at"] - pd.to_datetime(MODEL_RASTER_DATE_START)).dt.total_seconds() / 86400
    # Normalize so each model's first request lands at day 0 (models launch at different times within the query window).
    if not x_days.empty:
        x_days = x_days - x_days.min()
    # Keep only the first MODEL_PLOT_WINDOW_DAYS days after each model's first request.
    mask = (x_days < MODEL_PLOT_WINDOW_DAYS).values
    df = df.loc[mask].reset_index(drop=True)
    x_days = x_days.loc[mask].reset_index(drop=True)
    colors = plt.cm.tab20(df["user_idx"].astype(int) % 20) if len(df) else None
    ax.scatter(x_days, df["user_idx"], s=4, alpha=0.6, linewidths=0, c=colors, rasterized=True)
    ax.set_xlim(0, MODEL_PLOT_WINDOW_DAYS)
    ax.set_xticks(np.arange(0, MODEL_PLOT_WINDOW_DAYS + 1, 6))
    ax.set_xlabel("Days")
    ax.set_ylabel("User index")
    ax.grid(True, alpha=0.22)
    ax.set_title(f"Model rank {str(rank)[:5]}")
    out_path = PDF_DIR / out_name
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)
    return out_path


def render_model_rasters(selected):
    """Source cell's plotting loop, over whichever candidate rows are passed in."""
    outputs = []
    for idx, row in selected.iterrows():
        if idx <= 30:
            continue
        df_raster, cache_file = fetch_standalone_model_raster(row["chute_id"], int(row["rank"]))
        n_users = df_raster["user_idx"].nunique() if len(df_raster) else 0
        if n_users < MODEL_RASTER_MIN_USERS:
            print(f"Skipping model rank {int(row['rank'])}: {n_users} sampled users")
            continue
        out_name = PAPER_RASTER_MODELS[(int(row["rank"]), str(row["model_name"]))]
        auto_name = cache_file.name.replace(".parquet", ".pdf")
        if out_name.startswith("standalone_") and out_name != auto_name:
            print(f"  WARN: auto name {auto_name} != paper name {out_name}")
        out_path = plot_standalone_model_raster(df_raster, row["chute_id"], int(row["rank"]), out_name)
        outputs.append({"rank": int(row["rank"]), "path": str(out_path), "n_points": len(df_raster)})
    return pd.DataFrame(outputs)


selected_models = top_models[top_models["rank"] >= MODEL_RASTER_START_RANK].head(MODEL_RASTER_MAX_FIGURES)
# Restrict to the two paper models. The source cell loops over all candidates
# and skips `idx <= 30` (i.e. ranks up to 31) while browsing; both paper models
# (ranks 36 and 41) sit past that skip, so it is kept here unchanged.
selected_models = selected_models[
    [(int(r), str(n)) in PAPER_RASTER_MODELS
     for r, n in zip(selected_models["rank"], selected_models["model_name"])]
]
print("Selected models for raster plots:", selected_models.shape[0])

# ────────────────────────────────────────────────────────────────────────────
# Figure: standalone_model_rank41_Qwen_Qwen3-Next-80B-A3B-Instruct_
#         d166_d227_u10000bp_r10000bp_user_access_raster.pdf
#
# Qwen/Qwen3-Next-80B-A3B-Instruct, rank 41 in the day 166 → day 227
# window: long-lived horizontal bands of power users returning at regular
# intervals (fig:model_access_power_user, "Periodic user access").
# ────────────────────────────────────────────────────────────────────────────

print(render_model_rasters(
    selected_models[selected_models["model_name"] == "Qwen/Qwen3-Next-80B-A3B-Instruct"]
))

# ────────────────────────────────────────────────────────────────────────────
# Figure: openai_gpt-oss-20b_user_access_raster.pdf
#
# openai/gpt-oss-20b, rank 36 in the same window: sparse early activity that
# expands into dense vertical clusters of simultaneously active users
# (fig:model_access_correlated, "Correlated access").
# ────────────────────────────────────────────────────────────────────────────

print(render_model_rasters(
    selected_models[selected_models["model_name"] == "openai/gpt-oss-20b"]
))

# ────────────────────────────────────────────────────────────────────────────
# Model burstiness: popularity × CV(IAT)
#
# Shared data and axes for the two burstiness figures — a per-model table of
# (mean requests per active hour × mean hourly CV of IAT) plus lag-1 IAT
# autocorrelation. Copied from the burstiness figure script cell 20 with its
# helpers (cells 1/2) and, on the recompute path, its data chain: monthly
# hourly-burstiness batches (cell 4), the per-model summary (cell 6), and the
# plot_df quadrant table (cell 8, data part), the per-model lag-1 IAT
# autocorrelation batches (cell 13), and the metric suite that merges the two
# (cell 17). The density figure here needs only mean_req_per_hour ×
# cv_iat_mean; the autocorrelation columns feed the companion colored-scatter
# figure below, which shares this block's data and axes.
# ────────────────────────────────────────────────────────────────────────────

# ── Constants & helpers (from the burstiness figure script) ──
BATCH_DIR = CACHE_DIR / "_cache_model_hourly_burstiness_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

HOURLY_CACHE = CACHE_DIR / "_cache_model_hourly_burstiness.parquet"
SUMMARY_CACHE = CACHE_DIR / "_cache_model_burstiness_summary.parquet"
MODEL_COUNTS_CACHE = CACHE_DIR / "model_requests_vs_users.parquet"
MODEL_METRIC_AUTOCORR_CACHE = CACHE_DIR / "_cache_model_metric_autocorr_suite.parquet"

MIN_MODEL_REQUESTS = 100
MIN_HOUR_REQUESTS = 10
MIN_HOUR_IATS = 5
MIN_VALID_HOURS = 5


def style_grid(ax):  # burstiness variant (slightly different line weights)
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.32, linestyle="--", linewidth=0.55)
    ax.grid(True, which="minor", alpha=0.12, linestyle="--", linewidth=0.35)


def compact_number(x, _pos=None):
    if not np.isfinite(x):
        return ""
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:g}B"
    if x >= 1_000_000:
        return f"{x/1_000_000:g}M"
    if x >= 1_000:
        return f"{x/1_000:g}k"
    if x >= 1:
        return f"{x:g}"
    return f"{x:.2g}"


def short_model_name(name, chute_id=None, max_len=34):
    if pd.isna(name) or str(name).strip() == "":
        return str(chute_id)[:8] if chute_id is not None else "unknown"
    text = str(name).replace("/", "/")
    if len(text) <= max_len:
        return text
    head = text[: max_len - 1]
    return head.rstrip("-/_.") + "…"


def save_fig(fig, filename):
    out = PDF_DIR / filename
    fig.tight_layout(pad=0.45)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
    return out


def metric_label_map():  # from the burstiness figure script
    return {
        "log_total_requests": "Requests",
        "log_n_users": "Users",
        "mean_req_per_hour": "Mean req/hour",
        "cv_iat_mean": "Mean CV(IAT)",
        "log_mean_req_per_hour": "log10 mean req/hour",
        "log_median_req_per_hour": "log10 median req/hour",
        "log_cv_iat_median": "log10 CV(IAT) median",
        "log_cv_iat_p90": "log10 CV(IAT) P90",
        "burstiness_median": "Burstiness",
        "frac_hours_cv_gt_1": "Frac hours CV>1",
        "frac_hours_cv_gt_2": "Frac hours CV>2",
        "log_cv_req_per_hour": "log10 CV hourly load",
        "log_spike_to_median_req": "log10 peak/median load",
        "pearson_lag1_autocorr": "Pearson IAT autocorr",
        "spearman_lag1_autocorr": "Spearman IAT autocorr",
    }


metric_labels = metric_label_map()

# ── Data: metric suite (cached) or the cell-4/6/8 recompute chain ──
cached = CACHED

if cached and MODEL_METRIC_AUTOCORR_CACHE.exists():
    metric_suite = pd.read_parquet(MODEL_METRIC_AUTOCORR_CACHE)
    print(f"Loaded metric suite ({len(metric_suite):,} models) from {MODEL_METRIC_AUTOCORR_CACHE}")
else:
    # ── Hourly burstiness cache, one month per batch (cell 4) ──
    def _month_starts_from_trace():
        min_month, max_month = con.sql(f"""
            SELECT
                DATE_TRUNC('month', MIN(started_at))::DATE AS min_month,
                DATE_TRUNC('month', MAX(started_at))::DATE AS max_month
            FROM {TABLE_NAME}
            WHERE started_at IS NOT NULL
        """).fetchone()
        start = pd.Timestamp(min_month)
        stop = pd.Timestamp(max_month) + pd.offsets.MonthBegin(1)
        return list(pd.date_range(start, stop, freq="MS"))

    def _write_one_month(month_start, month_end, path, force=False):
        if path.exists() and not force:
            print(f"  exists {path.name}")
            return
        start_s = pd.Timestamp(month_start).strftime("%Y-%m-%d")
        end_s = pd.Timestamp(month_end).strftime("%Y-%m-%d")
        print(f"  computing {start_s} -> {end_s}")
        con.sql(f"""
            COPY (
                WITH gaps AS (
                    SELECT
                        chute_id,
                        DATE_TRUNC('hour', started_at)::TIMESTAMP AS hour,
                        EPOCH(started_at - LAG(started_at) OVER (
                            PARTITION BY chute_id, DATE_TRUNC('hour', started_at)
                            ORDER BY started_at, invocation_id
                        )) AS iat
                    FROM {TABLE_NAME}
                    WHERE chute_id IS NOT NULL
                      AND started_at >= TIMESTAMP '{start_s}'
                      AND started_at <  TIMESTAMP '{end_s}'
                ),
                hourly AS (
                    SELECT
                        chute_id,
                        hour,
                        COUNT(*)::BIGINT AS n_requests,
                        COUNT(iat) FILTER (WHERE iat > 0)::BIGINT AS n_iats,
                        AVG(iat) FILTER (WHERE iat > 0) AS mean_iat,
                        STDDEV_SAMP(iat) FILTER (WHERE iat > 0) AS std_iat
                    FROM gaps
                    GROUP BY 1, 2
                ),
                scored AS (
                    SELECT
                        chute_id,
                        hour,
                        n_requests,
                        n_iats,
                        mean_iat,
                        std_iat,
                        CASE
                            WHEN n_iats >= {MIN_HOUR_IATS}
                             AND mean_iat > 0
                             AND std_iat IS NOT NULL
                            THEN std_iat / mean_iat
                        END AS cv_iat,
                        n_requests / 3600.0 AS req_per_second,
                        n_requests::DOUBLE AS req_per_hour
                    FROM hourly
                    WHERE n_requests >= {MIN_HOUR_REQUESTS}
                )
                SELECT
                    chute_id,
                    hour,
                    n_requests,
                    n_iats,
                    mean_iat,
                    std_iat,
                    cv_iat,
                    CASE WHEN cv_iat IS NOT NULL THEN (cv_iat - 1.0) / (cv_iat + 1.0) END AS burstiness,
                    req_per_second,
                    req_per_hour
                FROM scored
                ORDER BY chute_id, hour
            ) TO '{path.as_posix()}' (FORMAT 'parquet')
        """)
        print(f"  wrote {path.name}")

    def compute_or_load_hourly_burstiness(cached=True, force_months=False):
        month_starts = _month_starts_from_trace()
        batch_paths = []
        for start, end in zip(month_starts[:-1], month_starts[1:]):
            path = BATCH_DIR / f"model_hourly_burstiness_{pd.Timestamp(start):%Y%m}.parquet"
            batch_paths.append(path)
            if (not cached) or force_months or (not path.exists()):
                _write_one_month(start, end, path, force=force_months)
            else:
                print(f"  exists {path.name}")

        if (not cached) or (not HOURLY_CACHE.exists()):
            glob_path = (BATCH_DIR / "model_hourly_burstiness_*.parquet").as_posix()
            con.sql(f"""
                COPY (
                    SELECT *
                    FROM read_parquet('{glob_path}')
                    ORDER BY chute_id, hour
                ) TO '{HOURLY_CACHE.as_posix()}' (FORMAT 'parquet')
            """)
            print(f"wrote consolidated hourly cache: {HOURLY_CACHE}")
        else:
            print(f"loaded consolidated hourly cache: {HOURLY_CACHE}")
        return pd.read_parquet(HOURLY_CACHE)

    # ── Model-level summary (cell 6) ──
    def load_or_compute_model_counts(cached=True):
        if cached and MODEL_COUNTS_CACHE.exists():
            return pd.read_parquet(MODEL_COUNTS_CACHE)
        print("computing per-model request/user counts")
        df = con.sql(f"""
            SELECT
                chute_id,
                COUNT(*)::BIGINT AS n_requests,
                COUNT(DISTINCT user_id)::BIGINT AS n_users
            FROM {TABLE_NAME}
            WHERE chute_id IS NOT NULL AND user_id IS NOT NULL
            GROUP BY 1
        """).fetchdf()
        df.to_parquet(MODEL_COUNTS_CACHE)
        return df

    def summarize_models(df_hourly, cached=True):
        if cached and SUMMARY_CACHE.exists():
            out = pd.read_parquet(SUMMARY_CACHE)
            out["first_hour"] = pd.to_datetime(out["first_hour"])
            out["last_hour"] = pd.to_datetime(out["last_hour"])
            return out

        model_counts = load_or_compute_model_counts(cached=True).rename(columns={"n_requests": "total_requests"})
        active = (
            df_hourly.groupby("chute_id", as_index=False)
            .agg(
                n_hour_windows=("hour", "count"),
                first_hour=("hour", "min"),
                last_hour=("hour", "max"),
                window_requests=("n_requests", "sum"),
                mean_req_per_hour_all=("req_per_hour", "mean"),
                median_req_per_hour_all=("req_per_hour", "median"),
                max_req_per_hour=("req_per_hour", "max"),
                std_req_per_hour=("req_per_hour", "std"),
            )
        )
        active["cv_req_per_hour"] = active["std_req_per_hour"] / active["mean_req_per_hour_all"].replace(0, np.nan)
        active["spike_to_median_req"] = active["max_req_per_hour"] / active["median_req_per_hour_all"].replace(0, np.nan)

        valid = df_hourly.dropna(subset=["cv_iat"]).replace([np.inf, -np.inf], np.nan)
        valid = valid[valid["cv_iat"] > 0].copy()
        cv_summary = (
            valid.groupby("chute_id")
            .agg(
                n_valid_hours=("hour", "count"),
                cv_iat_mean=("cv_iat", "mean"),
                cv_iat_median=("cv_iat", "median"),
                cv_iat_p75=("cv_iat", lambda s: np.nanquantile(s, 0.75)),
                cv_iat_p90=("cv_iat", lambda s: np.nanquantile(s, 0.90)),
                burstiness_mean=("burstiness", "mean"),
                burstiness_median=("burstiness", "median"),
                frac_hours_cv_gt_1=("cv_iat", lambda s: float(np.nanmean(s > 1.0))),
                frac_hours_cv_gt_2=("cv_iat", lambda s: float(np.nanmean(s > 2.0))),
                mean_req_per_hour=("req_per_hour", "mean"),
                median_req_per_hour=("req_per_hour", "median"),
            )
            .reset_index()
        )

        out = model_counts.merge(active, on="chute_id", how="left").merge(cv_summary, on="chute_id", how="left")
        out["model_name"] = out["chute_id"].map(CHUTE_TO_MODEL).fillna("unknown")
        out["short_name"] = [short_model_name(n, c) for n, c in zip(out["model_name"], out["chute_id"])]
        out = out[(out["total_requests"] > MIN_MODEL_REQUESTS) & (out["n_valid_hours"] >= MIN_VALID_HOURS)].copy()
        out = out.sort_values("total_requests", ascending=False).reset_index(drop=True)
        out.to_parquet(SUMMARY_CACHE)
        return out

    df_hourly = compute_or_load_hourly_burstiness(cached=cached)
    df_hourly["hour"] = pd.to_datetime(df_hourly["hour"])
    df_summary = summarize_models(df_hourly, cached=cached)

    # ── plot_df quadrant table (cell 8, data part) ──
    plot_df = df_summary.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mean_req_per_hour", "cv_iat_median", "burstiness_median"]
    ).copy()
    plot_df = plot_df[(plot_df["mean_req_per_hour"] > 0) & (plot_df["cv_iat_median"] > 0)]

    POPULARITY_THRESHOLD = float(plot_df["mean_req_per_hour"].median())
    CV_QUADRANT_THRESHOLD = float(plot_df["cv_iat_median"].median())

    plot_df["pop_class"] = np.where(plot_df["mean_req_per_hour"] >= POPULARITY_THRESHOLD, "popular", "less popular")
    plot_df["burst_class"] = np.where(plot_df["cv_iat_median"] >= CV_QUADRANT_THRESHOLD, "higher CV", "lower CV")
    plot_df["quadrant"] = plot_df["pop_class"] + " / " + plot_df["burst_class"]

    # ── Lag-1 IAT autocorrelation for the plotted models (cell 13) ──
    # Batched WHERE chute_id IN (...) queries so the raw-table scan stays light:
    # the first 25 models one at a time, then by 5 up to 100, by 100 up to 1000,
    # then by 1000. Each batch writes its own parquet, so an interrupted run
    # resumes where it stopped.
    AUTOCORR_BATCH_DIR = CACHE_DIR / "_cache_model_iat_lag1_autocorr_where_batches"
    AUTOCORR_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    AUTOCORR_CACHE = CACHE_DIR / "_cache_model_iat_lag1_autocorr_where_all_plot_models.parquet"
    AUTOCORR_MAX_REQUESTS = 100_000_000
    FORCE_AUTOCORR_BATCHES = False

    def sql_literal(value):
        return "'" + str(value).replace("'", "''") + "'"

    def _batch_is_done(path):
        return path.exists() and path.stat().st_size > 0

    def autocorr_model_list():
        # Start from the cached popularity dataframe, then keep only models present in the earlier plots.
        popularity = pd.read_parquet(MODEL_COUNTS_CACHE).rename(columns={"n_requests": "total_requests"})
        plotted = plot_df[["chute_id", "model_name", "short_name"]].drop_duplicates("chute_id")
        models = popularity.merge(plotted, on="chute_id", how="inner")
        models = models[models["total_requests"] < AUTOCORR_MAX_REQUESTS].copy()
        models = models.sort_values("total_requests", ascending=False).reset_index(drop=True)
        return models[["chute_id", "model_name", "short_name", "total_requests"]].copy()

    def build_autocorr_batch_plan(models):
        plan = []
        start = 0
        n = len(models)
        while start < n:
            if start < 25:
                batch_size = 1
            elif start < 100:
                batch_size = 5
            elif start < 1000:
                batch_size = 100
            else:
                batch_size = 1000
            end = min(start + batch_size, n)
            batch = models.iloc[start:end].copy()
            tag = f"batch_{len(plan) + 1:04d}_models_{start + 1:05d}_{end:05d}_n{len(batch):04d}"
            plan.append((tag, batch, AUTOCORR_BATCH_DIR / f"{tag}.parquet"))
            start = end
        return plan

    def compute_autocorr_batch(batch, out_path):
        ids_sql = ", ".join(sql_literal(mid) for mid in batch["chute_id"])
        if not ids_sql:
            raise ValueError("empty autocorrelation batch")
        con.sql(f"""
            COPY (
                WITH request_iat AS (
                    SELECT
                        chute_id,
                        invocation_id,
                        started_at,
                        EPOCH(started_at - LAG(started_at) OVER (
                            PARTITION BY chute_id
                            ORDER BY started_at, invocation_id
                        )) AS iat
                    FROM {TABLE_NAME}
                    WHERE started_at IS NOT NULL
                      AND chute_id IN ({ids_sql})
                ),
                iat_pairs AS (
                    SELECT
                        chute_id,
                        iat AS x,
                        LAG(iat) OVER (
                            PARTITION BY chute_id
                            ORDER BY started_at, invocation_id
                        ) AS x_lag
                    FROM request_iat
                    WHERE iat IS NOT NULL AND iat > 0
                ),
                clean_pairs AS (
                    SELECT *
                    FROM iat_pairs
                    WHERE x_lag IS NOT NULL AND x_lag > 0
                ),
                spearman_pairs AS (
                    SELECT
                        chute_id,
                        x,
                        x_lag,
                        RANK() OVER (PARTITION BY chute_id ORDER BY x) AS x_rank,
                        RANK() OVER (PARTITION BY chute_id ORDER BY x_lag) AS x_lag_rank
                    FROM clean_pairs
                )
                SELECT
                    chute_id,
                    COUNT(*)::BIGINT AS n_iat_pairs,
                    AVG(x) AS mean_iat,
                    STDDEV_SAMP(x) AS std_iat,
                    CORR(x, x_lag) AS pearson_lag1_autocorr,
                    CORR(x_rank::DOUBLE, x_lag_rank::DOUBLE) AS spearman_lag1_autocorr
                FROM spearman_pairs
                GROUP BY chute_id
            ) TO '{out_path.as_posix()}' (FORMAT 'parquet')
        """)

    def compute_or_load_all_model_autocorr(cached=True, force_batches=False):
        import time

        models = autocorr_model_list()
        plan = build_autocorr_batch_plan(models)
        print(
            f"Autocorr target: {len(models):,} plotted models with "
            f"total_requests < {AUTOCORR_MAX_REQUESTS:,}; {len(plan):,} batches"
        )
        print("Query shape: SELECT from raw table WHERE chute_id IN (<batch ids>)")
        print("Batch schedule: first 25 by 1, then by 5 to 100, by 100 to 1000, then by 1000")

        for batch_idx, (tag, batch, path) in enumerate(plan, 1):
            req_max = int(batch["total_requests"].max())
            req_min = int(batch["total_requests"].min())
            first_name = batch["short_name"].iloc[0]
            if cached and (not force_batches) and _batch_is_done(path):
                print(f"[{batch_idx:,}/{len(plan):,}] size={len(batch):,} exists -> {path.name}")
                continue

            print(
                f"[{batch_idx:,}/{len(plan):,}] size={len(batch):,} "
                f"req={req_max:,}..{req_min:,} first={first_name} | querying...",
                flush=True,
            )
            t0 = time.time()
            compute_autocorr_batch(batch, path)
            elapsed = time.time() - t0
            rows = len(pd.read_parquet(path)) if _batch_is_done(path) else 0
            print(f"    wrote {path.name} rows={rows:,} elapsed={elapsed:.1f}s", flush=True)

        batch_paths = sorted(AUTOCORR_BATCH_DIR.glob("batch_*.parquet"))
        if not batch_paths:
            raise RuntimeError(f"No autocorrelation batch files found in {AUTOCORR_BATCH_DIR}")

        df = pd.concat([pd.read_parquet(p) for p in batch_paths], ignore_index=True)
        df = df.merge(models, on="chute_id", how="left")
        df = df.sort_values("total_requests", ascending=False).reset_index(drop=True)
        df.to_parquet(AUTOCORR_CACHE)
        print(f"wrote {AUTOCORR_CACHE} rows={len(df):,}")
        return df

    df_autocorr = compute_or_load_all_model_autocorr(
        cached=cached,
        force_batches=FORCE_AUTOCORR_BATCHES,
    )

    # ── Autocorrelation-aware model metric suite (cell 17) ──
    def signed_log1p(values):
        values = pd.to_numeric(values, errors="coerce")
        return np.sign(values) * np.log10(1.0 + np.abs(values))

    def build_metric_autocorr_table(plot_df, df_autocorr):
        pop = pd.read_parquet(MODEL_COUNTS_CACHE).rename(
            columns={"n_requests": "total_requests_pop", "n_users": "n_users_pop"}
        )
        base = plot_df.merge(pop[["chute_id", "n_users_pop", "total_requests_pop"]], on="chute_id", how="left")
        if "total_requests" not in base.columns:
            base["total_requests"] = base["total_requests_pop"]
        else:
            base["total_requests"] = base["total_requests"].fillna(base["total_requests_pop"])
        if "n_users" not in base.columns:
            base["n_users"] = base["n_users_pop"]
        else:
            base["n_users"] = base["n_users"].fillna(base["n_users_pop"])

        auto_cols = [
            "chute_id",
            "n_iat_pairs",
            "pearson_lag1_autocorr",
            "spearman_lag1_autocorr",
        ]
        merged = base.merge(df_autocorr[auto_cols], on="chute_id", how="left")
        merged = merged.replace([np.inf, -np.inf], np.nan).copy()

        merged["log_total_requests"] = np.log10(merged["total_requests"].where(merged["total_requests"] > 0))
        merged["log_n_users"] = np.log10(merged["n_users"].where(merged["n_users"] > 0))
        merged["log_mean_req_per_hour"] = np.log10(merged["mean_req_per_hour"].where(merged["mean_req_per_hour"] > 0))
        merged["log_median_req_per_hour"] = np.log10(merged["median_req_per_hour"].where(merged["median_req_per_hour"] > 0))
        merged["log_cv_iat_median"] = np.log10(merged["cv_iat_median"].where(merged["cv_iat_median"] > 0))
        merged["log_cv_iat_p90"] = np.log10(merged["cv_iat_p90"].where(merged["cv_iat_p90"] > 0))
        merged["log_cv_req_per_hour"] = np.log10(merged["cv_req_per_hour"].where(merged["cv_req_per_hour"] > 0))
        merged["log_spike_to_median_req"] = np.log10(merged["spike_to_median_req"].where(merged["spike_to_median_req"] > 0))

        cols = ["chute_id", "model_name", "short_name", "quadrant"] + list(metric_label_map().keys()) + [
            "total_requests",
            "n_users",
            "cv_iat_median",
            "n_iat_pairs",
        ]
        # Keep export order stable while removing duplicates introduced by metrics
        # that are both plotted and useful as raw columns.
        cols = list(dict.fromkeys(c for c in cols if c in merged.columns))
        out = merged.loc[:, cols].copy()
        out.to_parquet(MODEL_METRIC_AUTOCORR_CACHE)
        print(f"metric suite rows: {len(out):,}; autocorr coverage: {out['pearson_lag1_autocorr'].notna().sum():,}")
        print(f"wrote {MODEL_METRIC_AUTOCORR_CACHE}")
        return out

    metric_suite = build_metric_autocorr_table(plot_df, df_autocorr)

# ── Fixed X/Y axes used for all heatmap-style views (cell 20) ──
fixed_xy = metric_suite.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["mean_req_per_hour", "cv_iat_mean"]
).copy()
fixed_xy = fixed_xy[(fixed_xy["mean_req_per_hour"] > 0) & (fixed_xy["cv_iat_mean"] > 0)]

FIXED_X = "mean_req_per_hour"
FIXED_Y = "cv_iat_mean"
FIXED_X_LABEL = "Requests/hour"
FIXED_Y_LABEL = "Mean CV(IAT)"

# Change this list to control which metrics appear in the fixed-axis scatter/hex maps.
# Density (# models) is plotted separately below and does not need to be listed here.
# The paper shows only the Spearman panel; add "pearson_lag1_autocorr" back to
# render both side by side.
FIXED_HEATMAP_COLOR_COLUMNS = [
    "spearman_lag1_autocorr",
]

FIXED_HEATMAP_GRIDSIZE = 32
FIXED_HEATMAP_SCATTER_SIZE = 12
FIXED_HEATMAP_SCATTER_ALPHA = 0.68

fixed_heatmap_metric_styles = {
    "pearson_lag1_autocorr": ("Pearson lag-1 IAT autocorr", "coolwarm", "centered"),
    "spearman_lag1_autocorr": ("Spearman lag-1 IAT autocorr", "coolwarm", "centered"),
    "burstiness_median": ("Median burstiness", "coolwarm", "centered"),
    "log_total_requests": ("log10 requests", "viridis", "plain"),
    "log_n_users": ("log10 users", "viridis", "plain"),
    "log_cv_req_per_hour": ("log10 CV hourly load", "magma", "plain"),
    "log_spike_to_median_req": ("log10 peak/median load", "magma", "plain"),
    "frac_hours_cv_gt_2": ("Fraction hours CV(IAT)>2", "plasma", "plain"),
}

def fixed_metric_spec(col):
    label, cmap, norm_kind = fixed_heatmap_metric_styles.get(
        col,
        (metric_labels.get(col, col), "viridis", "plain"),
    )
    return col, label, cmap, norm_kind

color_specs = [
    fixed_metric_spec(col)
    for col in FIXED_HEATMAP_COLOR_COLUMNS
    if col in fixed_xy.columns
]
missing_color_cols = [col for col in FIXED_HEATMAP_COLOR_COLUMNS if col not in fixed_xy.columns]
if missing_color_cols:
    print(f"Skipping missing fixed heatmap columns: {missing_color_cols}")
print("Fixed-axis color metrics:", [label for _, label, _, _ in color_specs])

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_model_heatmap_fixedxy_density.pdf
#
# Per-model hexbin on the fixed axes above, colored by model density
# (fig:model_burstiness_density — "Popularity vs. burstiness"). Density part of
# the burstiness figure script cell 20.
# ────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(3.7, 3.2))
hb = ax.hexbin(
    fixed_xy[FIXED_X],
    fixed_xy[FIXED_Y],
    xscale="log",
    yscale="log",
    gridsize=FIXED_HEATMAP_GRIDSIZE,
    mincnt=1,
    cmap="viridis",
    norm=LogNorm(),
    linewidths=0,
)
ax.set_xlabel(FIXED_X_LABEL)
ax.set_ylabel(FIXED_Y_LABEL)
ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
style_grid(ax)
cbar = fig.colorbar(hb, ax=ax, pad=0.015)
cbar.set_label("Models")
save_fig(fig, "fig_model_heatmap_fixedxy_density.pdf")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf
#
# Same fixed axes as the density figure, one point per model, colored by the
# lag-1 IAT autocorrelation (fig:model_burstiness_autocorr — "Lag-1 IAT
# autocorrelation"). Copied from the burstiness figure script cell 21; it
# reuses fixed_xy / color_specs / FIXED_* from the block above. The source
# emits one panel per entry of FIXED_HEATMAP_COLOR_COLUMNS (Spearman, then
# Pearson) side by side, and the paper cropped to the left panel with
# `trim=0 0 270pt 0`. Here only the Spearman panel is drawn, so that crop is no
# longer needed -- drop `trim=0 0 270pt 0, clip` from the \includegraphics call.
# ────────────────────────────────────────────────────────────────────────────

# ── Same X/Y, individual model scatter, varying only the selected color metric ──
from matplotlib.colors import Normalize, TwoSlopeNorm

if not color_specs:
    print("No fixed-axis scatter metrics selected. Add columns to FIXED_HEATMAP_COLOR_COLUMNS.")
else:
    n_cols = min(2, len(color_specs))
    n_rows = int(np.ceil(len(color_specs) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.7 * n_cols, max(3.2 * n_rows, 3.2)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    for ax, (col, label, cmap, norm_kind) in zip(axes, color_specs):
        sub = fixed_xy.dropna(subset=[col])
        if sub.empty:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        if norm_kind == "centered":
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        else:
            vals = sub[col].astype(float)
            norm = Normalize(vmin=np.nanquantile(vals, 0.02), vmax=np.nanquantile(vals, 0.98))
        sc = ax.scatter(
            sub[FIXED_X],
            sub[FIXED_Y],
            c=sub[col],
            s=FIXED_HEATMAP_SCATTER_SIZE,
            alpha=FIXED_HEATMAP_SCATTER_ALPHA,
            linewidth=0,
            cmap=cmap,
            norm=norm,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
        ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
        ax.set_xlabel(FIXED_X_LABEL)
        ax.set_ylabel(FIXED_Y_LABEL)
        style_grid(ax)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Lag-1 autocorr")
    for ax in axes[len(color_specs):]:
        ax.axis("off")
    save_fig(fig, "fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf")
    plt.close(fig)

print("Done — all Section 4 figures written to", PDF_DIR)
