#!/usr/bin/env python3
"""§5 One-Year Workload Evolution and §6 Prefix Caching: the prefix-caching figures.

Figures (written to figures/paper/04_prefix_caching/), in §5 unless noted:
  fig_request_cached_fraction_cdf_by_input.pdf  token hit ratio CDF: Global, <1K and >=8K input tokens
  fig_request_token_hit_rate_cdf_by_model.pdf   token hit ratio CDF: Global and the highlighted models
  fig_model_nreq_vs_avg_hitratio_hexbin.pdf     models by request count and average token hit ratio
  fig_user_nreq_vs_avg_hitratio_hexbin.pdf      users by request count and average token hit ratio
  fig_iat_monthly_per_user_band.pdf             per-user inter-arrival time (IAT) by trace month
  fig_per_user_iat_p50_by_model.pdf             each user's median IAT to each highlighted model
  fig_iat_pair_ttl_coverage.pdf                 (§6) share of (user, model) repeat requests within a TTL

The token hit ratio of a request is ct / it (cached over input tokens),
clipped to [0, 1]. The highlighted models are listed in
config/highlight_models.json. §6's two eviction-policy figures come from
libCacheSim rather than from the trace; pipeline/caching/ draws them.

Query results are cached in output/paper/cache/; --recompute reruns every
query against the trace.

Usage:
    python paper/04_prefix_caching/reproduce.py [--db PATH] [--recompute]
"""

import matplotlib
matplotlib.use("Agg")  # headless; must precede the pyplot import

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import (
    AutoMinorLocator, LogFormatterMathtext, LogLocator,
    MultipleLocator, NullFormatter, NullLocator,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import LazyDB, trace_month  # noqa: E402
from _paths import DB_DEFAULT, REPO, section_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the anonymized trace "
                         "(scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true", help="rerun the DuckDB queries even if their cache files exist")
args = parser.parse_args()
CACHED = not args.recompute   # True: reuse cache files and just replot

PDF_DIR, CACHE_DIR = section_paths("04_prefix_caching")
con = LazyDB(args.db)
TABLE_NAME = "all_metrics_user"

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


def style_grid(ax):
    """Dashed grid on both major and minor axes, with major more visible than minor."""
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.12, linestyle='--', linewidth=0.4)


def standard_ticks(ax):
    """Major/minor tick lengths shared across the paper figures."""
    ax.tick_params(axis='both', which='major', length=4.5, width=0.8)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)


def save(fig, fname):
    fig.tight_layout(pad=0.4)
    fig.savefig(PDF_DIR / fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {PDF_DIR / fname}")


# All requests: a dashed grey line under the per-cohort and per-model lines.
GLOBAL_KW = dict(color='#444444', linestyle=(0, (4, 2)),
                 alpha=0.7, linewidth=1.5, label='Global')

# The 101 percentiles 0.00, 0.01, ..., 1.00 of the token hit ratio queries.
PCTS = [i / 100 for i in range(101)]
PCT_SQL = ', '.join(f'{p:.2f}' for p in PCTS)

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_request_cached_fraction_cdf_by_input.pdf
#
# CDF of the per-request token hit ratio for all requests (Global) and for
# requests with under 1K or at least 8K input tokens.
# ────────────────────────────────────────────────────────────────────────────

cache_file = CACHE_DIR / 'request_hit_rate_cdf_by_input_length.parquet'
if not CACHED or not cache_file.exists():
    print('Computing global per-request token hit ratio percentiles …')
    global_q = con.sql(f"""
        SELECT APPROX_QUANTILE(cache_frac, [{PCT_SQL}]) AS q
        FROM (
            SELECT LEAST(GREATEST(CAST(ct AS DOUBLE) / it, 0.0), 1.0) AS cache_frac
            FROM {TABLE_NAME}
            WHERE ct IS NOT NULL
              AND it IS NOT NULL AND it > 0
        ) x
    """).fetchone()[0]

    print('Computing per-request token hit ratio percentiles by input length …')
    df_bucket_q = con.sql(f"""
        SELECT
            CASE
                WHEN it < 1000 THEN '<1K input'
                ELSE '>8K input'
            END AS input_bucket,
            APPROX_QUANTILE(LEAST(GREATEST(CAST(ct AS DOUBLE) / it, 0.0), 1.0), [{PCT_SQL}]) AS q,
            COUNT(*) AS n_requests
        FROM {TABLE_NAME}
        WHERE ct IS NOT NULL
          AND it IS NOT NULL AND it > 0
          AND (it < 1000 OR it >= 8000)
        GROUP BY 1
    """).fetchdf()

    rows = [{'series': 'Global', 'pct': pct, 'cache_frac': val, 'n_requests': None}
            for pct, val in zip(PCTS, global_q)]
    rows += [{'series': row['input_bucket'], 'pct': pct, 'cache_frac': val,
              'n_requests': row['n_requests']}
             for _, row in df_bucket_q.iterrows() for pct, val in zip(PCTS, row['q'])]
    pd.DataFrame(rows).to_parquet(cache_file)

df_cache_frac_cdf = pd.read_parquet(cache_file)
COHORT_COLORS = {'<1K input': '#1f77b4', '>8K input': '#ff7f0e'}  # tab10 blue, orange

fig, ax = plt.subplots(figsize=(3.5, 3.0))
global_sub = df_cache_frac_cdf[df_cache_frac_cdf['series'] == 'Global'].sort_values('pct')
ax.plot(global_sub['cache_frac'], global_sub['pct'], **GLOBAL_KW)
for series, color in COHORT_COLORS.items():
    sub = df_cache_frac_cdf[df_cache_frac_cdf['series'] == series].sort_values('pct')
    ax.plot(sub['cache_frac'], sub['pct'], color=color, linewidth=1.5, label=series)
ax.set_xlabel('Token hit ratio')
ax.set_ylabel('CDF')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='lower right', frameon=False, fontsize=8)
standard_ticks(ax)
style_grid(ax)
save(fig, 'fig_request_cached_fraction_cdf_by_input.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Highlighted models
#
# config/highlight_models.json maps each figure label to a chute_id. The
# cached table (shared with paper/02_token_shape_latency) adds each model's
# request count and its tab10 color, and the per-model queries join against
# it. After editing the config, rerun with --recompute.
# ────────────────────────────────────────────────────────────────────────────

HIGHLIGHT_MODELS = json.loads((REPO / "config/highlight_models.json").read_text())
HL_PARQUET = CACHE_DIR / "highlight_models.parquet"

if not CACHED or not HL_PARQUET.exists():
    print("Counting the requests of each highlighted model …")
    df_hl = pd.DataFrame({"label": list(HIGHLIGHT_MODELS),
                          "chute_id": list(HIGHLIGHT_MODELS.values())})
    con.register("__hl_chutes", df_hl[["chute_id"]])
    counts = con.sql(f"""
        SELECT m.chute_id, COUNT(*) AS n_requests
        FROM {TABLE_NAME} m
        JOIN __hl_chutes h ON m.chute_id = h.chute_id
        GROUP BY 1
    """).fetchdf()
    con.unregister("__hl_chutes")
    df_hl = df_hl.merge(counts, on="chute_id", how="left")
    df_hl["n_requests"] = df_hl["n_requests"].fillna(0).astype(int)
    tab10 = plt.get_cmap("tab10").colors
    df_hl["color"] = [tab10[i % len(tab10)] for i in range(len(df_hl))]
    df_hl.to_parquet(HL_PARQUET, index=False)

df_hl = pd.read_parquet(HL_PARQUET)

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_request_token_hit_rate_cdf_by_model.pdf
#
# CDF of the per-request token hit ratio for all requests (Global) and for
# each highlighted model.
# ────────────────────────────────────────────────────────────────────────────

model_cache = CACHE_DIR / "request_hit_rate_cdf_by_model.parquet"
if not CACHED or not model_cache.exists():
    print("Computing per-request token hit ratio percentiles per highlighted model …")
    df_q = con.sql(f"""
        SELECT m.chute_id,
               approx_quantile(LEAST(m.ct::DOUBLE / m.it, 1.0), [{PCT_SQL}]) AS q,
               COUNT(*) AS n
        FROM {TABLE_NAME} m
        JOIN read_parquet('{HL_PARQUET}') h ON m.chute_id = h.chute_id
        WHERE m.it IS NOT NULL AND m.it > 0 AND m.ct IS NOT NULL AND m.ct >= 0
        GROUP BY 1
    """).fetchdf()
    rows = [{"chute_id": r["chute_id"], "pct": p, "cache_frac": r["q"][i], "n": r["n"]}
            for _, r in df_q.iterrows() for i, p in enumerate(PCTS)]
    pd.DataFrame(rows).to_parquet(model_cache)

global_cache = CACHE_DIR / "request_hit_rate_cdf_global.parquet"
if not CACHED or not global_cache.exists():
    print("Computing global per-request token hit ratio percentiles …")
    global_q = con.sql(f"""
        SELECT approx_quantile(LEAST(ct::DOUBLE / it, 1.0), [{PCT_SQL}]) AS q
        FROM {TABLE_NAME}
        WHERE it IS NOT NULL AND it > 0 AND ct IS NOT NULL AND ct >= 0
    """).fetchone()[0]
    pd.DataFrame({"pct": PCTS, "cache_frac": list(global_q)}).to_parquet(global_cache)

df_models = pd.read_parquet(model_cache)
df_global = pd.read_parquet(global_cache).sort_values("pct")

fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot(df_global["cache_frac"], df_global["pct"], **GLOBAL_KW)
for _, r in df_hl.iterrows():
    sub = df_models[df_models["chute_id"] == r["chute_id"]].sort_values("cache_frac")
    ax.plot(sub["cache_frac"], sub["pct"],
            color=r["color"], linewidth=1.5, label=r["label"])
ax.set_xlabel("Token hit ratio")
ax.set_ylabel("CDF")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
standard_ticks(ax)
style_grid(ax)
save(fig, "fig_request_token_hit_rate_cdf_by_model.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figures: fig_model_nreq_vs_avg_hitratio_hexbin.pdf, fig_user_nreq_vs_avg_hitratio_hexbin.pdf
#
# Every model with more than 100 requests and every user with more than 10,
# by request count and by the average of its per-request token hit ratios.
# ────────────────────────────────────────────────────────────────────────────

cache_file = CACHE_DIR / 'avg_hit_rate_by_user_and_model.parquet'
if not CACHED or not cache_file.exists():
    print('Computing the average per-request token hit ratio of each user …')
    df_user_avg = con.sql(f"""
        SELECT
            'user' AS entity_type,
            user_id AS entity_id,
            COUNT(*) AS n_requests,
            AVG(LEAST(GREATEST(CAST(ct AS DOUBLE) / it, 0.0), 1.0)) AS avg_hit_ratio
        FROM {TABLE_NAME}
        WHERE user_id IS NOT NULL
          AND ct IS NOT NULL
          AND it IS NOT NULL AND it > 0
        GROUP BY user_id
        HAVING COUNT(*) > 10
    """).fetchdf()

    print('Computing the average per-request token hit ratio of each model …')
    df_model_avg = con.sql(f"""
        SELECT
            'model' AS entity_type,
            chute_id AS entity_id,
            COUNT(*) AS n_requests,
            AVG(LEAST(GREATEST(CAST(ct AS DOUBLE) / it, 0.0), 1.0)) AS avg_hit_ratio
        FROM {TABLE_NAME}
        WHERE chute_id IS NOT NULL
          AND ct IS NOT NULL
          AND it IS NOT NULL AND it > 0
        GROUP BY chute_id
        HAVING COUNT(*) > 100
    """).fetchdf()

    pd.concat([df_user_avg, df_model_avg], ignore_index=True).to_parquet(cache_file)

df_avg_ratio = pd.read_parquet(cache_file)


def plot_avg_hitratio_hexbin(entity_type, entity_label, filename):
    sub = df_avg_ratio[df_avg_ratio['entity_type'] == entity_type]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    hb = ax.hexbin(
        sub['n_requests'].values,
        sub['avg_hit_ratio'].values,
        xscale='log',
        gridsize=40,
        cmap='viridis',
        mincnt=1,
        norm=LogNorm(),
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label(f'# {entity_label}', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel('# Requests')
    ax.set_ylabel('Avg. Token Hit Ratio')
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) / 10.0), numticks=100))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    style_grid(ax)
    save(fig, filename)


plot_avg_hitratio_hexbin('model', 'Models', 'fig_model_nreq_vs_avg_hitratio_hexbin.pdf')
plot_avg_hitratio_hexbin('user', 'Users', 'fig_user_nreq_vs_avg_hitratio_hexbin.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Inter-arrival times (IAT)
#
# The IAT of a request is the time since the previous request of the same user
# (monthly figure) or of the same (user, model) pair (the other two). The two
# queries over all users compute each entity's IAT percentiles p1..p100; they
# run in batches of users to bound DuckDB's memory, and give the busiest users
# a batch of their own.
# ────────────────────────────────────────────────────────────────────────────

IAT_PCTS = [i / 100 for i in range(1, 101)]
IAT_PCT_SQL = ', '.join(f'{p:.2f}' for p in IAT_PCTS)
IAT_PCT_COLS = [f'p{i}' for i in range(1, 101)]


def setup_log_x_iat(ax, xlim):
    """Log x axis with a labelled tick at every decade."""
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())


def setup_log_y_iat(ax, ylim):
    """Log y axis with a labelled tick at every decade."""
    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())


# Time landmarks for a log IAT x axis.
IAT_LANDMARKS = [
    (60,    '1m'),
    (3600,  '1h'),
    (43200, '12h'),
]


def annotate_iat_landmarks(ax, xlim):
    """Dotted lines at the landmarks inside xlim; every other label sits higher so they do not collide."""
    lo, hi = xlim
    in_range = [(v, lbl) for v, lbl in IAT_LANDMARKS if lo <= v <= hi]
    for i, (v, lbl) in enumerate(in_range):
        y = 0.04 + (0.08 if i % 2 == 1 else 0.0)
        ax.axvline(v, color='gray', linestyle=':', linewidth=0.6, alpha=0.55)
        ax.text(v, y, lbl, ha='center', va='bottom',
                fontsize=7, color='gray', alpha=0.55,
                transform=ax.get_xaxis_transform())


# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_iat_monthly_per_user_band.pdf
#
# Per-user IAT, computed separately in every month of the trace (month 1 is the
# trace's first). For each month the line is the median over users of each
# user's P50 IAT, and the bands run between the medians over users of each
# user's P10 and P90, and P25 and P75.
# Only users with more than 10 requests, and months in which a user has at
# least 10 IATs, count. Each batch of users writes its percentiles to its own
# parquet file under iat_monthly_per_user_batches/.
# ────────────────────────────────────────────────────────────────────────────

MONTHLY_BATCH_DIR = CACHE_DIR / 'iat_monthly_per_user_batches'
MONTHLY_CACHE = CACHE_DIR / 'iat_monthly_per_user.parquet'
MIN_REQUESTS = 10          # users need more requests than this
MIN_PER_MONTH_IAT = 10     # IATs a user needs in a month
TOP_INDIVIDUAL = 10        # the busiest users each get a batch of their own
USER_BATCH_SIZE = 50_000   # users per batch after them


def _run_batches():
    """Write each batch's per-(user, month) IAT percentiles; return the batch files."""
    print(f'Ranking eligible users (req > {MIN_REQUESTS}) ...')
    counts = con.sql(f"""
        SELECT user_id AS entity_id, COUNT(*) AS cnt
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL AND user_id IS NOT NULL
        GROUP BY 1
        HAVING COUNT(*) > {MIN_REQUESTS}
        ORDER BY cnt DESC
    """).fetchdf()
    print(f'  eligible users: {len(counts):,}')

    top_ids = counts['entity_id'].head(TOP_INDIVIDUAL).tolist()
    rest_ids = counts['entity_id'].iloc[TOP_INDIVIDUAL:].tolist()
    plan = [([uid], f'top_{i:02d}') for i, uid in enumerate(top_ids, 1)]
    plan += [(rest_ids[start:start + USER_BATCH_SIZE], f'batch_{k:05d}')
             for k, start in enumerate(range(0, len(rest_ids), USER_BATCH_SIZE), 1)]

    MONTHLY_BATCH_DIR.mkdir(exist_ok=True)
    paths = []
    for ids, tag in plan:
        path = MONTHLY_BATCH_DIR / f'{tag}.parquet'
        paths.append(path)
        if CACHED and path.exists():   # finished by an earlier, interrupted run
            continue
        rel = '__chunk_iat_user_monthly'
        con.register(rel, pd.DataFrame({'entity_id': ids}))
        con.sql(f"""
            COPY (
                SELECT
                    user_id AS entity_id,
                    month,
                    COUNT(*) AS n_iat,
                    APPROX_QUANTILE(iat, [{IAT_PCT_SQL}]) AS q
                FROM (
                    SELECT
                        t.user_id,
                        DATE_TRUNC('month', t.started_at)::DATE AS month,
                        EPOCH(t.started_at - LAG(t.started_at) OVER (
                            PARTITION BY t.user_id, DATE_TRUNC('month', t.started_at)
                            ORDER BY t.started_at
                        )) AS iat
                    FROM {TABLE_NAME} t
                    JOIN {rel} c ON t.user_id = c.entity_id
                    WHERE t.started_at IS NOT NULL AND t.user_id IS NOT NULL
                ) x
                WHERE iat IS NOT NULL AND iat > 0
                GROUP BY 1, 2
                HAVING COUNT(*) >= {MIN_PER_MONTH_IAT}
            ) TO '{path}' (FORMAT 'parquet')
        """)
        con.unregister(rel)
        print(f'  wrote {tag}: {len(ids):,} users -> {path.name}')
    return sorted(paths)


def _aggregate_across_batches(paths):
    """Per month and percentile p: the P25/P50/P75 over users of each user's p-th IAT percentile."""
    print(f'Aggregating cross-user P25/P50/P75 from {len(paths)} batches ...')
    files = ', '.join(f"'{p}'" for p in paths)
    df = con.sql(f"""
        WITH unnested AS (
            SELECT month, idx, q[idx] AS value
            FROM read_parquet([{files}]) AS t,
                 UNNEST(generate_series(1, 100)) AS s(idx)
            WHERE q IS NOT NULL
        )
        SELECT
            month,
            idx AS pct_idx,
            APPROX_QUANTILE(value, 0.25) AS p25_iat,
            APPROX_QUANTILE(value, 0.50) AS p50_iat,
            APPROX_QUANTILE(value, 0.75) AS p75_iat,
            COUNT(*) AS n_entries
        FROM unnested
        WHERE value IS NOT NULL AND value > 0
        GROUP BY month, idx
        ORDER BY month, idx
    """).fetchdf()
    df['pct'] = df['pct_idx'] / 100.0
    df['month'] = pd.to_datetime(df['month'])
    return df


if not CACHED or not MONTHLY_CACHE.exists():
    _aggregate_across_batches(_run_batches()).to_parquet(MONTHLY_CACHE)

# Median over users of each user's p-th percentile: one row per month, one column per p.
median_over_users = pd.read_parquet(MONTHLY_CACHE).pivot(
    index='month', columns='pct_idx', values='p50_iat')
month = trace_month(median_over_users.index)   # 1, 2, ...: the months of the trace
ylim = (10 ** (np.log10(median_over_users[10].min()) - 0.3),
        10 ** (np.log10(median_over_users[90].max()) + 0.3))

COLOR = '#1f77b4'
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for (p_lo, p_hi), alpha in (((10, 90), 0.10), ((25, 75), 0.25)):
    ax.fill_between(month, median_over_users[p_lo], median_over_users[p_hi],
                    color=COLOR, alpha=alpha, linewidth=0, label=f'P{p_lo}-P{p_hi}')
ax.plot(month, median_over_users[50], color=COLOR, linewidth=1.6, label='Median')
setup_log_y_iat(ax, ylim)
ax.set_xticks(month[::3])
ax.set_xlabel('Month')
ax.set_ylabel('Inter-arrival time (s)')
style_grid(ax)
ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=3,
          handlelength=1.4, columnspacing=0.8)
save(fig, 'fig_iat_monthly_per_user_band.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_per_user_iat_p50_by_model.pdf
#
# Boxplot over users of each user's median IAT to each highlighted model
# (pairs with at least 10 IATs).
# ────────────────────────────────────────────────────────────────────────────

MIN_PAIR_IAT = 10   # IATs a (user, model) pair needs

pair_cache = CACHE_DIR / "pair_iat_quantiles_by_model.parquet"
if not CACHED or not pair_cache.exists():
    print("Computing per-(model, user) IAT quantiles for the highlighted models …")
    hl_ids_sql = ", ".join(f"'{cid}'" for cid in df_hl["chute_id"])
    con.sql(f"""
        SELECT user_id, chute_id, COUNT(*) AS n_iat,
               quantile_cont(iat, 0.25) AS p25,
               quantile_cont(iat, 0.50) AS p50,
               quantile_cont(iat, 0.75) AS p75
        FROM (
            SELECT user_id, chute_id,
                   EPOCH(started_at - LAG(started_at) OVER (
                       PARTITION BY user_id, chute_id ORDER BY started_at
                   )) AS iat
            FROM {TABLE_NAME}
            WHERE started_at IS NOT NULL
              AND user_id IS NOT NULL
              AND chute_id IN ({hl_ids_sql})
        ) x
        WHERE iat IS NOT NULL AND iat > 0
        GROUP BY user_id, chute_id
        HAVING COUNT(*) >= {MIN_PAIR_IAT}
    """).fetchdf().to_parquet(pair_cache, index=False)

df_pair_iat = pd.read_parquet(pair_cache)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
for gi, (cid, color) in enumerate(zip(df_hl["chute_id"], df_hl["color"])):
    vals = df_pair_iat[df_pair_iat["chute_id"] == cid]["p50"].to_numpy()
    bp = ax.boxplot([vals],
                    positions=[gi],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False)
    for box in bp["boxes"]:
        box.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=0.7)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.1)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=color, linewidth=0.6)
ax.set_xticks(range(len(df_hl)))
ax.set_xticklabels(df_hl["label"].tolist(), rotation=25, ha="right")
ax.set_xlim(-0.6, len(df_hl) - 0.4)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_ylabel("Per-user IAT p50 (s)")
standard_ticks(ax)
style_grid(ax)
save(fig, "fig_per_user_iat_p50_by_model.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_iat_pair_ttl_coverage.pdf
#
# The share of repeat requests on a (user, model) pair that arrive within T
# seconds of the pair's previous request: the reuse a prefix cache with a
# time-to-live of T could capture. Every pair with at least 10 IATs, from
# users with more than 10 requests, contributes its IAT CDF (interpolated in
# log IAT between its percentiles p1..p100), weighted by its number of IATs,
# so that every repeat request counts once.
# ────────────────────────────────────────────────────────────────────────────

PAIR_IAT_CACHE = CACHE_DIR / 'iat_per_user_model_pair.parquet'
MIN_USER_REQS = 10         # users need more requests than this
TOP_USERS_INDIVIDUAL = 10  # the busiest users each get a query of their own
PAIR_BATCH_SIZE = 5_000    # users per query after them


def _compute_per_pair_iat():
    """One row per (user, model) pair: its IAT count, mean and percentiles p1..p100."""
    print('Ranking eligible users for (user, model) IAT ...')
    user_counts = con.sql(f"""
        SELECT user_id, COUNT(*) AS cnt
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL AND user_id IS NOT NULL AND chute_id IS NOT NULL
        GROUP BY user_id
        HAVING COUNT(*) > {MIN_USER_REQS}
        ORDER BY cnt DESC
    """).fetchdf()
    print(f'  eligible users: {len(user_counts):,}')

    top_ids = user_counts['user_id'].head(TOP_USERS_INDIVIDUAL).tolist()
    rest_ids = user_counts['user_id'].iloc[TOP_USERS_INDIVIDUAL:].tolist()
    parts = []

    def run_chunk(user_ids, chunk_label):
        rel = '__chunk_per_pair_iat'
        con.register(rel, pd.DataFrame({'user_id': user_ids}))
        part = con.sql(f"""
            SELECT
                user_id,
                chute_id,
                COUNT(*) AS n_iat,
                AVG(iat) AS mean_iat,
                APPROX_QUANTILE(iat, [{IAT_PCT_SQL}]) AS q
            FROM (
                SELECT
                    t.user_id,
                    t.chute_id,
                    EPOCH(t.started_at - LAG(t.started_at) OVER (
                        PARTITION BY t.user_id, t.chute_id ORDER BY t.started_at
                    )) AS iat
                FROM {TABLE_NAME} t
                JOIN {rel} c ON t.user_id = c.user_id
                WHERE t.started_at IS NOT NULL AND t.user_id IS NOT NULL AND t.chute_id IS NOT NULL
            ) x
            WHERE iat IS NOT NULL AND iat > 0
            GROUP BY user_id, chute_id
            HAVING COUNT(*) >= {MIN_PAIR_IAT}
        """).fetchdf()
        con.unregister(rel)
        parts.append(part)
        print(f'    {chunk_label}: {len(user_ids):,} users -> {len(part):,} pairs')

    for i, uid in enumerate(top_ids, 1):
        run_chunk([uid], f'top user {i}/{len(top_ids)}')
    n_batches = int(np.ceil(len(rest_ids) / PAIR_BATCH_SIZE))
    for start in range(0, len(rest_ids), PAIR_BATCH_SIZE):
        run_chunk(rest_ids[start:start + PAIR_BATCH_SIZE],
                  f'user batch {start // PAIR_BATCH_SIZE + 1}/{n_batches}')

    result = pd.concat(parts, ignore_index=True)
    expanded = pd.DataFrame(result['q'].tolist(), columns=IAT_PCT_COLS)
    expanded.insert(0, 'user_id', result['user_id'].values)
    expanded.insert(1, 'chute_id', result['chute_id'].values)
    expanded.insert(2, 'n_iat', result['n_iat'].values)
    expanded.insert(3, 'mean_iat', result['mean_iat'].values)
    return expanded


if not CACHED or not PAIR_IAT_CACHE.exists():
    _compute_per_pair_iat().to_parquet(PAIR_IAT_CACHE)

df_iat_pair = pd.read_parquet(PAIR_IAT_CACHE)

TTLS = np.logspace(-3, 4, 2000)
XLIM = (1e-3, 1e4)

log_ttls = np.log10(TTLS)
log_qs = np.log10(df_iat_pair[IAT_PCT_COLS].to_numpy(dtype=float))
pct_arr = np.asarray(IAT_PCTS, dtype=float)
print(f'Building TTL coverage from {len(log_qs):,} pairs × {len(TTLS)} TTLs ...')
pair_coverage = np.empty((len(log_qs), len(TTLS)), dtype=np.float32)
for i, row in enumerate(log_qs):
    pair_coverage[i, :] = np.interp(log_ttls, row, pct_arr, left=0.0, right=1.0)

weights = df_iat_pair['n_iat'].to_numpy(dtype=float)[:, None]
coverage_request_weighted = np.nansum(pair_coverage * weights, axis=0) / weights.sum()

fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot(TTLS, coverage_request_weighted,
        color='#d62728', linewidth=1.6, linestyle='-',
        label='Request')
setup_log_x_iat(ax, XLIM)
ax.set_ylim(0, 1.02)
ax.set_xlabel('IAT (s)')
ax.set_ylabel('CDF')
style_grid(ax)
annotate_iat_landmarks(ax, XLIM)
ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=1,
          handlelength=1.4, columnspacing=0.8)
save(fig, 'fig_iat_pair_ttl_coverage.pdf')
