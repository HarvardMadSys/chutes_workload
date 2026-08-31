#!/usr/bin/env python3
"""Section 6 — Prefix Caching: figure reproduction.

Reproduces every figure in the paper's Section 6 (*Prefix Caching*) from the
anonymized one-year trace. The code is the paper's own figure scripts; only
the paths and the cache toggles were adapted.

Figures produced (written to figures/paper/04_prefix_caching/):

1. fig_request_cached_fraction_cdf_by_input.pdf — per-request token hit ratio CDF, Global vs. <1K / >8K input cohorts
2. fig_request_token_hit_rate_cdf_modelhl.pdf   — per-request token hit ratio CDF, Global + highlighted models
   (filename matches the paper's include; axes/prose use "hit ratio")
3. fig_model_nreq_vs_avg_hitratio_hexbin.pdf    — hexbin: per-model request count vs. avg token hit ratio
4. fig_user_nreq_vs_avg_hitratio_hexbin.pdf     — hexbin: per-user request count vs. avg token hit ratio
5. fig_iat_monthly_per_user_band.pdf            — monthly per-user IAT, median with P10–P90 / P25–P75 bands
6. fig_per_user_iat_p50_by_model.pdf            — boxplot of per-user IAT p50 per highlighted model
7. fig_iat_pair_ttl_coverage.pdf                — request-weighted TTL coverage of (user, model) pair gaps

Section 6's two eviction-policy figures (token hit ratio vs. cache size) are
not drawn here: they come from libCacheSim rather than from the trace, and
`pipeline/caching/` runs that sweep and draws them.

Requirements: the anonymized one-year DuckDB trace (see --db, table
all_metrics_user), chute_models.csv next to <repo>/, plus Python packages
duckdb, pandas, numpy, matplotlib, and seaborn. Heavy queries cache their
results as parquet files under output/paper/cache/.

Usage:
    python 04_section6_prefix_caching.py [--db PATH] [--recompute]

By default, cached parquet files are reused when present (only plotting is
redone). Pass --recompute to rerun the heavy DuckDB queries from scratch.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import hashlib
import json
import re
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import (
    AutoMinorLocator, FuncFormatter, LogFormatterMathtext, LogLocator,
    MultipleLocator, NullFormatter, NullLocator,
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
# Setup: plotting style, output dirs, DB connection
#
# The trace is a DuckDB (table `all_metrics_user`; one row per API invocation
# with `it` = input tokens, `ot` = output tokens, `ct` = cached tokens,
# `started_at`, `user_id`, `chute_id`, …). chute_models.csv is the
# chute_id → model-name lookup, shipped in the repo root.
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


def standard_ticks(ax):
    """Major/minor tick lengths shared across the paper figures."""
    ax.tick_params(axis='both', which='major', length=4.5, width=0.8)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)


# Dashed grey reference style used for the 'Global' line in model-highlight figures.
GLOBAL_COLOR = '#444444'
GLOBAL_KW = dict(color=GLOBAL_COLOR, linestyle=(0, (4, 2)),
                 alpha=0.7, linewidth=1.5, label='Global')

# Generic 0..1 percentile grid (quantile queries)
PCTS = [i / 100 for i in range(101)]
PCT_SQL = ', '.join(f'{p:.2f}' for p in PCTS)

# IAT percentile grid: p1..p100, no p0
IAT_PCTS = [i / 100 for i in range(1, 101)]
IAT_PCT_SQL = ', '.join(f'{p:.2f}' for p in IAT_PCTS)
IAT_PCT_COLS = [f'p{i}' for i in range(1, 101)]

PDF_DIR, CACHE_DIR = section_paths("04_prefix_caching")
PDF_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(exist_ok=True)
con = duckdb.connect(DB_PATH, read_only=True)
con.sql("PRAGMA threads=96;")            # adjust to the local machine
con.sql("PRAGMA memory_limit='1000GB'")  # adjust to the local machine

TABLE_NAME = "all_metrics_user"
row_count = con.sql(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}").fetchone()[0]
print(f"Connected to DuckDB — {row_count:,} rows in {TABLE_NAME}")

# Manual chute_id → model name lookup, loaded from CSV (ships in the repo root)
chute_df = pd.read_csv(CHUTE_MODELS_CSV)
CHUTE_TO_MODEL = dict(zip(chute_df["chute_id"], chute_df["name"]))
print(f"Loaded {len(CHUTE_TO_MODEL):,} chute_id → model name mappings")

# ────────────────────────────────────────────────────────────────────────────
# Cache hit ratio distributions
#
# Per-request token hit ratio is ct / it (cached tokens over input tokens),
# clipped to [0, 1]. The two figures here show its CDF (i) split by
# input-length cohort and (ii) split by highlighted model, each against the
# global distribution.
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_request_cached_fraction_cdf_by_input.pdf
#
# Per-request token hit ratio CDF for all requests (Global) and for the
# <1K / >8K input-token cohorts. Quantile computation from the global-characterization figure script
# (Cacheability CDFs cell); plot styling from the model-highlight figure script
# (Token hit ratio CDF — Global + <1K / >8K input cohorts), which is the
# version whose output the paper includes.
# ────────────────────────────────────────────────────────────────────────────

# ── Per-request cached-fraction CDF, globally and by input-token bucket ──
# (computation from the global-characterization figure script; plot from the model-highlight figure script)
cached = CACHED
frac_cache_file = CACHE_DIR / '_cache_cacheability_request_cached_fraction_cdf_global_lt1k_gt8k.parquet'

if not cached or not frac_cache_file.exists():
    print('Computing global per-request cached-fraction CDF ...')
    global_q = con.sql(f"""
        SELECT APPROX_QUANTILE(cache_frac, [{PCT_SQL}]) AS q
        FROM (
            SELECT LEAST(GREATEST(CAST(ct AS DOUBLE) / it, 0.0), 1.0) AS cache_frac
            FROM {TABLE_NAME}
            WHERE ct IS NOT NULL
              AND it IS NOT NULL AND it > 0
        ) x
    """).fetchone()[0]

    print('Computing per-request cached-fraction CDF by input-token bucket ...')
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

    rows = []
    for pct, val in zip(PCTS, global_q):
        rows.append({'series': 'Global', 'pct': pct, 'cache_frac': val, 'n_requests': None})
    for _, row in df_bucket_q.iterrows():
        for pct, val in zip(PCTS, row['q']):
            rows.append({
                'series': row['input_bucket'],
                'pct': pct,
                'cache_frac': val,
                'n_requests': row['n_requests'],
            })
    df_cache_frac_cdf = pd.DataFrame(rows)
    df_cache_frac_cdf.to_parquet(frac_cache_file)
    print(f'  saved {len(df_cache_frac_cdf):,} CDF rows to {frac_cache_file}')
else:
    df_cache_frac_cdf = pd.read_parquet(frac_cache_file)
    print(f'Loaded {len(df_cache_frac_cdf):,} CDF rows from {frac_cache_file}')

COHORT_COLORS = {
    '<1K input': '#1f77b4',   # tab10 blue
    '>8K input': '#ff7f0e',   # tab10 orange
}
cohort_order = ['<1K input', '>8K input']

fig, ax = plt.subplots(figsize=(3.5, 3.0))

global_sub = (
    df_cache_frac_cdf[df_cache_frac_cdf['series'] == 'Global']
    .sort_values('pct')
)
if not global_sub.empty:
    ax.plot(global_sub['cache_frac'], global_sub['pct'], **GLOBAL_KW)

for series in cohort_order:
    sub = df_cache_frac_cdf[df_cache_frac_cdf['series'] == series].sort_values('pct')
    if sub.empty:
        continue
    ax.plot(sub['cache_frac'], sub['pct'],
            color=COHORT_COLORS[series], linewidth=1.5, label=series)

ax.set_xlabel('Token hit ratio')
ax.set_ylabel('CDF')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='lower right', frameon=False, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
standard_ticks(ax)
style_grid(ax)
fig.tight_layout(pad=0.4)
fig.savefig(PDF_DIR / 'fig_request_cached_fraction_cdf_by_input.pdf',
            bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved {PDF_DIR / 'fig_request_cached_fraction_cdf_by_input.pdf'}")

# ────────────────────────────────────────────────────────────────────────────
# Highlighted-model configuration
#
# Shared by the per-model token hit ratio CDF below and the per-user IAT
# boxplot later in this script. Copied from the model-highlight figure script
# (the standalone legend-PDF export at the end of the original cell is
# omitted — the figures here carry their own inline legends).
#
# Edit HIGHLIGHT_MODELS first. Keys are short labels used in figures. Provide
# a model_name and, when known, the chute_id. Missing chute_id values are
# resolved from chute_models.csv.
# ────────────────────────────────────────────────────────────────────────────

HIGHLIGHT_MODELS = {
    # ── RolePlay
    "DeepSeek V3.2": {
        "model_name": "deepseek-ai/DeepSeek-V3-0324-TEE",
        "chute_id": "0df3133d-c477-56d2-b4db-f2093bb150a1",
        "note": "Roleplay",
    },
    "Deepseek R1": {
        "model_name": "deepseek-ai/DeepSeek-R1-TEE",
        "chute_id": "6ff97a2a-ab6d-5a36-91e0-156339182e5f",
        "note": "Reasoning",
    },
    "Minimax M2.5": {
        "model_name": "minimax/Minimax-2.5-TEE",
        "chute_id": "ce6a92e4-5c2f-5681-9742-c80a4447bbdf",
        "note": "Coding/Agentic",
    },
}

SAVE_FIGURES = True

# ── Resolve & validate the selected models ──
NAME_TO_CHUTE = {v: k for k, v in CHUTE_TO_MODEL.items()}
hl_rows = []
for label, info in HIGHLIGHT_MODELS.items():
    chute_id = info.get("chute_id")
    model_name = info.get("model_name")
    if not chute_id and model_name in NAME_TO_CHUTE:
        chute_id = NAME_TO_CHUTE[model_name]
    if not chute_id:
        print(f"  WARN: no chute_id found for {label} ({model_name})")
        continue
    if chute_id not in CHUTE_TO_MODEL:
        print(f"  WARN: chute_id {chute_id} for {label} not in chute_models.csv")
    hl_rows.append({"label": label, "model_name": model_name, "chute_id": chute_id})

df_hl = pd.DataFrame(hl_rows)


def _highlight_model_hash(df):
    payload_cols = ["label", "model_name", "chute_id"]
    payload = df[payload_cols].fillna("").astype(str).to_dict("records")
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# The hash only depends on (label, model_name, chute_id), so it can be
# computed before the n_requests query — which lets the cached run below
# skip that DB scan entirely.
HIGHLIGHT_MODEL_HASH = _highlight_model_hash(df_hl)
HIGHLIGHT_CACHE_SUFFIX = f"models_{HIGHLIGHT_MODEL_HASH}"


def modelhl_cache_path(stem, ext=".parquet"):
    return CACHE_DIR / f"{stem}_{HIGHLIGHT_CACHE_SUFFIX}{ext}"


HL_PARQUET = modelhl_cache_path("_cache_modelhl_chutes")

if CACHED and HL_PARQUET.exists():
    df_hl = pd.read_parquet(HL_PARQUET)
    df_hl["color"] = df_hl["color"].apply(tuple)
    print(f"Loaded highlighted-model table ({len(df_hl)} models) from {HL_PARQUET.name}")
else:
    # Pull n_requests for each highlighted chute
    chunk_df = pd.DataFrame({"chute_id": df_hl["chute_id"].tolist()})
    con.register("__hl_chutes", chunk_df)
    try:
        counts = con.sql(f"""
            SELECT m.chute_id, COUNT(*) AS n_requests
            FROM {TABLE_NAME} m
            JOIN __hl_chutes h ON m.chute_id = h.chute_id
            GROUP BY 1
        """).fetchdf()
    finally:
        con.unregister("__hl_chutes")

    df_hl = df_hl.merge(counts, on="chute_id", how="left")
    df_hl["n_requests"] = df_hl["n_requests"].fillna(0).astype(int)

    PALETTE = plt.get_cmap("tab10").colors
    df_hl = df_hl.reset_index(drop=True)
    df_hl["color"] = [PALETTE[i % len(PALETTE)] for i in range(len(df_hl))]
    df_hl.to_parquet(HL_PARQUET, index=False)

HIGHLIGHT_PARQUET = str(HL_PARQUET)
print(f"Highlight cache suffix: {HIGHLIGHT_CACHE_SUFFIX}")
print(df_hl[["label", "model_name", "chute_id", "n_requests"]].to_string(index=False))

# Convenient lookups for plotting cells
HL_LABELS = df_hl["label"].tolist()
HL_CHUTES = df_hl["chute_id"].tolist()
HL_COLOR = dict(zip(df_hl["label"], df_hl["color"]))
HL_CHUTE_TO_LABEL = dict(zip(df_hl["chute_id"], df_hl["label"]))

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_request_token_hit_rate_cdf_modelhl.pdf
#
# CDF of the per-request token hit ratio for each highlighted model, with the
# Global distribution as the dashed grey reference, both computed directly
# from the trace. NOTE: the paper embeds a PNG screenshot of this plot
# (floats/z_highlight/image.png, x-axis labelled "Token hit rate"); this
# script saves the same plot as a proper vector PDF. Source:
# the model-highlight figure script (Per-request token hit ratio CDF per
# highlighted model).
# ────────────────────────────────────────────────────────────────────────────

# ── Per-request token hit ratio CDF per highlighted model ──
cached = CACHED
cache_file = modelhl_cache_path("_cache_modelhl_request_hit_rate_cdf")
global_cache = CACHE_DIR / "_cache_global_request_token_hit_ratio_cdf.parquet"

if not cached or not cache_file.exists():
    print("Computing per-request hit ratio quantiles per highlighted model …")
    df_q = con.sql(f"""
        SELECT m.chute_id,
               approx_quantile(LEAST(m.ct::DOUBLE / m.it, 1.0), [{PCT_SQL}]) AS q,
               COUNT(*) AS n
        FROM {TABLE_NAME} m
        JOIN read_parquet('{HIGHLIGHT_PARQUET}') h ON m.chute_id = h.chute_id
        WHERE m.it IS NOT NULL AND m.it > 0 AND m.ct IS NOT NULL AND m.ct >= 0
        GROUP BY 1
    """).fetchdf()
    rows = [{"chute_id": r["chute_id"], "pct": p, "cache_frac": r["q"][i], "n": r["n"]}
            for _, r in df_q.iterrows() for i, p in enumerate(PCTS)]
    pd.DataFrame(rows).to_parquet(cache_file)
    print(f"  saved → {cache_file}")

if not cached or not global_cache.exists():
    print("Computing global per-request token hit ratio quantiles …")
    global_q = con.sql(f"""
        SELECT approx_quantile(LEAST(ct::DOUBLE / it, 1.0), [{PCT_SQL}]) AS q
        FROM {TABLE_NAME}
        WHERE it IS NOT NULL AND it > 0 AND ct IS NOT NULL AND ct >= 0
    """).fetchone()[0]
    pd.DataFrame({"pct": PCTS, "cache_frac": list(global_q)}).to_parquet(global_cache)
    print(f"  saved → {global_cache}")

df_seg = pd.read_parquet(cache_file).merge(
    df_hl[["chute_id", "label", "color"]], on="chute_id", how="inner")
df_global = pd.read_parquet(global_cache).sort_values("pct")

fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot(df_global["cache_frac"], df_global["pct"], **GLOBAL_KW)
for _, r in df_hl.iterrows():
    sub = df_seg[df_seg["chute_id"] == r["chute_id"]].sort_values("cache_frac")
    if sub.empty:
        continue
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
fig.tight_layout(pad=0.4)
if SAVE_FIGURES:
    fig.savefig(PDF_DIR / "fig_request_token_hit_rate_cdf_modelhl.pdf",
                bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved {PDF_DIR / 'fig_request_token_hit_rate_cdf_modelhl.pdf'}")

# ────────────────────────────────────────────────────────────────────────────
# Hit ratio vs. request volume (hexbins)
#
# For every model (≥ 100 requests) and every user (≥ 10 requests): the average
# per-request token hit ratio, AVG(ct / it) clipped to [0, 1], against the
# entity's request count. Source: the global-characterization figure script (Hexbin heatmap:
# # requests vs AVG per-request token hit ratio, per entity). An older variant
# of these figures (fig_{model,user}_nreq_vs_hitrate_hexbin.pdf, using the
# cumulative SUM(ct)/SUM(it)) exists in the source notebook; the paper uses
# the avg_hitratio variant reproduced here. Both figures share one per-entity
# aggregate, computed once below.
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_model_nreq_vs_avg_hitratio_hexbin.pdf
# ────────────────────────────────────────────────────────────────────────────

# ── Hexbin heatmap: # requests vs AVG per-request token hit ratio, per entity ──
# Fresh SQL: average (ct/it) across requests for each user/model (no SUM/SUM).
cached = CACHED
avg_ratio_cache_file = CACHE_DIR / '_cache_cacheability_entity_avg_request_ratio.parquet'

if not cached or not avg_ratio_cache_file.exists():
    print('Computing per-user avg per-request token hit ratio ...')
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

    print('Computing per-model avg per-request token hit ratio ...')
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

    df_avg_ratio = pd.concat([df_user_avg, df_model_avg], ignore_index=True)
    df_avg_ratio.to_parquet(avg_ratio_cache_file)
    print(f'  saved {len(df_avg_ratio):,} entity rows to {avg_ratio_cache_file}')
else:
    df_avg_ratio = pd.read_parquet(avg_ratio_cache_file)
    print(f'Loaded {len(df_avg_ratio):,} entity rows from {avg_ratio_cache_file}')


def plot_avg_hitratio_hexbin(entity_type, entity_label, filename):
    sub = df_avg_ratio[df_avg_ratio['entity_type'] == entity_type].copy()
    sub = sub.dropna(subset=['avg_hit_ratio', 'n_requests'])
    sub = sub[(sub['avg_hit_ratio'] >= 0) & (sub['avg_hit_ratio'] <= 1)
              & (sub['n_requests'] > 0)]

    if sub.empty:
        print(f'Skipping {entity_type}: no data')
        return

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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    style_grid(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(PDF_DIR / filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Saved {PDF_DIR / filename}')


plot_avg_hitratio_hexbin('model', 'Models', 'fig_model_nreq_vs_avg_hitratio_hexbin.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_user_nreq_vs_avg_hitratio_hexbin.pdf
# ────────────────────────────────────────────────────────────────────────────

plot_avg_hitratio_hexbin('user', 'Users', 'fig_user_nreq_vs_avg_hitratio_hexbin.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Inter-arrival times and TTL coverage
#
# Three figures on how quickly requests repeat — which bounds what a prefix
# cache with a finite time-to-live (TTL) can reuse. The helpers and shared SQL
# constants below come from the setup cell of the IAT figure script.
# ────────────────────────────────────────────────────────────────────────────

# ── IAT helpers and shared constants (from the IAT figure script setup) ──

def setup_log_x_iat(ax, xlim):
    """Standard log-x styling for IAT plots — consistent ticks and grid."""
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())


def setup_log_y_iat(ax, ylim):
    """Log-y companion for IAT box-plot / shaded-line style figures."""
    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())


# Subtle landmarks on a log IAT axis. Only the landmarks that fall inside
# the axis range are drawn.
IAT_LANDMARKS = [
    (60,    '1m'),
    (3600,  '1h'),
    (43200, '12h'),
]


def annotate_iat_landmarks(ax, axis='x', xlim=None, ylim=None,
                           pos_frac=0.04, alt_offset=0.08,
                           fontsize=7, alpha=0.55,
                           enabled=True):
    """Draw subtle 1m / 1h / 12h markers on a log IAT axis.

    Adjacent landmarks are within ~0.3 decades of each other so labels would
    visually collide; the helper alternates perpendicular position by index
    so they stay readable. Pass enabled=False to skip drawing entirely.
    """
    if not enabled:
        return
    if axis == 'x':
        lo, hi = xlim if xlim else ax.get_xlim()
        in_range = [(v, lbl) for v, lbl in IAT_LANDMARKS if lo <= v <= hi]
        for i, (v, lbl) in enumerate(in_range):
            y = pos_frac + (alt_offset if i % 2 == 1 else 0.0)
            ax.axvline(v, color='gray', linestyle=':', linewidth=0.6, alpha=alpha)
            ax.text(v, y, lbl, ha='center', va='bottom',
                    fontsize=fontsize, color='gray', alpha=alpha,
                    transform=ax.get_xaxis_transform())
    else:
        lo, hi = ylim if ylim else ax.get_ylim()
        in_range = [(v, lbl) for v, lbl in IAT_LANDMARKS if lo <= v <= hi]
        for i, (v, lbl) in enumerate(in_range):
            x = (1 - pos_frac) - (alt_offset if i % 2 == 1 else 0.0)
            ax.axhline(v, color='gray', linestyle=':', linewidth=0.6, alpha=alpha)
            ax.text(x, v, lbl, ha='right', va='center',
                    fontsize=fontsize, color='gray', alpha=alpha,
                    transform=ax.get_yaxis_transform())


def shaded_line_at(ax, value_at_pct, positions, color,
                   bands=((0.10, 0.90), (0.25, 0.75)),
                   median_pct=0.50, line_kw=None):
    """Draw a P50 line + shaded P-band(s) from precomputed quantile lookups.

    Each entry in `bands` is a (low, high) percentile pair — earlier entries
    draw with lower alpha (outer band).
    """
    pos = np.asarray(positions, dtype=float)
    medians = np.array([value_at_pct(i, median_pct) for i in range(len(pos))])

    band_alphas = np.linspace(0.10, 0.25, len(bands)) if len(bands) > 0 else []
    for (p_lo, p_hi), alpha in zip(bands, band_alphas):
        lo = np.array([value_at_pct(i, p_lo) for i in range(len(pos))])
        hi = np.array([value_at_pct(i, p_hi) for i in range(len(pos))])
        ax.fill_between(pos, lo, hi, color=color, alpha=float(alpha),
                        linewidth=0,
                        label=f'P{int(p_lo*100)}-P{int(p_hi*100)}')

    line_kw = {'color': color, 'linewidth': 1.6, 'label': 'Median', **(line_kw or {})}
    ax.plot(pos, medians, **line_kw)


def tight_iat_ylim(value_at_pct, n, p_lo=0.10, p_hi=0.90, margin_decades=0.3):
    """Compute a tight log-y range bracketing P-low and P-high across positions."""
    los = np.array([value_at_pct(i, p_lo) for i in range(n)], dtype=float)
    his = np.array([value_at_pct(i, p_hi) for i in range(n)], dtype=float)
    los = los[np.isfinite(los) & (los > 0)]
    his = his[np.isfinite(his) & (his > 0)]
    if len(los) == 0 or len(his) == 0:
        return (1e-2, 1e2)
    lo = float(los.min())
    hi = float(his.max())
    return (10 ** (np.log10(lo) - margin_decades),
            10 ** (np.log10(hi) + margin_decades))


# ── Global cache toggle ──────────────────────────────────────────────────
# Master switch. In the source notebook each IAT cell also had its own local
# CACHED flag (default False); here the argparse-driven global CACHED plays
# that role, so the effective cache decision is (GLOBAL_CACHED and CACHED).
GLOBAL_CACHED = True


# ── Shared row-eligibility WHERE clauses ────────────────────────────────
# Use the unaliased variants in count/eligibility queries that read from
# TABLE_NAME directly. Use the `_T` variants inside the IAT subqueries that
# JOIN against a chunk relation and alias the data table as `t`.
WHERE_BASE    = "started_at IS NOT NULL"
WHERE_USER    = "started_at IS NOT NULL AND user_id IS NOT NULL"
WHERE_MODEL   = "started_at IS NOT NULL AND chute_id IS NOT NULL"
WHERE_PAIR    = "started_at IS NOT NULL AND user_id IS NOT NULL AND chute_id IS NOT NULL"

WHERE_BASE_T  = "t.started_at IS NOT NULL"
WHERE_USER_T  = "t.started_at IS NOT NULL AND t.user_id IS NOT NULL"
WHERE_MODEL_T = "t.started_at IS NOT NULL AND t.chute_id IS NOT NULL"
WHERE_PAIR_T  = "t.started_at IS NOT NULL AND t.user_id IS NOT NULL AND t.chute_id IS NOT NULL"

# Applied *after* the LAG() subquery — strips negative/null IAT gaps from
# partition boundaries before quantile aggregation.
WHERE_IAT_VALID = "iat IS NOT NULL AND iat > 0"

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_iat_monthly_per_user_band.pdf
#
# Per-user inter-arrival time computed independently for every calendar month,
# summarised as the cross-user P50 with P10–P90 / P25–P75 bands per month.
# Source: the IAT figure script (IAT evolution: monthly per-USER). The
# source cell also emits CDF and boxplot companions
# (fig_iat_monthly_per_user_cdfs.pdf, fig_iat_monthly_per_user_boxplot.pdf);
# only the band figure appears in the paper, so only that plot is kept here.
# The computation batches users to bound DuckDB memory and writes intermediate
# parquet files under cache/_iat_per_user_monthly_batches/.
# ────────────────────────────────────────────────────────────────────────────

# ── IAT evolution: monthly per-USER — shaded band ────────────────────────
# Repeats the per-user IAT calculation once per calendar month.

# ── Constants ────────────────────────────────────────────────────────────
BATCHES_DIR = CACHE_DIR / '_iat_per_user_monthly_batches'
CACHE_FILE = CACHE_DIR / '_iat_per_user_monthly_aggregated.parquet'
OUT_PDF_BAND = PDF_DIR / 'fig_iat_monthly_per_user_band.pdf'
COLOR = '#1f77b4'

SHOW_LANDMARKS_ON_BOX_AND_BAND = False

MIN_REQUESTS = 10
MIN_PER_MONTH_IAT = 10
TOP_INDIVIDUAL = 10
USER_BATCH_SIZE = 50_000
ENTITY_COL = 'user_id'

BATCHES_DIR.mkdir(parents=True, exist_ok=True)


def _run_batches():
    print(f'Ranking eligible users (req > {MIN_REQUESTS}) ...')
    counts = con.sql(f"""
        SELECT user_id AS entity_id, COUNT(*) AS cnt
        FROM {TABLE_NAME}
        WHERE {WHERE_USER}
        GROUP BY 1
        HAVING COUNT(*) > {MIN_REQUESTS}
        ORDER BY cnt DESC
    """).fetchdf()
    print(f'  eligible users: {len(counts):,}')

    top_ids = counts['entity_id'].head(TOP_INDIVIDUAL).tolist()
    rest_ids = counts['entity_id'].iloc[TOP_INDIVIDUAL:].tolist()

    plan = []
    for i, uid in enumerate(top_ids, 1):
        plan.append(([uid], f'top_{i:02d}'))
    n_batches = int(np.ceil(len(rest_ids) / USER_BATCH_SIZE)) if rest_ids else 0
    for k, start in enumerate(range(0, len(rest_ids), USER_BATCH_SIZE), 1):
        plan.append((rest_ids[start:start + USER_BATCH_SIZE], f'batch_{k:05d}'))

    paths = []
    for ids, tag in plan:
        path = BATCHES_DIR / f'{tag}.parquet'
        if path.exists():
            paths.append(path)
            continue
        chunk_df = pd.DataFrame({'entity_id': ids})
        rel = '__chunk_iat_user_monthly'
        con.register(rel, chunk_df)
        try:
            con.sql(f"""
                COPY (
                    SELECT
                        {ENTITY_COL} AS entity_id,
                        month,
                        COUNT(*) AS n_iat,
                        APPROX_QUANTILE(iat, [{IAT_PCT_SQL}]) AS q
                    FROM (
                        SELECT
                            t.{ENTITY_COL},
                            DATE_TRUNC('month', t.started_at)::DATE AS month,
                            EPOCH(t.started_at - LAG(t.started_at) OVER (
                                PARTITION BY t.{ENTITY_COL}, DATE_TRUNC('month', t.started_at)
                                ORDER BY t.started_at
                            )) AS iat
                        FROM {TABLE_NAME} t
                        JOIN {rel} c ON t.{ENTITY_COL} = c.entity_id
                        WHERE {WHERE_USER_T}
                    ) x
                    WHERE {WHERE_IAT_VALID}
                    GROUP BY 1, 2
                    HAVING COUNT(*) >= {MIN_PER_MONTH_IAT}
                ) TO '{path}' (FORMAT 'parquet')
            """)
        finally:
            con.unregister(rel)
        paths.append(path)
        print(f'  wrote {tag}: {len(ids):,} users -> {path.name}')
    return sorted(paths)


def _aggregate_across_batches(paths):
    glob_path = str(BATCHES_DIR / '*.parquet')
    print(f'Aggregating cross-user P25/P50/P75 from {len(paths)} batches ...')
    df = con.sql(f"""
        WITH unnested AS (
            SELECT month, idx, q[idx] AS value
            FROM read_parquet('{glob_path}') AS t,
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


# ── Compute / load ──────────────────────────────────────────────────────
if GLOBAL_CACHED and CACHED and CACHE_FILE.exists():
    df_monthly_user = pd.read_parquet(CACHE_FILE)
    print(f'Loaded {len(df_monthly_user):,} (month, pct) rows from {CACHE_FILE}')
else:
    paths = _run_batches()
    df_monthly_user = _aggregate_across_batches(paths)
    df_monthly_user.to_parquet(CACHE_FILE)
    print(f'Saved {len(df_monthly_user):,} (month, pct) rows to {CACHE_FILE}')

df_monthly_user['month'] = pd.to_datetime(df_monthly_user['month'])
months_sorted = sorted(df_monthly_user['month'].unique())


def _value_at_pct(i, p):
    sub = df_monthly_user[df_monthly_user['month'] == months_sorted[i]]
    match = sub.loc[np.isclose(sub['pct'], p), 'p50_iat']
    return float(match.iloc[0]) if len(match) else np.nan


YLIM_BOX = tight_iat_ylim(_value_at_pct, len(months_sorted), p_lo=0.10, p_hi=0.90)

# ── Plot: per-month shaded line ─────────────────────────────────────────
positions = np.arange(len(months_sorted))

fig_s, ax_s = plt.subplots(figsize=(3.5, 3.0))
shaded_line_at(ax_s, _value_at_pct, positions, COLOR)
setup_log_y_iat(ax_s, YLIM_BOX)
_tick_idx = list(range(0, len(months_sorted), 3))
ax_s.set_xticks([positions[i] for i in _tick_idx])
ax_s.set_xticklabels(
    [pd.Timestamp(months_sorted[i]).strftime("%b\n'%y") for i in _tick_idx]
)
ax_s.set_xlabel('Month')
ax_s.set_ylabel('Inter-arrival time (s)')
style_grid(ax_s)
annotate_iat_landmarks(ax_s, axis='y', ylim=YLIM_BOX,
                       enabled=SHOW_LANDMARKS_ON_BOX_AND_BAND)
ax_s.spines['top'].set_visible(False)
ax_s.spines['right'].set_visible(False)
ax_s.legend(loc='upper left', frameon=False, fontsize=7, ncol=3,
            handlelength=1.4, columnspacing=0.8)
fig_s.tight_layout(pad=0.4)
fig_s.savefig(OUT_PDF_BAND, bbox_inches='tight', dpi=300)
plt.close(fig_s)
print(f'Saved {OUT_PDF_BAND}')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_per_user_iat_p50_by_model.pdf
#
# Distribution (boxplot across users) of each user's median inter-arrival time
# to each of the three highlighted models. NOTE: the paper embeds a PNG
# screenshot of this plot (floats/useriatmodel.png, linear y-axis, "Per-user
# IAT p50 (s)"); this script saves the same plot as a proper vector PDF.
# Source: the model-highlight figure script (Per-(model, user) IAT
# distribution); the source cell also draws grouped p25/p50/p75 and log-y
# companions, which are omitted here — the paper uses the single-percentile
# p50 linear-y variant.
# ────────────────────────────────────────────────────────────────────────────

# ── Per-(model, user) IAT distribution — p50 boxplot per highlighted model ──
CACHE_PAIR_IAT = modelhl_cache_path("_cache_pair_iat_quantiles")
MIN_PAIR_IAT_HL = 10

_hl_ids_sql = ", ".join(f"'{cid}'" for cid in df_hl["chute_id"].tolist())

if GLOBAL_CACHED and CACHED and CACHE_PAIR_IAT.exists():
    df_pair_iat = pd.read_parquet(CACHE_PAIR_IAT)
    print(f"Loaded {len(df_pair_iat):,} (model, user) pairs from {CACHE_PAIR_IAT.name}")
else:
    print("Computing per-(model, user) IAT quantiles for the highlighted models …")
    df_pair_iat = con.sql(f"""
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
              AND chute_id IN ({_hl_ids_sql})
        ) x
        WHERE iat IS NOT NULL AND iat > 0
        GROUP BY user_id, chute_id
        HAVING COUNT(*) >= {MIN_PAIR_IAT_HL}
    """).fetchdf()
    df_pair_iat.to_parquet(CACHE_PAIR_IAT, index=False)
    print(f"Saved {len(df_pair_iat):,} (model, user) pairs to {CACHE_PAIR_IAT.name}")

print(df_pair_iat.groupby("chute_id")["user_id"].nunique()
      .rename("n_users").to_frame()
      .merge(df_hl.set_index("chute_id")[["label"]], left_index=True, right_index=True)
      .sort_values("n_users", ascending=False).to_string())

# ── Boxplot: per-user IAT p50, one box per highlighted model (linear y) ──
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for gi, (cid, color) in enumerate(zip(df_hl["chute_id"], df_hl["color"])):
    vals = df_pair_iat[df_pair_iat["chute_id"] == cid]["p50"].dropna().to_numpy()
    if vals.size == 0:
        continue
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
fig.tight_layout(pad=0.4)
if SAVE_FIGURES:
    fig.savefig(PDF_DIR / "fig_per_user_iat_p50_by_model.pdf",
                bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved {PDF_DIR / 'fig_per_user_iat_p50_by_model.pdf'}")

# ────────────────────────────────────────────────────────────────────────────
# Data prep: per-(user, model) pair IAT quantiles
#
# The TTL-coverage figure below consumes the 100 per-pair IAT percentiles of
# every (user, model) pair with ≥ 10 gaps. Source: the IAT figure script
# (IAT type 4: Per (user, model) pair); that cell's own figure
# (fig_iat_per_pair_cdf_bands.pdf) is not part of Section 6 and is omitted —
# only the computation is kept.
# ────────────────────────────────────────────────────────────────────────────

# ── IAT type 4: Per (user, model) pair — quantile computation ────────────
# Gap between consecutive requests for one (user, model) relationship.
# SQL partitions by (user_id, chute_id); one row per pair with that pair's
# 100 IAT percentiles.

CACHE_FILE = CACHE_DIR / '_cache_iat_per_user_model_pair_quantiles_fresh.parquet'

MIN_PAIR_IAT = 10
MIN_USER_REQS = 10
TOP_USERS_INDIVIDUAL = 10
USER_BATCH_SIZE = 5_000


def _compute_per_pair_iat():
    print('Ranking eligible users for (user, model) IAT ...')
    user_counts = con.sql(f"""
        SELECT user_id, COUNT(*) AS cnt
        FROM {TABLE_NAME}
        WHERE {WHERE_PAIR}
        GROUP BY user_id
        HAVING COUNT(*) > {MIN_USER_REQS}
        ORDER BY cnt DESC
    """).fetchdf()
    print(f'  eligible users: {len(user_counts):,}')

    top_ids = user_counts['user_id'].head(TOP_USERS_INDIVIDUAL).tolist()
    rest_ids = user_counts['user_id'].iloc[TOP_USERS_INDIVIDUAL:].tolist()
    parts = []

    def run_chunk(user_ids, chunk_label):
        if not user_ids:
            return
        chunk_df = pd.DataFrame({'user_id': user_ids})
        rel = '__chunk_per_pair_iat'
        con.register(rel, chunk_df)
        try:
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
                    WHERE {WHERE_PAIR_T}
                ) x
                WHERE {WHERE_IAT_VALID}
                GROUP BY user_id, chute_id
                HAVING COUNT(*) >= {MIN_PAIR_IAT}
            """).fetchdf()
        finally:
            con.unregister(rel)
        parts.append(part)
        print(f'    {chunk_label}: {len(user_ids):,} users -> {len(part):,} pairs')

    for i, uid in enumerate(top_ids, 1):
        run_chunk([uid], f'top user {i}/{len(top_ids)}')
    n_batches = int(np.ceil(len(rest_ids) / USER_BATCH_SIZE)) if rest_ids else 0
    for start in range(0, len(rest_ids), USER_BATCH_SIZE):
        run_chunk(rest_ids[start:start + USER_BATCH_SIZE],
                  f'user batch {start // USER_BATCH_SIZE + 1}/{n_batches}')

    if not parts:
        return pd.DataFrame(columns=['user_id', 'chute_id', 'n_iat', 'mean_iat'] + IAT_PCT_COLS)
    result = pd.concat(parts, ignore_index=True)
    expanded = pd.DataFrame(result['q'].tolist(), columns=IAT_PCT_COLS)
    expanded.insert(0, 'user_id', result['user_id'].values)
    expanded.insert(1, 'chute_id', result['chute_id'].values)
    expanded.insert(2, 'n_iat', result['n_iat'].values)
    expanded.insert(3, 'mean_iat', result['mean_iat'].values)
    return expanded


if GLOBAL_CACHED and CACHED and CACHE_FILE.exists():
    df_iat_pair = pd.read_parquet(CACHE_FILE)
    print(f'Loaded {len(df_iat_pair):,} pair rows from {CACHE_FILE}')
else:
    df_iat_pair = _compute_per_pair_iat()
    df_iat_pair.to_parquet(CACHE_FILE)
    print(f'Saved {len(df_iat_pair):,} pair rows to {CACHE_FILE}')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_iat_pair_ttl_coverage.pdf
#
# For a candidate cache TTL of T seconds, the fraction of repeat requests
# (gaps on (user, model) pairs, request-weighted) that would have found a live
# prefix — i.e. the operational token-reuse opportunity a TTL-bounded cache
# could capture. Source: the IAT figure script (TTL coverage on
# (user, model) pair gaps).
# ────────────────────────────────────────────────────────────────────────────

# ── TTL coverage on (user, model) pair gaps ──────────────────────────────
# Same data as type 4 (per-pair IAT), but presented as coverage(TTL):
# for a candidate cache TTL of T seconds, what fraction of repeat requests
# would have been a hit?
#
# Request-weighted: every gap counts equally (weighted by repeat volume) —
# this is the operational system-wide hit ratio the cache would see.

OUT_PDF = PDF_DIR / 'fig_iat_pair_ttl_coverage.pdf'
PAIR_CACHE = CACHE_DIR / '_cache_iat_per_user_model_pair_quantiles_fresh.parquet'
XLIM = (1e-3, 1e4)

if not PAIR_CACHE.exists():
    raise RuntimeError(f'Missing {PAIR_CACHE}; run the per-pair IAT step above first.')

df_pair = pd.read_parquet(PAIR_CACHE)

TTLS = np.logspace(-3, 4, 2000)
pct_arr = np.asarray(IAT_PCTS, dtype=float)

sub = df_pair.dropna(subset=IAT_PCT_COLS, how='all').copy()
quantiles = sub[IAT_PCT_COLS].astype(float).to_numpy()
n_iat_per_pair = sub['n_iat'].to_numpy(dtype=float)

log_ttls = np.log10(TTLS)
log_qs = np.log10(quantiles, where=quantiles > 0, out=np.full_like(quantiles, np.nan))

print(f'Building TTL coverage from {len(sub):,} pairs × {len(TTLS)} TTLs ...')
pair_coverage = np.empty((len(sub), len(TTLS)), dtype=np.float32)
for i, row in enumerate(log_qs):
    valid = np.isfinite(row)
    if not valid.any():
        pair_coverage[i, :] = np.nan
        continue
    pair_coverage[i, :] = np.interp(log_ttls, row[valid], pct_arr[valid],
                                    left=0.0, right=1.0)

weights = n_iat_per_pair[:, None]
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
annotate_iat_landmarks(ax, axis='x', xlim=XLIM)
ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=1,
          handlelength=1.4, columnspacing=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'Pairs aggregated: {len(sub):,}')
print(f'Saved {OUT_PDF}')

print("Done — Section 6 figures written to", PDF_DIR)
