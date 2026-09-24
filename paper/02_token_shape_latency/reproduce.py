#!/usr/bin/env python3
"""§3 Production Trace Analysis: the token-shape and latency figures.

Figures (written to figures/paper/02_token_shape_latency/):
  fig_token_length_input_cdf_by_model.pdf   CDF of input tokens per request
  fig_token_length_output_cdf_by_model.pdf  CDF of output tokens per request
  fig_output_input_ratio_cdf_by_model.pdf   CDF of output / input tokens per request
  fig_latency_e2e_cdf_by_model.pdf          CDF of end-to-end latency
  fig_prefill_fraction_cdf_by_model.pdf     CDF of TTFT / duration, the share of a request spent in prefill
  fig_duration_p50_heatmap.pdf              P50 duration over input x output token buckets
  fig_ttft_p50_heatmap_input_output.pdf     P50 TTFT over input x output token buckets
  fig_ttft_heatmap_input_cached.pdf         median TTFT over input tokens x token hit ratio

Each CDF shows all requests (dashed "Global") and one line per model in
config/highlight_models.json. Query results are cached in output/paper/cache/,
and all of them ship in data/paper/cache/, so a replot does not open the trace.
--recompute reruns every query against the trace.

Usage:
    python paper/02_token_shape_latency/reproduce.py [--db PATH] [--recompute]
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
from matplotlib.ticker import (
    AutoMinorLocator, LogFormatterMathtext, LogLocator, NullFormatter, NullLocator,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import LazyDB  # noqa: E402
from _paths import DB_DEFAULT, REPO, section_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the anonymized trace "
                         "(scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true", help="rerun the DuckDB queries even if their cache files exist")
args = parser.parse_args()
CACHED = not args.recompute   # True: reuse cache files and just replot

PDF_DIR, CACHE_DIR = section_paths("02_token_shape_latency")
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
    fig.savefig(PDF_DIR / fname, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {PDF_DIR / fname}")


# All requests: a dashed grey line under the per-model lines.
GLOBAL_KW = dict(color='#444444', linestyle=(0, (4, 2)),
                 alpha=0.7, linewidth=1.5, label='Global')

# The 101 percentiles 0.00, 0.01, ..., 1.00 that every CDF query computes.
PCTS = [i / 100 for i in range(101)]
PCT_SQL = ', '.join(f'{p:.2f}' for p in PCTS)

# ────────────────────────────────────────────────────────────────────────────
# Highlighted models
#
# config/highlight_models.json maps each figure label to a chute_id. The
# cached table adds each model's request count and its tab10 color, and the
# per-model queries join against it. After editing the config, rerun with
# --recompute.
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
print(df_hl[["label", "chute_id", "n_requests"]].to_string(index=False))


def plot_model_cdfs(df_global, df_models, col):
    """A new CDF figure of `col`: Global (dashed) and one line per highlighted model."""
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot(df_global[col], df_global["pct"], **GLOBAL_KW)
    for _, r in df_hl.iterrows():
        sub = df_models[df_models["chute_id"] == r["chute_id"]].sort_values("pct")
        ax.plot(sub[col], sub["pct"], color=r["color"], linewidth=1.5, label=r["label"])
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    return fig, ax


def plot_log_cdf(df_global, df_models, col, xlabel, fname, xlim):
    """plot_model_cdfs on a log x axis with a labelled tick at every decade."""
    fig, ax = plot_model_cdfs(df_global, df_models, col)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_xlim(*xlim)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    standard_ticks(ax)
    style_grid(ax)
    save(fig, fname)


# ────────────────────────────────────────────────────────────────────────────
# Figures: fig_token_length_input_cdf_by_model.pdf, fig_token_length_output_cdf_by_model.pdf
#
# CDFs of the input and output tokens of each completed request. The global
# table also holds the cached-token percentiles, which no figure plots.
# ────────────────────────────────────────────────────────────────────────────

global_cache = CACHE_DIR / "token_length_cdf_global.parquet"
if not CACHED or not global_cache.exists():
    print("Computing global token-length percentiles …")
    row = con.sql(f"""
        SELECT
            approx_quantile(it, [{PCT_SQL}]) AS input_q,
            approx_quantile(ot, [{PCT_SQL}]) AS output_q,
            approx_quantile(ct, [{PCT_SQL}]) AS cached_q
        FROM {TABLE_NAME}
        WHERE it IS NOT NULL AND it > 0 AND completed_at IS NOT NULL
    """).fetchone()
    pd.DataFrame({
        'pct': PCTS,
        'input_tokens': row[0],
        'output_tokens': row[1],
        'cached_tokens': row[2],
    }).to_parquet(global_cache)

model_cache = CACHE_DIR / "token_length_cdf_by_model.parquet"
if not CACHED or not model_cache.exists():
    print("Computing token-length percentiles per highlighted model …")
    df_q = con.sql(f"""
        SELECT m.chute_id,
               approx_quantile(m.it, [{PCT_SQL}]) AS input_q,
               approx_quantile(m.ot, [{PCT_SQL}]) AS output_q,
               COUNT(*) AS n
        FROM {TABLE_NAME} m
        JOIN read_parquet('{HL_PARQUET}') h ON m.chute_id = h.chute_id
        WHERE m.it IS NOT NULL AND m.it > 0
          AND m.completed_at IS NOT NULL
        GROUP BY 1
    """).fetchdf()
    rows = [{"chute_id": r["chute_id"], "pct": p,
             "input_tokens": r["input_q"][i],
             "output_tokens": r["output_q"][i],
             "n": r["n"]}
            for _, r in df_q.iterrows() for i, p in enumerate(PCTS)]
    pd.DataFrame(rows).to_parquet(model_cache)

df_global = pd.read_parquet(global_cache)
df_models = pd.read_parquet(model_cache)
plot_log_cdf(df_global, df_models, "input_tokens", "Input tokens",
             "fig_token_length_input_cdf_by_model.pdf", xlim=(1, 4e5))
plot_log_cdf(df_global, df_models, "output_tokens", "Output tokens",
             "fig_token_length_output_cdf_by_model.pdf", xlim=(1, 2e4))

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_output_input_ratio_cdf_by_model.pdf
#
# CDF of output / input tokens per completed request.
# ────────────────────────────────────────────────────────────────────────────

global_cache = CACHE_DIR / "ratio_cdf_global.parquet"
if not CACHED or not global_cache.exists():
    print("Computing global output/input ratio percentiles …")
    q = con.sql(f"""
        SELECT approx_quantile(ot::DOUBLE / it, [{PCT_SQL}]) AS ratio_q
        FROM {TABLE_NAME}
        WHERE completed_at IS NOT NULL
          AND it IS NOT NULL AND it > 0
          AND ot IS NOT NULL AND ot > 0
    """).fetchone()[0]
    pd.DataFrame({'pct': PCTS, 'output_input_ratio': q}).to_parquet(global_cache)

model_cache = CACHE_DIR / "ratio_cdf_by_model.parquet"
if not CACHED or not model_cache.exists():
    print("Computing output/input ratio percentiles per highlighted model …")
    df_q = con.sql(f"""
        SELECT m.chute_id,
               approx_quantile(m.ot::DOUBLE / m.it, [{PCT_SQL}]) AS ratio_q,
               COUNT(*) AS n
        FROM {TABLE_NAME} m
        JOIN read_parquet('{HL_PARQUET}') h ON m.chute_id = h.chute_id
        WHERE m.completed_at IS NOT NULL
          AND m.it IS NOT NULL AND m.it > 0
          AND m.ot IS NOT NULL AND m.ot > 0
        GROUP BY 1
    """).fetchdf()
    rows = [{"chute_id": r["chute_id"], "pct": p,
             "output_input_ratio": r["ratio_q"][i], "n": r["n"]}
            for _, r in df_q.iterrows() for i, p in enumerate(PCTS)]
    pd.DataFrame(rows).to_parquet(model_cache)

fig, ax = plot_model_cdfs(pd.read_parquet(global_cache), pd.read_parquet(model_cache),
                          "output_input_ratio")
ax.set_xscale("log")
ax.set_xlabel("Output / input tokens")
ax.set_xlim(5e-4, 1e2)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
standard_ticks(ax)
style_grid(ax)
save(fig, "fig_output_input_ratio_cdf_by_model.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_latency_e2e_cdf_by_model.pdf
#
# CDF of end-to-end latency (completed_at - started_at). The cached table
# also holds the prefill (TTFT) and decode percentiles; the paper plots
# end-to-end latency only.
# ────────────────────────────────────────────────────────────────────────────

cache_file = CACHE_DIR / "latency_cdf_by_model.parquet"
if not CACHED or not cache_file.exists():
    chute_list_sql = ", ".join(f"'{c}'" for c in df_hl["chute_id"])

    LATENCY_FILTER = """
        ttft IS NOT NULL AND ttft > 0
    AND started_at IS NOT NULL AND completed_at IS NOT NULL
    AND completed_at > started_at
    AND EPOCH(completed_at - started_at) > 0
    AND ttft <= EPOCH(completed_at - started_at)
    """

    LATENCY_QUANTILE_COLS = f"""
        approx_quantile(ttft, [{PCT_SQL}]) AS pf_q,
        approx_quantile(EPOCH(completed_at - started_at) - ttft, [{PCT_SQL}]) AS dc_q,
        approx_quantile(EPOCH(completed_at - started_at), [{PCT_SQL}]) AS e2_q,
        COUNT(*) AS n
    """

    print("Computing global latency percentiles …")
    g_row = con.sql(f"""
        SELECT {LATENCY_QUANTILE_COLS}
        FROM {TABLE_NAME}
        WHERE {LATENCY_FILTER}
    """).fetchone()

    print("Computing per-model latency percentiles …")
    df_q = con.sql(f"""
        SELECT chute_id, {LATENCY_QUANTILE_COLS}
        FROM {TABLE_NAME}
        WHERE {LATENCY_FILTER}
          AND chute_id IN ({chute_list_sql})
        GROUP BY chute_id
    """).fetchdf()

    def _expand(chute_id, pf_q, dc_q, e2_q):
        return [
            {"chute_id": chute_id, "pct": p,
             "prefill": pf_q[i], "decode": dc_q[i], "end_to_end": e2_q[i]}
            for i, p in enumerate(PCTS)
        ]

    rows = _expand("__GLOBAL__", g_row[0], g_row[1], g_row[2])
    for _, r in df_q.iterrows():
        rows.extend(_expand(r["chute_id"], r["pf_q"], r["dc_q"], r["e2_q"]))
    pd.DataFrame(rows).to_parquet(cache_file)

df_lat = pd.read_parquet(cache_file)
is_global = df_lat["chute_id"] == "__GLOBAL__"
plot_log_cdf(df_lat[is_global].sort_values("pct"), df_lat[~is_global], "end_to_end",
             "End-to-end latency (s)", "fig_latency_e2e_cdf_by_model.pdf", xlim=(1e-1, 3e2))

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_prefill_fraction_cdf_by_model.pdf
#
# CDF of TTFT / duration, the share of each request (under 300 s) spent in
# prefill.
# ────────────────────────────────────────────────────────────────────────────

cache_file = CACHE_DIR / "prefill_frac_cdf_by_model.parquet"
if not CACHED or not cache_file.exists():
    chute_list_sql = ", ".join(f"'{c}'" for c in df_hl["chute_id"])

    PREFILL_FRAC_FILTER = """
        ttft IS NOT NULL AND ttft > 0
    AND started_at IS NOT NULL AND completed_at IS NOT NULL
    AND completed_at > started_at
    AND EPOCH(completed_at - started_at) > 0
    AND EPOCH(completed_at - started_at) < 300
    AND ttft <= EPOCH(completed_at - started_at)
    """

    print("Computing global TTFT/duration percentiles …")
    g_q = con.sql(f"""
        SELECT approx_quantile(
                   ttft / EPOCH(completed_at - started_at), [{PCT_SQL}]
               ) AS q,
               COUNT(*) AS n
        FROM {TABLE_NAME}
        WHERE {PREFILL_FRAC_FILTER}
    """).fetchone()

    print("Computing per-model TTFT/duration percentiles …")
    df_q = con.sql(f"""
        SELECT chute_id,
               approx_quantile(
                   ttft / EPOCH(completed_at - started_at), [{PCT_SQL}]
               ) AS q,
               COUNT(*) AS n
        FROM {TABLE_NAME}
        WHERE {PREFILL_FRAC_FILTER}
          AND chute_id IN ({chute_list_sql})
        GROUP BY chute_id
    """).fetchdf()

    rows = [{"chute_id": "__GLOBAL__", "pct": p, "prefill_frac": g_q[0][i]}
            for i, p in enumerate(PCTS)]
    rows += [{"chute_id": r["chute_id"], "pct": p, "prefill_frac": r["q"][i]}
             for _, r in df_q.iterrows() for i, p in enumerate(PCTS)]
    pd.DataFrame(rows).to_parquet(cache_file)

df_pf = pd.read_parquet(cache_file)
is_global = df_pf["chute_id"] == "__GLOBAL__"
fig, ax = plot_model_cdfs(df_pf[is_global].sort_values("pct"), df_pf[~is_global], "prefill_frac")
ax.set_xlabel('TTFT / duration')
ax.set_xlim(0, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='lower right', frameon=False, fontsize=8)
standard_ticks(ax)
style_grid(ax)
save(fig, 'fig_prefill_fraction_cdf_by_model.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Latency heatmaps
#
# The same style as above with a lighter grid (grid.alpha 0.3).
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
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_duration_p50_heatmap.pdf
#
# P50 duration (completed_at - started_at, under 300 s) over 2K-input-token x
# 200-output-token buckets with at least 50 requests.
# ────────────────────────────────────────────────────────────────────────────

cache_file = CACHE_DIR / 'duration_p50_by_input_output.parquet'
if not CACHED or not cache_file.exists():
    print('Computing P50 duration heatmap (input × output) …')
    con.sql(f"""
        SELECT
            FLOOR(it / 2000) * 2  AS input_kb,
            FLOOR(ot / 200) * 200 AS output_bucket,
            COUNT(*)              AS n,
            APPROX_QUANTILE(EPOCH(completed_at - started_at), 0.50) AS dur_p50
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL AND completed_at IS NOT NULL
          AND completed_at > started_at
          AND it > 0 AND ot > 0
          AND EPOCH(completed_at - started_at) < 300
        GROUP BY 1, 2
        HAVING COUNT(*) >= 50
    """).fetchdf().to_parquet(cache_file)

df_duration = pd.read_parquet(cache_file)
df_plot = df_duration[(df_duration['input_kb'] <= 32) & (df_duration['output_bucket'] <= 8000)]
pivot = df_plot.pivot_table(index='output_bucket', columns='input_kb',
                            values='dur_p50', aggfunc='first')
pivot = pivot.sort_index(ascending=False)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                   cmap='YlOrRd', shading='auto', rasterized=True)
ax.set_xlabel('Input tokens (K)')
ax.set_ylabel('Output tokens')
fig.colorbar(im, ax=ax, label='P50 duration (s)', shrink=0.85)
save(fig, 'fig_duration_p50_heatmap.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_ttft_p50_heatmap_input_output.pdf
#
# P50 TTFT (under 300 s) over input x output token buckets.
# ────────────────────────────────────────────────────────────────────────────

INPUT_BIN = 2000      # input tokens per bucket
OUTPUT_BIN = 200      # output tokens per bucket
MIN_CELL_COUNT = 50   # requests a bucket needs
MAX_INPUT_K = 32      # plotted range, in K input tokens
MAX_OUTPUT = 8000     # plotted range, in output tokens

cache_file = CACHE_DIR / 'ttft_p50_by_input_output.parquet'
if not CACHED or not cache_file.exists():
    print('Computing P50 TTFT heatmap (input × output) …')
    con.sql(f"""
        SELECT
            FLOOR(it / {INPUT_BIN}) * ({INPUT_BIN} / 1000) AS input_kb,
            FLOOR(ot / {OUTPUT_BIN}) * {OUTPUT_BIN} AS output_bucket,
            COUNT(*) AS n,
            APPROX_QUANTILE(ttft, 0.50) AS ttft_p50
        FROM {TABLE_NAME}
        WHERE ttft IS NOT NULL AND ttft > 0 AND ttft < 300
          AND it IS NOT NULL AND it > 0
          AND ot IS NOT NULL AND ot > 0
        GROUP BY 1, 2
        HAVING COUNT(*) >= {MIN_CELL_COUNT}
    """).fetchdf().to_parquet(cache_file)

df_ttft_io = pd.read_parquet(cache_file)
df_plot = df_ttft_io[(df_ttft_io['input_kb'] <= MAX_INPUT_K)
                     & (df_ttft_io['output_bucket'] <= MAX_OUTPUT)]
pivot = df_plot.pivot_table(index='output_bucket', columns='input_kb',
                            values='ttft_p50', aggfunc='first')
pivot = pivot.sort_index(ascending=False)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                   cmap='YlOrRd', shading='auto', rasterized=True)
ax.set_xlabel('Input tokens (K)')
ax.set_ylabel('Output tokens')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)
fig.colorbar(im, ax=ax, label='P50 TTFT (s)', shrink=0.85)
save(fig, 'fig_ttft_p50_heatmap_input_output.pdf')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_ttft_heatmap_input_cached.pdf
#
# Median TTFT (under 300 s) over 2K-input-token buckets x token hit ratio
# (ct / it, in tenths) buckets with at least 200 requests: how prefix caching
# shifts TTFT at each input length.
# ────────────────────────────────────────────────────────────────────────────

cache_file = CACHE_DIR / 'ttft_by_input_cached.parquet'
if not CACHED or not cache_file.exists():
    print('Computing TTFT heatmap (input × token hit ratio) …')
    con.sql(f"""
        WITH base AS (
            SELECT
                ttft,
                it,
                CAST(ct AS DOUBLE) / NULLIF(it, 0) AS cache_frac
            FROM {TABLE_NAME}
            WHERE ttft IS NOT NULL AND ttft > 0 AND ttft < 300
              AND it > 0 AND ct IS NOT NULL
        )
        SELECT
            FLOOR(it / 2000) * 2                      AS input_kb,
            FLOOR(LEAST(cache_frac, 1.0) * 10) / 10.0 AS cache_frac_bin,
            COUNT(*)                                  AS n,
            APPROX_QUANTILE(ttft, 0.50)               AS ttft_p50,
            APPROX_QUANTILE(ttft, 0.95)               AS ttft_p95
        FROM base
        WHERE cache_frac >= 0
        GROUP BY 1, 2
        HAVING COUNT(*) >= 200
    """).fetchdf().to_parquet(cache_file)

df_ttft_cached = pd.read_parquet(cache_file)
df_plot = df_ttft_cached[df_ttft_cached['input_kb'] <= 32]
pivot = df_plot.pivot_table(index='cache_frac_bin', columns='input_kb',
                            values='ttft_p50', aggfunc='first')
pivot = pivot.sort_index(ascending=True)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                   cmap='YlOrRd', shading='auto', rasterized=True)
ax.set_xlabel('Input tokens (K)')
ax.set_ylabel('Token hit ratio')
fig.colorbar(im, ax=ax, label='Median TTFT (s)', shrink=0.85)
save(fig, 'fig_ttft_heatmap_input_cached.pdf')
