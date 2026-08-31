#!/usr/bin/env python3
"""Section 3 — Global Characterization: figure reproduction.

Regenerates every figure the paper's Section 3 ("Global Characterization")
includes, from the anonymized one-year trace (DuckDB). The code is the
paper's own figure scripts — the model-highlight CDFs and the latency
heatmaps — with only the paths adapted.

Figures (written to figures/paper/02_token_shape_latency/):
  fig_token_length_input_cdf_modelhl.pdf   CDF of per-request input token length — global + highlighted models
  fig_token_length_output_cdf_modelhl.pdf  CDF of per-request output token length — global + highlighted models
  fig_output_input_ratio_cdf_modelhl.pdf   CDF of the per-request output/input token ratio — global + highlighted models
  fig_latency_e2e_cdf_modelhl.pdf          CDF of end-to-end request latency — global + highlighted models
  fig_prefill_fraction_cdf_modelhl.pdf     CDF of TTFT / total duration (prefill fraction) — global + highlighted models
  fig17c_duration_p50_heatmap.pdf          P50 request duration heatmap over input × output token buckets
  fig_ttft_p50_heatmap_input_output.pdf    P50 TTFT heatmap over input × output token buckets
  fig16a_ttft_heatmap_input_cache.pdf      Median TTFT heatmap over input tokens × per-request token hit ratio
  fig_modelhl_legend.pdf                   shared per-model legend (with the Global entry)
  fig_modelhl_legend_no_global.pdf         shared per-model legend (models only)

Intermediate query results land in output/paper/cache/. Heavy DuckDB queries
are skipped on reruns when their cache file exists; pass --recompute to force
them to rerun against the trace.

Usage:
    python 02_section3_global_characterization.py [--db PATH] [--recompute]
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
import hashlib
import json
from pathlib import Path
from matplotlib.ticker import (
    AutoMinorLocator, NullLocator, NullFormatter, FuncFormatter,
    LogLocator, LogFormatterMathtext,
)
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

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
# Only the trace database is required; chute_models.csv (shipped at the repo
# root) is an optional chute_id → model-name lookup used for validation prints.
# ────────────────────────────────────────────────────────────────────────────

PDF_DIR, CACHE_DIR = section_paths("02_token_shape_latency")
PDF_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(exist_ok=True)
con = duckdb.connect(DB_PATH, read_only=True)
con.sql("PRAGMA threads=96;")
con.sql("PRAGMA memory_limit='1000GB'")
TABLE_NAME = "all_metrics_user"

row_count = con.sql(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}").fetchone()[0]
print(f"Connected to DuckDB — {row_count:,} rows in {TABLE_NAME}")

# Optional chute_id → model name lookup. The highlighted models below carry
# explicit chute_ids, so the script still runs when this CSV is absent.
CHUTE_TO_MODEL = {}
try:
    chute_df = pd.read_csv(CHUTE_MODELS_CSV)  # ships at the repo root
    CHUTE_TO_MODEL = dict(zip(chute_df["chute_id"], chute_df["name"]))
    print(f"Loaded {len(CHUTE_TO_MODEL):,} chute_id → model name mappings")
except FileNotFoundError:
    print("chute_models.csv not found — continuing with explicit chute_ids only")

# ────────────────────────────────────────────────────────────────────────────
# Token & latency CDFs with model highlights
#
# Helpers, matplotlib style, and the highlighted-model selection, copied from
# the model-highlight figure script. Every CDF below overlays a dashed
# "Global" line (all requests) with one colored line per highlighted model.
# ────────────────────────────────────────────────────────────────────────────

# ── Plot style & helpers (from the model-highlight figure script) ──
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


def save_legend(handles, labels, path, ncol=None, figsize=None):
    if ncol is None:
        ncol = len(labels)
    if figsize is None:
        figsize = (max(1.5, 1.1 * len(labels)) + 0.5, 0.5)
    fig_l = plt.figure(figsize=figsize)
    fig_l.legend(handles, labels, loc='center', ncol=ncol, frameon=False,
                 columnspacing=1.2, handlelength=1.8)
    fig_l.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig_l)


def standard_ticks(ax):
    """Major/minor tick lengths matching notebooks 1/2."""
    ax.tick_params(axis='both', which='major', length=4.5, width=0.8)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)


GLOBAL_COLOR = '#444444'
GLOBAL_KW = dict(color=GLOBAL_COLOR, linestyle=(0, (4, 2)),
                 alpha=0.7, linewidth=1.5, label='Global')

# Generic 0..1 percentile grid used by the quantile queries
PCTS = [i / 100 for i in range(101)]
PCT_SQL = ', '.join(f'{p:.2f}' for p in PCTS)


def make_model_legend_handles(df_hl, include_global=True):
    """Return (handles, labels) for the shared per-model legend."""
    handles, labels = [], []
    if include_global:
        handles.append(Line2D([0], [0], color=GLOBAL_COLOR,
                              linestyle=(0, (4, 2)), linewidth=1.5))
        labels.append('Global')
    for _, r in df_hl.iterrows():
        handles.append(Line2D([0], [0], color=r['color'], linewidth=1.5))
        labels.append(r['label'])
    return handles, labels


# ── Highlighted-model selection (edit this block first) ──
# Keys are short labels used in figures. Provide a model_name and, when known,
# the chute_id. The script resolves missing chute_id values from chute_models.csv.

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
import matplotlib.pyplot as _plt

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


HIGHLIGHT_MODEL_HASH = _highlight_model_hash(df_hl)
HIGHLIGHT_CACHE_SUFFIX = f"models_{HIGHLIGHT_MODEL_HASH}"


def modelhl_cache_path(stem, ext=".parquet"):
    return CACHE_DIR / f"{stem}_{HIGHLIGHT_CACHE_SUFFIX}{ext}"


HL_PARQUET = modelhl_cache_path("_cache_modelhl_chutes")

if CACHED and HL_PARQUET.exists():
    # Replot path: reuse the resolved highlight table (n_requests + colors)
    df_hl = pd.read_parquet(HL_PARQUET)
    print(f"Loaded highlighted-model table from {HL_PARQUET}")
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

    PALETTE = _plt.get_cmap("tab10").colors
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

# ── Save the shared legend PDFs once (companions to the per-figure legends) ──
_hdls_g, _lbls_g = make_model_legend_handles(df_hl, include_global=True)
_hdls, _lbls = make_model_legend_handles(df_hl, include_global=False)

save_legend(_hdls_g, _lbls_g, PDF_DIR / "fig_modelhl_legend.pdf",
            ncol=min(len(_lbls_g), 4))
save_legend(_hdls, _lbls, PDF_DIR / "fig_modelhl_legend_no_global.pdf",
            ncol=min(len(_lbls), 4))
print(f"Saved shared legends to {PDF_DIR}/fig_modelhl_legend{{,_no_global}}.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_token_length_input_cdf_modelhl.pdf
#
# Per-request input token-length CDF. The Global percentiles use the same
# query as the workload-overview figure script and are built here when the cache file is absent.
# This block also prepares the data and helper reused by the output-token
# figure below.
# ────────────────────────────────────────────────────────────────────────────

# ── Token-length CDFs per highlighted model ──
cached = CACHED
cache_file = modelhl_cache_path("_cache_modelhl_token_length_cdf")
global_cache = CACHE_DIR / "_cache_token_length_cdfs.parquet"

# Global token-length percentiles (query copied from the workload-overview figure script) — build if absent
if not global_cache.exists():
    print('Computing global token length percentiles …')
    row = con.sql(f"""
        SELECT
            approx_quantile(it,  [{PCT_SQL}]) AS input_q          ,
            approx_quantile(ot,  [{PCT_SQL}]) AS output_q          ,
            approx_quantile(ct,            [{PCT_SQL}]) AS cached_q
        FROM {TABLE_NAME}
        WHERE it IS NOT NULL AND it > 0 and completed_at IS NOT NULL
    """).fetchone()
    pd.DataFrame({
        'pct': PCTS,
        'input_tokens': row[0],
        'output_tokens': row[1],
        'cached_tokens': row[2],
    }).to_parquet(global_cache)
    print(f'  saved → {global_cache}')

if not cached or not cache_file.exists():
    print("Computing token-length percentiles per highlighted model …")
    df_q = con.sql(f"""
        SELECT m.chute_id,
               approx_quantile(m.it, [{PCT_SQL}]) AS input_q,
               approx_quantile(m.ot, [{PCT_SQL}]) AS output_q,
               COUNT(*) AS n
        FROM {TABLE_NAME} m
        JOIN read_parquet('{HIGHLIGHT_PARQUET}') h ON m.chute_id = h.chute_id
        WHERE m.it IS NOT NULL AND m.it > 0
          AND m.completed_at IS NOT NULL
        GROUP BY 1
    """).fetchdf()
    rows = [{"chute_id": r["chute_id"], "pct": p,
             "input_tokens": r["input_q"][i],
             "output_tokens": r["output_q"][i],
             "n": r["n"]}
            for _, r in df_q.iterrows() for i, p in enumerate(PCTS)]
    pd.DataFrame(rows).to_parquet(cache_file)
    print(f"  saved → {cache_file}")

df_seg = pd.read_parquet(cache_file).merge(
    df_hl[["chute_id", "label", "color"]], on="chute_id", how="inner")
df_global = pd.read_parquet(global_cache) if global_cache.exists() else None


def _plot_token_cdf(col, xlabel, fname, xlim):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    if df_global is not None:
        ax.plot(df_global[col], df_global["pct"], **GLOBAL_KW)
    for _, r in df_hl.iterrows():
        sub = df_seg[df_seg["chute_id"] == r["chute_id"]].sort_values("pct")
        if sub.empty:
            continue
        ax.plot(sub[col], sub["pct"],
                color=r["color"], linewidth=1.5, label=r["label"])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    standard_ticks(ax)
    style_grid(ax)
    fig.tight_layout(pad=0.4)
    if SAVE_FIGURES:
        fig.savefig(PDF_DIR / fname, bbox_inches="tight", dpi=300)
        print(f"Saved {PDF_DIR / fname}")
    plt.close(fig)


_plot_token_cdf("input_tokens",  "Input tokens",
                "fig_token_length_input_cdf_modelhl.pdf",  xlim=(1, 4e5))

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_token_length_output_cdf_modelhl.pdf
#
# Per-request output token-length CDF — reuses df_seg / df_global /
# _plot_token_cdf prepared in the previous block.
# ────────────────────────────────────────────────────────────────────────────

_plot_token_cdf("output_tokens", "Output tokens",
                "fig_token_length_output_cdf_modelhl.pdf", xlim=(1, 2e4))

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_output_input_ratio_cdf_modelhl.pdf
#
# CDF of the per-request output/input token ratio. The Global percentiles use
# the same query as the workload-overview figure script and are built here when the cache file is
# absent.
# ────────────────────────────────────────────────────────────────────────────

# ── Output / input ratio CDF per highlighted model ──
cached = CACHED
cache_file = modelhl_cache_path("_cache_modelhl_ratio_cdf")
global_cache = CACHE_DIR / "global_output_input_ratio_cdf.parquet"

# Global ratio percentiles (query copied from the workload-overview figure script) — build if absent
if not global_cache.exists():
    print('Computing global output/input ratio percentiles ...')
    q = con.sql(f"""
        SELECT approx_quantile(ot::DOUBLE / it, [{PCT_SQL}]) AS ratio_q
        FROM {TABLE_NAME}
        WHERE completed_at IS NOT NULL
          AND it IS NOT NULL AND it > 0
          AND ot IS NOT NULL AND ot > 0
    """).fetchone()[0]
    pd.DataFrame({
        'pct': PCTS,
        'output_input_ratio': q,
    }).to_parquet(global_cache)
    print(f'  saved → {global_cache}')

if not cached or not cache_file.exists():
    print("Computing output/input ratio percentiles per highlighted model …")
    df_q = con.sql(f"""
        SELECT m.chute_id,
               approx_quantile(m.ot::DOUBLE / m.it, [{PCT_SQL}]) AS ratio_q,
               COUNT(*) AS n
        FROM {TABLE_NAME} m
        JOIN read_parquet('{HIGHLIGHT_PARQUET}') h ON m.chute_id = h.chute_id
        WHERE m.completed_at IS NOT NULL
          AND m.it IS NOT NULL AND m.it > 0
          AND m.ot IS NOT NULL AND m.ot > 0
        GROUP BY 1
    """).fetchdf()
    rows = [{"chute_id": r["chute_id"], "pct": p,
             "output_input_ratio": r["ratio_q"][i], "n": r["n"]}
            for _, r in df_q.iterrows() for i, p in enumerate(PCTS)]
    pd.DataFrame(rows).to_parquet(cache_file)
    print(f"  saved → {cache_file}")

df_seg = pd.read_parquet(cache_file).merge(
    df_hl[["chute_id", "label", "color"]], on="chute_id", how="inner")
df_global = pd.read_parquet(global_cache) if global_cache.exists() else None

fig, ax = plt.subplots(figsize=(3.2, 3.0))
if df_global is not None:
    ax.plot(df_global["output_input_ratio"], df_global["pct"], **GLOBAL_KW)
for _, r in df_hl.iterrows():
    sub = df_seg[df_seg["chute_id"] == r["chute_id"]].sort_values("pct")
    if sub.empty:
        continue
    ax.plot(sub["output_input_ratio"], sub["pct"],
            color=r["color"], linewidth=1.5, label=r["label"])
ax.set_xscale("log")
ax.set_xlabel("Output / input tokens")
ax.set_ylabel("CDF")
ax.set_ylim(0, 1)
ax.set_xlim(5e-4, 1e2)
# legend
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
standard_ticks(ax)
style_grid(ax)
fig.tight_layout(pad=0.4)
if SAVE_FIGURES:
    fig.savefig(PDF_DIR / "fig_output_input_ratio_cdf_modelhl.pdf",
                bbox_inches="tight", dpi=300)
    print(f"Saved {PDF_DIR / 'fig_output_input_ratio_cdf_modelhl.pdf'}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_latency_e2e_cdf_modelhl.pdf
#
# End-to-end latency CDF, global plus highlighted models. Direct queries
# against the trace in the source notebook (no cache there); the script caches
# the expanded quantile table so reruns replot without touching the DB. The
# same query also yields the prefill (TTFT) and decode quantiles, of which the
# paper's Section 3 uses the end-to-end variant.
# ────────────────────────────────────────────────────────────────────────────

# ── Latency CDFs (prefill / decode / e2e): Global + highlighted models ──
cached = CACHED
cache_file = modelhl_cache_path("_cache_modelhl_latency_cdf")

if not cached or not cache_file.exists():
    # Two direct queries: one for the global aggregate, one per highlighted model.
    chute_ids = df_hl["chute_id"].tolist()
    chute_list_sql = ", ".join(f"'{c}'" for c in chute_ids)

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
    df_lat = pd.DataFrame(rows)
    df_lat.to_parquet(cache_file)
    print(f"  saved → {cache_file}")
else:
    df_lat = pd.read_parquet(cache_file)
    print(f"Loaded latency percentiles from {cache_file}")

df_global = df_lat[df_lat["chute_id"] == "__GLOBAL__"].sort_values("pct")
df_seg = df_lat[df_lat["chute_id"] != "__GLOBAL__"].merge(
    df_hl[["chute_id", "label", "color"]], on="chute_id", how="inner")


def _plot_latency_cdf(col, xlabel, fname, xlim=(1e-2, 3e2)):
    fig, ax = plt.subplots(figsize=(3.2, 3))
    if not df_global.empty:
        ax.plot(df_global[col], df_global["pct"], **GLOBAL_KW)
    for _, r in df_hl.iterrows():
        sub = df_seg[df_seg["chute_id"] == r["chute_id"]].sort_values("pct")
        if sub.empty:
            continue
        ax.plot(sub[col], sub["pct"],
                color=r["color"], linewidth=1.5, label=r["label"])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    standard_ticks(ax)
    style_grid(ax)
    fig.tight_layout(pad=0.4)
    if SAVE_FIGURES:
        fig.savefig(PDF_DIR / fname, bbox_inches="tight", dpi=300)
        print(f"Saved {PDF_DIR / fname}")
    plt.close(fig)


# Section 3 uses the end-to-end variant. (The source notebook also plots the
# prefill and decode companions with the same helper.)
_plot_latency_cdf("end_to_end", "End-to-end latency (s)", "fig_latency_e2e_cdf_modelhl.pdf", xlim=(1e-1, 3e2))

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_prefill_fraction_cdf_modelhl.pdf
#
# CDF of TTFT / total duration — the fraction of each request spent in
# prefill — global plus highlighted models. Direct query in the source
# notebook; the script caches the quantile table so reruns just replot.
# ────────────────────────────────────────────────────────────────────────────

# ── TTFT / Duration CDF: Global + highlighted models ──
cached = CACHED
cache_file = modelhl_cache_path("_cache_modelhl_prefill_frac_cdf")

# X-axis range — edit to zoom in/out
PREFILL_FRAC_XLIM = (0, 1)

if not cached or not cache_file.exists():
    chute_ids = df_hl["chute_id"].tolist()
    chute_list_sql = ", ".join(f"'{c}'" for c in chute_ids)

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
    for _, r in df_q.iterrows():
        for i, p in enumerate(PCTS):
            rows.append({"chute_id": r["chute_id"], "pct": p,
                         "prefill_frac": r["q"][i]})
    df_pf = pd.DataFrame(rows)
    df_pf.to_parquet(cache_file)
    print(f"  saved → {cache_file}")
else:
    df_pf = pd.read_parquet(cache_file)
    print(f"Loaded TTFT/duration percentiles from {cache_file}")

df_global = df_pf[df_pf["chute_id"] == "__GLOBAL__"].sort_values("pct")
df_seg = df_pf[df_pf["chute_id"] != "__GLOBAL__"].merge(
    df_hl[["chute_id", "label", "color"]], on="chute_id", how="inner")

fig, ax = plt.subplots(figsize=(3.2, 3))
if not df_global.empty:
    ax.plot(df_global["prefill_frac"], df_global["pct"], **GLOBAL_KW)
for _, r in df_hl.iterrows():
    sub = df_seg[df_seg["chute_id"] == r["chute_id"]].sort_values("pct")
    if sub.empty:
        continue
    ax.plot(sub["prefill_frac"], sub["pct"],
            color=r["color"], linewidth=1.5, label=r["label"])
ax.set_xlabel('TTFT / duration')
ax.set_ylabel('CDF')
ax.set_xlim(*PREFILL_FRAC_XLIM)
ax.set_ylim(0, 1)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='lower right', frameon=False, fontsize=8)
standard_ticks(ax)
style_grid(ax)
fig.tight_layout(pad=0.4)
if SAVE_FIGURES:
    fig.savefig(PDF_DIR / 'fig_prefill_fraction_cdf_modelhl.pdf',
                bbox_inches='tight', dpi=300)
    print(f"Saved {PDF_DIR / 'fig_prefill_fraction_cdf_modelhl.pdf'}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Latency heatmaps
#
# Duration and TTFT heatmaps from the global-characterization figure script. The style block below
# applies that notebook's matplotlib settings, scoped to these figures.
# ────────────────────────────────────────────────────────────────────────────

# ── Plot style from the global-characterization figure script (scoped to the heatmap figures) ──
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
    # line width 1.5
    'lines.linewidth': 1.5,
})
sns.set_palette('tab10')

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig17c_duration_p50_heatmap.pdf
#
# P50 end-to-end duration over (input tokens × output tokens) buckets.
# ────────────────────────────────────────────────────────────────────────────

# ── Fig 17c: P50 duration heatmap by input × output tokens ──
cached = CACHED
cache_file = CACHE_DIR / '_cache_perf_duration_p50_heatmap.parquet'

if not cached or not cache_file.exists():
    print('Computing P50 duration heatmap (input × output) …')
    df_17c = con.sql("""
        SELECT
            FLOOR(it / 2000) * 2  AS input_kb,
            FLOOR(ot / 200) * 200 AS output_bucket,
            COUNT(*)                          AS n,
            APPROX_QUANTILE(EPOCH(completed_at - started_at), 0.50) AS dur_p50
        FROM all_metrics_user
        WHERE started_at IS NOT NULL AND completed_at IS NOT NULL
          AND completed_at > started_at
          AND it > 0 AND ot > 0
          AND EPOCH(completed_at - started_at) < 300
        GROUP BY 1, 2
        HAVING COUNT(*) >= 50
    """).fetchdf()
    df_17c.to_parquet(cache_file)
    print(f'  ✓ {len(df_17c)} cells → {cache_file}')
else:
    df_17c = pd.read_parquet(cache_file)
    print('Loaded from cache')

df_plot = df_17c[(df_17c['input_kb'] <= 32) & (df_17c['output_bucket'] <= 8000)]
pivot = df_plot.pivot_table(index='output_bucket', columns='input_kb',
                            values='dur_p50', aggfunc='first')
pivot = pivot.sort_index(ascending=False)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                   cmap='YlOrRd', shading='auto', rasterized=True)
ax.set_xlabel('Input tokens (K)')
ax.set_ylabel('Output tokens')
fig.colorbar(im, ax=ax, label='P50 duration (s)', shrink=0.85)

fig.tight_layout(pad=0.4)
plt.savefig(PDF_DIR / 'fig17c_duration_p50_heatmap.pdf',
            bbox_inches='tight', dpi=300)
print(f"Saved {PDF_DIR / 'fig17c_duration_p50_heatmap.pdf'}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig_ttft_p50_heatmap_input_output.pdf
#
# P50 TTFT over (input tokens × output tokens) buckets.
# ────────────────────────────────────────────────────────────────────────────

# ── Figure: P50 TTFT heatmap by input × output tokens ──
from matplotlib.ticker import AutoMinorLocator

cached = CACHED
cache_file = CACHE_DIR / '_cache_perf_ttft_p50_heatmap_input_output.parquet'

INPUT_BIN = 2000
OUTPUT_BIN = 200
MAX_INPUT_K = 32
MAX_OUTPUT = 8000
MIN_CELL_COUNT = 50

if not cached or not cache_file.exists():
    print('Computing P50 TTFT heatmap (input × output) ...')
    df_ttft_io = con.sql(f"""
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
    """).fetchdf()
    df_ttft_io.to_parquet(cache_file)
    print(f'  saved {len(df_ttft_io):,} cells to {cache_file}')
else:
    df_ttft_io = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_ttft_io):,} cells from {cache_file}')

df_plot = df_ttft_io[
    (df_ttft_io['input_kb'] >= 0) & (df_ttft_io['input_kb'] <= MAX_INPUT_K) &
    (df_ttft_io['output_bucket'] >= 0) & (df_ttft_io['output_bucket'] <= MAX_OUTPUT)
].copy()

pivot = df_plot.pivot_table(
    index='output_bucket', columns='input_kb', values='ttft_p50', aggfunc='first'
).sort_index(ascending=False)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.pcolormesh(
    pivot.columns, pivot.index, pivot.values,
    cmap='YlOrRd', shading='auto', rasterized=True,
)
ax.set_xlabel('Input tokens (K)')
ax.set_ylabel('Output tokens')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)
fig.colorbar(im, ax=ax, label='P50 TTFT (s)', shrink=0.85)
fig.tight_layout(pad=0.4)
fig.savefig(PDF_DIR / 'fig_ttft_p50_heatmap_input_output.pdf',
            bbox_inches='tight', dpi=300)
print(f"Saved {PDF_DIR / 'fig_ttft_p50_heatmap_input_output.pdf'}")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Figure: fig16a_ttft_heatmap_input_cache.pdf
#
# Median TTFT over (input tokens x per-request token hit ratio) buckets - how
# prefix caching shifts TTFT at each input length.
# The VLDB paper includes this as a PNG screenshot,
# newer_floats/section2/chat_input_hit.png (caption "TTFT vs. hit ratio");
# this script saves the same plot as a vector PDF.
# ────────────────────────────────────────────────────────────────────────────

# ── Fig 16a: TTFT heatmap by input tokens × token hit ratio ──
cached = CACHED
cache_file = CACHE_DIR / '_cache_perf_ttft_heatmap_input_cache.parquet'

if not cached or not cache_file.exists():
    print('Computing TTFT heatmap (input × cache_fraction) …')
    df_h16a = con.sql("""
        WITH base AS (
            SELECT
                ttft,
                it,
                CAST(ct AS DOUBLE) / NULLIF(it, 0) AS cache_frac
            FROM all_metrics_user
            WHERE ttft IS NOT NULL AND ttft > 0 AND ttft < 300
              AND it > 0 AND ct IS NOT NULL
        )
        SELECT
            FLOOR(it / 2000) * 2                AS input_kb,
            FLOOR(LEAST(cache_frac, 1.0) * 10) / 10.0      AS cache_frac_bin,
            COUNT(*)                                       AS n,
            APPROX_QUANTILE(ttft, 0.50)                    AS ttft_p50,
            APPROX_QUANTILE(ttft, 0.95)                    AS ttft_p95
        FROM base
        WHERE cache_frac >= 0
        GROUP BY 1, 2
        HAVING COUNT(*) >= 200
    """).fetchdf()
    df_h16a.to_parquet(cache_file)
    print(f'  ✓ {len(df_h16a)} cells → {cache_file}')
else:
    df_h16a = pd.read_parquet(cache_file)
    print(f'Loaded {len(df_h16a)} cells from cache')

df_plot = df_h16a[(df_h16a['input_kb'] >= 0) & (df_h16a['input_kb'] <= 32) &
                  (df_h16a['cache_frac_bin'] >= 0) & (df_h16a['cache_frac_bin'] <= 1)].copy()

pivot = df_plot.pivot_table(index='cache_frac_bin', columns='input_kb',
                            values='ttft_p50', aggfunc='first')
pivot = pivot.sort_index(ascending=True)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                   cmap='YlOrRd', shading='auto', rasterized=True)
ax.set_xlabel('Input tokens (K)')
ax.set_ylabel('Token hit ratio')
fig.colorbar(im, ax=ax, label='Median TTFT (s)', shrink=0.85)

fig.tight_layout(pad=0.4)
plt.savefig(PDF_DIR / 'fig16a_ttft_heatmap_input_cache.pdf',
            bbox_inches='tight', dpi=300)
print(f"Saved {PDF_DIR / 'fig16a_ttft_heatmap_input_cache.pdf'}")
plt.close(fig)

print("Done — all Section 3 figures written to", PDF_DIR)
