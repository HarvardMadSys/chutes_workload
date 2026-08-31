#!/usr/bin/env python3
"""Section 6 — Load Balancing Case Study: figure reproduction.

Reproduces every figure of the paper's load-balancing case study (Section 6).
The code is the paper's own figure script; only the paths, the cache toggle,
part gating, and the axis labels were adapted. The labels
were rewritten for the paper's four-panel load-balancing float: the notebook's
shorthand ("requests / h", "inst. / h (median)", "instance_idx") became full
one-line labels of comparable width, so the panels align when each is set at
0.24\textwidth. Nothing that is plotted changed.

Part A — trace analysis (needs the anonymized one-year DuckDB trace, see --db).
Figures written to figures/paper/<section>/:

  fig_lb_request_rate.pdf       — hourly request rate, DeepSeek-V3-0324-TEE (paper LB_reqrate.png)
  fig_lb_active_instances.pdf   — median active instances per hour (paper LB_active_instance.png)
  fig_lb_maxmean_tokens.pdf     — per-minute max/mean per-instance token load, 1-h smoothed (paper LB_maxmean.png)
  fig_lb_concurrency.pdf        — per-instance request heatmap, densest 24-h window (paper concurrency.png)
  f7b_ttft_p90_<model>.pdf      — instance load x input tokens hexbin, colored by P90 TTFT
  f7d_decode_p90_<model>.pdf    — instance load x output tokens hexbin, colored by P90 decode time
  f1f_user_zoom_<tier>_<variant>_<model>.pdf — per-instance request share vs. token
                                  hit ratio for a picked user's busiest hour (the
                                  paper uses high_top / high_min for DeepSeek-V3.2-TEE)

The four fig_lb_* names are canonical repro names for plots the paper embeds as
PNG screenshots; the loops also emit the same plots for the other chosen model
under their original f3a_/f4_/f5a_/f1b_ names.

Part B — simulator sweep figures (needs only the Chutes Load Simulator sweep
output directory, NOT the trace database). Figures written to figures/paper/<section>/:

  fig_paper_legend.pdf                              — shared standalone policy legend
  fig_paper_hitrate_vs_cache_FINAL_{v32,minimax}_N20_.pdf  — token hit ratio vs cache size (N=20)
  fig_paper_repl_vs_cache_FINAL_{v32,minimax}_N20.pdf      — KV replication ratio vs cache size (N=20)
  fig_paper_repl_vs_N_FINAL_{v32,minimax}_cache25000.pdf   — replication ratio vs instance count (cache=25k)
  fig_paper_imbalance_vs_N_FINAL_{v32,minimax}_cache5000000.pdf — load imbalance (%) vs instance count (cache=5M)
  fig_lb_minimax_imbalance.pdf                      — copy of the minimax imbalance panel (paper minimaximbal.png)

Requirements: duckdb (Part A only), pandas, numpy, matplotlib, seaborn, pyarrow.
Heavy queries/derivations cache their results as parquet files under output/paper/cache/.

Usage:
    python 05_section6_loadbalancing_casestudy.py [--db PATH] [--recompute] [--part {a,b,all}]

By default, cached parquet files are reused when present (only plotting is
redone). Pass --recompute to rerun the heavy DuckDB queries / sweep-directory
scans from scratch. Pass --part b to skip the trace analysis entirely (no DB
needed).
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import re
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import (AutoMinorLocator, FuncFormatter, LogLocator,
                               MultipleLocator, NullFormatter, NullLocator,
                               ScalarFormatter)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import (  # noqa: E402
    CHUTE_MODELS_CSV, DB_DEFAULT, ROUTING_SWEEP_CSV, ROUTING_SWEEP_DIR, section_paths,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the anonymized trace "
                         "(scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true", help="rerun heavy queries/derivations even if cache files exist")
parser.add_argument("--sweep-dir", default=None,
                    help="raw routing-sweep directory to scan instead of the shipped "
                         "data/routing/figure20_sweep.csv (Part B)")
parser.add_argument("--part", choices=["a", "b", "all"], default="all", help="run only trace analysis (a), only simulator figures (b), or both")
args = parser.parse_args()
DB_PATH = args.db
CACHED = not args.recompute   # True: reuse cache files and just replot

# ────────────────────────────────────────────────────────────────────────────
# Output / cache directories (flat pdf/ + cache/ at the repo root)
# ────────────────────────────────────────────────────────────────────────────
PDF_DIR, CACHE_DIR = section_paths("05_load_balancing")
PDF_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Part A — Trace analysis
# (source: the paper's load-balancing figure script)
#
# Data prep first (curated models, CIM/MIM aggregates, shared helpers), then
# one section per figure. WARNING: the trace is a ~91 GB parquet; on first
# execution the DuckDB scans take minutes per model. Results are parquet-cached
# under cache/ and reused on later runs (unless --recompute).
# ════════════════════════════════════════════════════════════════════════════
def run_part_a():
    # ─────────────────────────────────────────────────────────────────────
    # Setup: plotting rc, DuckDB connection, window, chute lookup
    # ─────────────────────────────────────────────────────────────────────
    import duckdb

    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.6,
    })
    sns.set_palette("tab10")

    def style_grid(ax):
        ax.set_axisbelow(True)
        if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linestyle="--", linewidth=0.4)

    def short_model_name(name, chute_id=None, max_len=32):
        if pd.isna(name) or str(name).strip() == "":
            return str(chute_id)[:8] if chute_id is not None else "unknown"
        text = str(name)
        if "/" in text:
            text = text.split("/", 1)[-1]
        return text if len(text) <= max_len else text[:max_len - 1].rstrip("-/_.") + "…"

    def safe_slug(name):
        s = str(name).replace("/", "__")
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        return s.strip("._-") or "unknown"

    TABLE_NAME = "all_metrics_user"
    con = duckdb.connect(DB_PATH, read_only=True)
    con.sql("PRAGMA threads=48;")
    con.sql("PRAGMA memory_limit=\"1000GB\";")

    # Window: last 2 months of the trace. Timestamps are elapsed time from
    # the trace start (origin 1970-01-01), so this is day 364 back two months.
    LAST_MONTH_END   = "1970-12-31"
    LAST_MONTH_START = (pd.Timestamp(LAST_MONTH_END) - pd.DateOffset(months=2)
                        ).strftime("%Y-%m-%d")
    print(f"window: {LAST_MONTH_START} -> {LAST_MONTH_END}")

    # Chute lookup
    chute_df = pd.read_csv(CHUTE_MODELS_CSV)
    CHUTE_TO_MODEL = dict(zip(chute_df["chute_id"], chute_df["name"]))
    MODEL_TO_CHUTE = dict(zip(chute_df["name"], chute_df["chute_id"]))

    # The plots the paper embeds as PNG screenshots get canonical repro filenames.
    FIG_NAME_MAP = {
        "f3a_requests_deepseek-ai__DeepSeek-V3-0324-TEE.pdf":        "fig_lb_request_rate.pdf",
        "f4_active_instances_deepseek-ai__DeepSeek-V3-0324-TEE.pdf": "fig_lb_active_instances.pdf",
        "f5a_maxmean_tokens_deepseek-ai__DeepSeek-V3-0324-TEE.pdf":  "fig_lb_maxmean_tokens.pdf",
        "f1b_concurrency_24h_deepseek-ai__DeepSeek-V3-0324-TEE.pdf": "fig_lb_concurrency.pdf",
    }

    def save_fig(fig, name, subdir=None):
        # `subdir` is ignored: this repro script writes every PDF flat into PDF_DIR.
        name = FIG_NAME_MAP.get(name, name)
        path = PDF_DIR / name
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"saved {path}")
        return path

    # ─────────────────────────────────────────────────────────────────────
    # Data: curated model list (source cell 2)
    # ─────────────────────────────────────────────────────────────────────
    CHOSEN_MODEL_NAMES = [
        # "Qwen/Qwen3-32B-TEE",
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "deepseek-ai/DeepSeek-V3-0324-TEE",
        # "Qwen/Qwen3-32B",
        # "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
        # "moonshotai/Kimi-K2.5-TEE",
        # "minimax/Minimax-2.5-TEE"
        # "MiniMaxAI/MiniMax-M2.5-TEE"

    ]

    missing = [n for n in CHOSEN_MODEL_NAMES if MODEL_TO_CHUTE.get(n) is None]
    if missing:
        raise KeyError(f"missing in chute_models.csv: {missing}")

    CHOSEN_CHUTES = [MODEL_TO_CHUTE[n] for n in CHOSEN_MODEL_NAMES]
    CHOSEN_NAMES  = {c: CHUTE_TO_MODEL[c] for c in CHOSEN_CHUTES}
    CHOSEN_SHORT  = {c: short_model_name(CHOSEN_NAMES[c], c) for c in CHOSEN_CHUTES}
    CHOSEN_SLUG   = {c: safe_slug(CHOSEN_NAMES[c]) for c in CHOSEN_CHUTES}

    for c in CHOSEN_CHUTES:
        print(f"  {CHOSEN_SHORT[c]:<32}  chute_id={c}")

    # ─────────────────────────────────────────────────────────────────────
    # Data: CIM — chute x instance x minute aggregate (source cell 4)
    # Per-(instance, minute) request/token/latency aggregates over the
    # analysis window for each chosen chute. Parquet-cached per chute
    # under cache/cim/.
    # ─────────────────────────────────────────────────────────────────────
    CIM_DIR = CACHE_DIR / "cim"
    CIM_DIR.mkdir(parents=True, exist_ok=True)

    CIM_SQL = f"""
        SELECT
            chute_id, instance_id,
            DATE_TRUNC('minute', started_at) AS min_ts,
            COUNT(*)                                   AS req_count,
            COALESCE(SUM(it), 0)                       AS input_tokens,
            COALESCE(SUM(ot), 0)                       AS output_tokens,
            COALESCE(SUM(it + ot), 0)                  AS total_tokens,
            COALESCE(SUM(ct), 0)                       AS cached_tokens,
            approx_quantile(ttft, 0.5)                 AS ttft_p50,
            approx_quantile(ttft, 0.9)                 AS ttft_p90,
            approx_quantile(EXTRACT(EPOCH FROM (completed_at - started_at)), 0.5) AS dur_p50,
            approx_quantile(EXTRACT(EPOCH FROM (completed_at - started_at)), 0.9) AS dur_p90,
            COUNT(DISTINCT user_id)                    AS n_users
        FROM {TABLE_NAME}
        WHERE chute_id    = '{{chute_id}}'
          AND instance_id IS NOT NULL
          AND started_at >= TIMESTAMP '{LAST_MONTH_START}'
          AND started_at <  TIMESTAMP '{LAST_MONTH_END}'
        GROUP BY 1, 2, 3
    """

    def load_cim(chute_id):
        path = CIM_DIR / f"chute_{chute_id}.parquet"
        if CACHED and path.exists():
            return pd.read_parquet(path)
        t0 = time.time()
        df = con.sql(CIM_SQL.format(chute_id=chute_id)).fetchdf()
        df["min_ts"] = pd.to_datetime(df["min_ts"])
        df.to_parquet(path)
        print(f"  cim[{CHOSEN_SHORT[chute_id]}] -> {len(df):,} rows  ({time.time() - t0:.1f}s)")
        return df

    CIM = {}
    for c in CHOSEN_CHUTES:
        df = load_cim(c)
        df["min_ts"] = pd.to_datetime(df["min_ts"])
        CIM[c] = df
        print(f"  {CHOSEN_SHORT[c]:<32}  {len(df):>10,} (inst, min) rows  "
              f"  {df['instance_id'].nunique():>4} unique instances")

    # ─────────────────────────────────────────────────────────────────────
    # Data: MIM — per-minute imbalance summary (source cell 5)
    # Per-minute cross-instance summary: CV, max/mean, max/min, Jain's
    # fairness, and request-weighted latency percentiles. Cached under
    # cache/mim/.
    # ─────────────────────────────────────────────────────────────────────
    MIM_DIR = CACHE_DIR / "mim"
    MIM_DIR.mkdir(parents=True, exist_ok=True)

    def _cv(x):
        x = np.asarray(x, dtype=np.float64)
        m = x.mean()
        return float(x.std(ddof=0) / m) if m > 0 else np.nan

    def _max_mean(x):
        x = np.asarray(x, dtype=np.float64)
        m = x.mean()
        return float(x.max() / m) if m > 0 else np.nan

    def _max_min(x):
        # Ratio of busiest to least-busy active instance in the minute.
        # Only considers positive (active) values so an idle instance with 0
        # work doesn't make the ratio explode to infinity.
        x = np.asarray(x, dtype=np.float64)
        pos = x[x > 0]
        if pos.size == 0:
            return np.nan
        mn = pos.min()
        return float(pos.max() / mn) if mn > 0 else np.nan

    def _jains(x):
        """Jain's fairness index: (sum_i x_i)^2 / (n * sum_i x_i^2). Range
        [1/n, 1] where 1 is perfectly fair."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return np.nan
        s = x.sum()
        sq = (x * x).sum()
        return float((s * s) / (x.size * sq)) if sq > 0 else np.nan

    def _derive_minute(g):
        reqs   = g["req_count"].to_numpy()
        in_tok = g["input_tokens"].to_numpy()
        tt_tok = g["total_tokens"].to_numpy()
        cd_tok = g["cached_tokens"].to_numpy()
        w      = reqs.astype(np.float64)
        wsum   = w.sum()

        def wavg(col):
            v = g[col].to_numpy(dtype=np.float64)
            mask = ~np.isnan(v)
            if wsum <= 0 or not mask.any():
                return np.nan
            return float(np.average(v[mask], weights=w[mask]))

        return pd.Series({
            "n_active_instances":     int(reqs.size),
            "total_requests":         int(reqs.sum()),
            "total_input_tokens":     int(in_tok.sum()),
            "total_tokens":           int(tt_tok.sum()),
            "total_cached_tokens":    int(cd_tok.sum()),
            # Imbalance metrics
            "cv_req_count":           _cv(reqs),
            "cv_total_tokens":        _cv(tt_tok),
            "max_mean_req_count":     _max_mean(reqs),
            "max_mean_total_tokens":  _max_mean(tt_tok),
            "max_min_req_count":      _max_min(reqs),
            "max_min_total_tokens":   _max_min(tt_tok),
            "jains_req_count":        _jains(reqs),
            "jains_total_tokens":     _jains(tt_tok),
            # Latency (request-weighted)
            "ttft_p50": wavg("ttft_p50"),
            "ttft_p90": wavg("ttft_p90"),
            "dur_p50":  wavg("dur_p50"),
            "dur_p90":  wavg("dur_p90"),
        })

    def build_mim(chute_id):
        path = MIM_DIR / f"chute_{chute_id}.parquet"
        if CACHED and path.exists():
            df = pd.read_parquet(path)
            if "max_min_req_count" in df.columns:
                return df
            print(f"  mim[{CHOSEN_SHORT[chute_id]}] cache missing max/min — rebuilding")
        cim = CIM[chute_id]
        t0 = time.time()
        out = (cim.groupby("min_ts", sort=True).apply(_derive_minute).reset_index())
        out["chute_id"] = chute_id
        out.to_parquet(path)
        print(f"  mim[{CHOSEN_SHORT[chute_id]}] -> {len(out):,} minutes  ({time.time() - t0:.1f}s)")
        return out

    MIM = {c: build_mim(c) for c in CHOSEN_CHUTES}
    for c in CHOSEN_CHUTES:
        df = MIM[c]
        print(f"  {CHOSEN_SHORT[c]:<32}  {len(df):>6,} minutes  "
              f"median_n_inst={int(df['n_active_instances'].median())}")

    # ─────────────────────────────────────────────────────────────────────
    # Shared plot helpers (source cell 9, helper block)
    # _time_axis, _abbr_count, and _heatmap_pivot from the F1 heatmap cell —
    # needed by the concurrency, request-rate, active-instance, and max/mean
    # figures. The full-window F1 heatmap itself is not a paper figure and
    # is omitted.
    # ─────────────────────────────────────────────────────────────────────
    F1_BIN      = "5min"
    F1_MAX_INST = 400
    F1_FIGSIZE  = (3.5, 3.5)

    def _time_axis(ax, span_days):
        if span_days <= 14:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        elif span_days <= 45:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n'%y"))
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))

    def _abbr_count(x, _pos):
        if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
        if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
        if abs(x) >= 1e3: return f"{x/1e3:.0f}k"
        return f"{int(x)}"

    def _heatmap_pivot(cim, value_col, bin_freq, max_inst):
        df = cim.copy()
        df["bin_ts"] = df["min_ts"].dt.floor(bin_freq)
        first_seen = df.groupby("instance_id")["bin_ts"].min().sort_values()
        if len(first_seen) > max_inst:
            top = (df.groupby("instance_id")[value_col].sum()
                      .nlargest(max_inst).index)
            first_seen = first_seen.loc[top].sort_values()
        order = list(first_seen.index)
        pivot = (df[df["instance_id"].isin(order)]
                    .groupby(["instance_id", "bin_ts"])[value_col].sum()
                    .unstack(fill_value=0.0).reindex(order))
        full_cols = pd.date_range(pivot.columns.min(), pivot.columns.max(), freq=bin_freq)
        pivot = pivot.reindex(columns=full_cols, fill_value=0.0)
        return pivot, order

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_lb_request_rate.pdf (source cell 28, F3a)
    # Hourly request totals. The DeepSeek-V3-0324-TEE output is the panel
    # the paper embeds as LB_reqrate.png.
    # ─────────────────────────────────────────────────────────────────────
    F3_FIGSIZE = (3.5, 3)

    def fig3a_requests(chute_id):
        mim = MIM[chute_id]
        if len(mim) == 0:
            return
        name = CHOSEN_SHORT[chute_id]
        slug = CHOSEN_SLUG[chute_id]
        rs = (mim.set_index("min_ts")["total_requests"].resample("1h").sum())
        span_days = (mim["min_ts"].max() - mim["min_ts"].min()).days + 1

        fig, ax = plt.subplots(figsize=F3_FIGSIZE)
        ax.plot(rs.index, rs.values, color="C0", linewidth=1.3)
        ax.set_ylabel("Requests / hour")
        ax.set_title(name)
        ax.yaxis.set_major_formatter(FuncFormatter(_abbr_count))
        style_grid(ax)
        _time_axis(ax, span_days)
        fig.tight_layout()
        save_fig(fig, f"f3a_requests_{slug}.pdf", subdir=slug)
        plt.close(fig)

    for c in CHOSEN_CHUTES:
        fig3a_requests(c)

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_lb_active_instances.pdf (source cell 32, F4)
    # Median number of active instances per hour (paper LB_active_instance.png).
    # ─────────────────────────────────────────────────────────────────────
    def fig4_active_instances(chute_id):
        mim = MIM[chute_id]
        if len(mim) == 0:
            return
        name = CHOSEN_SHORT[chute_id]
        slug = CHOSEN_SLUG[chute_id]
        rs = (mim.set_index("min_ts")["n_active_instances"]
                  .resample("1h").median())
        span_days = (mim["min_ts"].max() - mim["min_ts"].min()).days + 1

        fig, ax = plt.subplots(figsize=F3_FIGSIZE)
        ax.plot(rs.index, rs.values, color="C2", linewidth=1.4)
        # The series is the median instance count *within* each hour, not a
        # rate; the subfigure caption carries the aggregation.
        ax.set_ylabel("Active instances")
        ax.set_title(name)
        style_grid(ax)
        _time_axis(ax, span_days)
        fig.tight_layout()
        save_fig(fig, f"f4_active_instances_{slug}.pdf", subdir=slug)
        plt.close(fig)

    for c in CHOSEN_CHUTES:
        fig4_active_instances(c)

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_lb_maxmean_tokens.pdf (source cell 34, F5a)
    # Max/mean per-instance token load per minute, 1-h smoothed
    # (paper LB_maxmean.png).
    # ─────────────────────────────────────────────────────────────────────
    F5_SMOOTH = "1h"
    F5_FIGSIZE = (3.5, 3)
    F5_MIN_INST = 2
    F5_MIN_REQ  = 10

    def _f5_filter(mim):
        return mim[(mim["n_active_instances"] >= F5_MIN_INST)
                   & (mim["total_requests"]    >= F5_MIN_REQ)].copy()

    def _f5_overlay(chute_id, col, color, ylabel, suffix, ylim=None):
        mim = MIM[chute_id]
        bal = _f5_filter(mim)
        if len(bal) == 0:
            return
        name = CHOSEN_SHORT[chute_id]
        slug = CHOSEN_SLUG[chute_id]
        rs_smth = bal.set_index("min_ts")[col].resample(F5_SMOOTH).mean()
        span_days = (bal["min_ts"].max() - bal["min_ts"].min()).days + 1

        fig, ax = plt.subplots(figsize=F5_FIGSIZE)
        ax.plot(rs_smth.index, rs_smth.values, color=color, linewidth=1.6)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        if ylim is not None:
            ax.set_ylim(*ylim)
        style_grid(ax)
        _time_axis(ax, span_days)
        fig.tight_layout()
        save_fig(fig, f"f5{suffix}_{slug}.pdf", subdir=slug)
        plt.close(fig)

    for c in CHOSEN_CHUTES:
        _f5_overlay(c, "max_mean_total_tokens", "C0",
                    "Max/mean token load", "a_maxmean_tokens")

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_lb_concurrency.pdf (source cell 13, F1b)
    # Per-instance request heatmap (1-min bins) over the densest 24-h
    # window (paper concurrency.png).
    # ─────────────────────────────────────────────────────────────────────
    F1B_BIN          = "1min"
    F1B_BIN_LABEL    = "min"        # unit shown on the colorbar
    F1B_MAX_INST     = 60
    F1B_FIGSIZE      = (4.2, 3.5)
    F1B_WINDOW_HOURS = 24            # zoom width
    F1B_HOUR_TICK    = 2             # major x tick every N hours

    def _densest_window(cim, hours):
        """End timestamp of the rolling `hours`-wide window with the highest
        MEAN count of distinct active instances per minute. Mean (not median)
        so windows with empty stretches are penalised proportionally."""
        g = cim[["min_ts", "instance_id"]].copy()
        per_min = (g.groupby("min_ts")["instance_id"].nunique()
                      .rename("n_inst").sort_index())
        # Reindex to a dense per-minute series so empty minutes count as 0.
        full_idx = pd.date_range(per_min.index.min(), per_min.index.max(),
                                 freq="1min")
        per_min = per_min.reindex(full_idx, fill_value=0)
        width = max(1, int(hours * 60))
        rolled = per_min.rolling(window=f"{hours}h", min_periods=width).mean()
        return rolled.idxmax()

    def fig1b_concurrency_oneday(chute_id, hours=F1B_WINDOW_HOURS):
        cim = CIM[chute_id]
        if len(cim) == 0:
            return
        name = CHOSEN_SHORT[chute_id]
        slug = CHOSEN_SLUG[chute_id]

        win_end_anchor = _densest_window(cim, hours)
        win_start = (pd.Timestamp(win_end_anchor) - pd.Timedelta(hours=hours)).floor("h")
        win_end   = win_start + pd.Timedelta(hours=hours)
        sl = cim[(cim["min_ts"] >= win_start) & (cim["min_ts"] < win_end)].copy()
        if len(sl) == 0:
            return

        pivot, order = _heatmap_pivot(sl, "req_count", F1B_BIN, F1B_MAX_INST)
        full_cols = pd.date_range(win_start, win_end - pd.Timedelta(F1B_BIN),
                                  freq=F1B_BIN)
        pivot = pivot.reindex(columns=full_cols, fill_value=0.0)

        arr = pivot.to_numpy(dtype=float)
        arr_plot = np.where(arr > 0, arr, np.nan)
        if np.isfinite(arr_plot).any():
            vmax = np.nanpercentile(arr_plot, 99)
            vmin = max(1.0, np.nanpercentile(arr_plot, 5))
        else:
            vmin, vmax = 1.0, 1.0

        n_inst = len(order)
        fig, ax = plt.subplots(figsize=F1B_FIGSIZE, constrained_layout=True)
        im = ax.imshow(
            arr_plot, aspect="auto", origin="lower",
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=max(vmin * 1.01, vmax)),
            extent=[0, hours, 0, n_inst],
            interpolation="nearest",
        )
        # Rows are ordered by the bin each instance first appears in.
        ax.set_ylabel("Instance index")
        ax.set_xlabel(f"Hours since {win_start:%Y-%m-%d %H:%M}")
        ax.set_title(name)

        # Compact colorbar with plain integer labels (no 4×10¹).
        cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.4)
        cb.set_label(f"Requests / {F1B_BIN_LABEL}")
        # cb.minorticks_off()
        sf = ScalarFormatter()
        sf.set_scientific(False)
        ticks = [10, 20,25, 40, 60, 80, 100, 150, 200]
        ticks = [t for t in ticks if vmin <= t <= vmax]

        cb.set_ticks(ticks)
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.set_label(f"Requests / {F1B_BIN_LABEL}")

        # min rounded to nearest 10, middle, max
        vmin_tick = int(round((vmin + 9)/ 10) * 10)
        vmax_tick = int(round(vmax/ 10) * 10)
        vmid_tick = int(round(((vmin_tick + vmax_tick) / 2) / 10) * 10)

        ticks = sorted(set([vmin_tick, vmid_tick, vmax_tick]))
        ticks = [max(1, t) for t in ticks]

        cb.set_ticks(ticks)
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        cb.ax.yaxis.set_minor_locator(NullLocator())

        # cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

        # Discrete integer y ticks (no .5 offsets).
        ax.set_ylim(0, n_inst)
        if n_inst <= 12:
            y_step = 1
        elif n_inst <= 30:
            y_step = 5
        else:
            y_step = 10
        yticks = np.arange(0, n_inst + 1, y_step)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(t)) for t in yticks])
        ax.yaxis.set_minor_locator(NullLocator())

        # X ticks every F1B_HOUR_TICK hours starting at 0.
        xticks = np.arange(0, hours + 1, F1B_HOUR_TICK)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(t)) for t in xticks])
        ax.set_xlim(0, hours)
        ax.xaxis.set_minor_locator(AutoMinorLocator(F1B_HOUR_TICK))
        ax.grid(False, which="both")

        save_fig(fig, f"f1b_concurrency_{hours}h_{slug}.pdf", subdir=slug)
        plt.close(fig)

    for c in CHOSEN_CHUTES:
        fig1b_concurrency_oneday(c)

    # ─────────────────────────────────────────────────────────────────────
    # Data: F7 per-request load join (source cell 7)
    # One row per request joined with its (instance, arrival-second)
    # bucket's total tokens — the load axis of the F7 hexbins. Cached under
    # cache/f7_req_bucket/.
    # ─────────────────────────────────────────────────────────────────────
    LB3B_DIR = CACHE_DIR / "f7_req_bucket"
    LB3B_DIR.mkdir(parents=True, exist_ok=True)

    F7_SQL = f"""
    WITH filtered AS (
        SELECT instance_id,
               started_at,
               ttft,
               it,
               ot,
               EXTRACT(EPOCH FROM (completed_at - started_at)) AS dur_s,
               DATE_TRUNC('second', started_at)                AS sec_ts
        FROM {TABLE_NAME}
        WHERE chute_id    = '{{chute_id}}'
          AND instance_id IS NOT NULL
          AND started_at >= TIMESTAMP '{LAST_MONTH_START}'
          AND started_at <  TIMESTAMP '{LAST_MONTH_END}'
    ),
    bucket AS (
        SELECT instance_id, sec_ts,
               COALESCE(SUM(it + ot), 0) AS bucket_total_tokens
        FROM filtered
        GROUP BY 1, 2
    )
    SELECT
        s.it                              AS req_input_tokens,
        s.ot                              AS req_output_tokens,
        s.ttft                            AS req_ttft,
        GREATEST(s.dur_s - s.ttft, 0.0)   AS req_gen_time,
        b.bucket_total_tokens
    FROM filtered s
    JOIN bucket   b
      ON s.instance_id = b.instance_id
     AND s.sec_ts      = b.sec_ts
    """

    def load_req_bucket(chute_id):
        path = LB3B_DIR / f"chute_{chute_id}.parquet"
        if CACHED and path.exists():
            return pd.read_parquet(path)
        t0 = time.time()
        print(f"  *** F7 per-request scan {CHOSEN_SHORT[chute_id]} — may take minutes")
        df = con.sql(F7_SQL.format(chute_id=chute_id)).fetchdf()
        df.to_parquet(path)
        print(f"  f7[{CHOSEN_SHORT[chute_id]}] -> {len(df):,} rows  ({time.time() - t0:.1f}s)")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # F7 shared helpers (source cell 48)
    # ─────────────────────────────────────────────────────────────────────
    F7_HEX_GRID   = 36
    F7_HEX_MINCNT = 30
    F7_FIGSIZE    = (3.5, 3)

    def _p90(v):
        if len(v) == 0:
            return float("nan")
        return float(np.percentile(v, 90))

    def _f7_panel(chute_id, y_col, y_label, c_col, reduce_fn, c_label, suffix):
        df = load_req_bucket(chute_id)
        if len(df) == 0:
            print(f"  skip {CHOSEN_SHORT[chute_id]}: empty F7 dataset")
            return
        name = CHOSEN_SHORT[chute_id]
        slug = CHOSEN_SLUG[chute_id]

        x = df["bucket_total_tokens"].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        c = df[c_col].to_numpy(dtype=float)
        mask = (x > 0) & (y > 0) & np.isfinite(c)
        if mask.sum() == 0:
            return

        fig, ax = plt.subplots(figsize=F7_FIGSIZE)
        cfin = c[mask][np.isfinite(c[mask])]
        vmin = float(np.nanpercentile(cfin, 5))
        vmax = float(np.nanpercentile(cfin, 95))
        hb = ax.hexbin(
            x[mask], y[mask], C=c[mask],
            reduce_C_function=reduce_fn,
            xscale="log", yscale="log",
            gridsize=F7_HEX_GRID, mincnt=F7_HEX_MINCNT,
            cmap="magma_r",
            vmin=vmin, vmax=max(vmin + 1e-6, vmax),
        )
        cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label(c_label)
        ax.set_xlabel("Instance load (tokens/s)")
        ax.set_ylabel(y_label)
        ax.set_title(name)
        style_grid(ax)
        fig.tight_layout()
        # save fig just to this dir
        plt.savefig(PDF_DIR / f"f7{suffix}_{slug}.pdf")
        save_fig(fig, f"f7{suffix}_{slug}.pdf")
        print("Saved to Path:", f"f7{suffix}_{slug}.pdf")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # Figure: f7b_ttft_p90_deepseek-ai__DeepSeek-V3-0324-TEE.pdf
    # (source cell 52, F7b)
    # Input tokens x instance load, hexbin colored by P90 TTFT.
    # ─────────────────────────────────────────────────────────────────────
    for c in CHOSEN_CHUTES:
        _f7_panel(c,
                  y_col="req_input_tokens",  y_label="Input tokens",
                  c_col="req_ttft",          reduce_fn=_p90,
                  c_label="P90 TTFT (s)",    suffix="b_ttft_p90")

    # ─────────────────────────────────────────────────────────────────────
    # Figure: f7d_decode_p90_deepseek-ai__DeepSeek-V3.2-TEE.pdf
    # (source cell 56, F7d)
    # Output tokens x instance load, hexbin colored by P90 decode time.
    # ─────────────────────────────────────────────────────────────────────
    for c in CHOSEN_CHUTES:
        _f7_panel(c,
                  y_col="req_output_tokens", y_label="Output tokens",
                  c_col="req_gen_time",      reduce_fn=_p90,
                  c_label="P90 decode time (s)",
                  suffix="d_decode_p90")

    # ─────────────────────────────────────────────────────────────────────
    # F1d helpers — user picking + per-user access queries
    # (source cell 18, helper block)
    # Constants and helpers the F1f figure needs: each user's busiest hour
    # with its hit ratio (_f1d_user_busy_hours), tier picking
    # (_f1d_pick_users), and per-user / per-window instance queries. The
    # F1d multi-panel figure itself is not a paper figure and its plotting
    # function is omitted.
    # ─────────────────────────────────────────────────────────────────────
    F1D_GRAN_S        = 1
    F1D_WIN_HOURS     = 1
    F1D_MIN_REQ_HOUR  = 200
    F1D_HI_CACHE      = 0.4
    F1D_LO_CACHE      = 0.2
    # ── Figure dimensions (inches) ────────────────────────────────────
    # F1D_FIGSIZE is the default for every tier. The figure has 4
    # panels (heatmap + 3 side bars) glued with constrained_layout.
    #   width  → makes the heatmap wider; side bars stay 1/8 of width
    #            each (controlled by width_ratios=[5,1,1,1] below).
    #   height → vertical room per instance row. With ~80 rows, 5.0in
    #            is ~16 rows per inch — bump if rows look squashed.
    F1D_FIGSIZE       = (7, 3.0)

    # Per-pick overrides, keyed by "<tier>_<variant>" where variant is
    # "top" (heaviest user in tier) or "min" (lightest qualifying user).
    # Set a tuple to resize that PDF only; leave as None to fall back to
    # F1D_FIGSIZE.
    F1D_FIGSIZE_BY_TIER = {
        "high_top": None,   # e.g. (10.5, 6.0)
        "high_min": None,
        "med_top":  None,
        "med_min":  None,
        "low_top":  None,
        "low_min":  None,
    }

    # Relative width of each panel inside a figure: heatmap, %reqs,
    # %util, cache%. Edit these to change the heatmap-vs-bars ratio.
    F1D_WIDTH_RATIOS  = [2, 1, 1, 1]
    F1D_X_TICK_MIN    = 10
    F1D_MAX_INST      = 80
    F1D_BG_CMAP       = "Greys"
    F1D_HIT_CMAP      = LinearSegmentedColormap.from_list("f1d", ["black", "red"])
    F1D_REQ_COLOR     = "C0"
    F1D_UTIL_COLOR    = "C2"
    F1D_CACHE_DIR     = CACHE_DIR / "f1d"
    F1D_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _f1d_user_busy_hours(chute_id):
        """Each user's busiest hour-aligned window with its hit rate / count.
        One row per user, request floor enforced. Parquet-cached."""
        path = F1D_CACHE_DIR / f"win_summary_{chute_id}.parquet"
        if CACHED and path.exists():
            df = pd.read_parquet(path)
        else:
            df = con.sql(f"""
                WITH per_user_hour AS (
                    SELECT user_id,
                           DATE_TRUNC('hour', started_at) AS hr_ts,
                           COUNT(*)                          AS n_req,
                           COALESCE(SUM(it), 0)::BIGINT      AS it_sum,
                           COALESCE(SUM(ct), 0)::BIGINT      AS ct_sum
                    FROM {TABLE_NAME}
                    WHERE chute_id = '{chute_id}'
                      AND user_id  IS NOT NULL
                      AND started_at >= TIMESTAMP '{LAST_MONTH_START}'
                      AND started_at <  TIMESTAMP '{LAST_MONTH_END}'
                    GROUP BY 1, 2
                ),
                ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY user_id
                                              ORDER BY n_req DESC, hr_ts ASC) AS rnk
                    FROM per_user_hour
                )
                SELECT user_id,
                       hr_ts AS busy_hr,
                       n_req,
                       CASE WHEN it_sum > 0
                            THEN ct_sum::DOUBLE / it_sum
                            ELSE 0.0 END AS hit_rate
                FROM ranked
                WHERE rnk = 1
                  AND n_req >= {F1D_MIN_REQ_HOUR}
            """).fetchdf()
            df.to_parquet(path)
        df["busy_hr"] = pd.to_datetime(df["busy_hr"])
        return df

    def _f1d_pick_users(df):
        """Two picks per tier: "top" (most reqs in their busy hour) and
        "min" (fewest reqs that still clears F1D_MIN_REQ_HOUR)."""
        pools = [
            ("high", df[df["hit_rate"] >= F1D_HI_CACHE]),
            ("med",  df[(df["hit_rate"] > F1D_LO_CACHE)
                        & (df["hit_rate"] < F1D_HI_CACHE)]),
            ("low",  df[df["hit_rate"] <= F1D_LO_CACHE]),
        ]
        out = []
        for tier, sub in pools:
            if len(sub) == 0:
                out.append((tier, "top", None))
                out.append((tier, "min", None))
                continue
            sub_sorted = sub.sort_values("n_req", ascending=False)
            out.append((tier, "top", sub_sorted.iloc[0]))
            if len(sub_sorted) > 1:
                out.append((tier, "min", sub_sorted.iloc[-1]))
            else:
                out.append((tier, "min", None))
        return out

    def _f1d_user_acc(chute_id, user_id, start, end):
        return con.sql(f"""
            SELECT started_at, instance_id,
                   COALESCE(it, 0)::DOUBLE AS it,
                   COALESCE(ct, 0)::DOUBLE AS ct
            FROM {TABLE_NAME}
            WHERE chute_id    = '{chute_id}'
              AND user_id     = '{user_id}'
              AND instance_id IS NOT NULL
              AND started_at >= TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}'
              AND started_at <  TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'
        """).fetchdf()

    def _f1d_bg_load(chute_id, start, end, instances):
        if not instances:
            return pd.DataFrame(columns=["instance_id", "bin_idx", "n_req"])
        inst_list = ", ".join("'" + str(i) + "'" for i in instances)
        return con.sql(f"""
            SELECT instance_id,
                   CAST(FLOOR(EXTRACT(EPOCH FROM
                       (started_at - TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}'))
                       / {F1D_GRAN_S}) AS BIGINT)            AS bin_idx,
                   COUNT(*)                                   AS n_req
            FROM {TABLE_NAME}
            WHERE chute_id    = '{chute_id}'
              AND instance_id IN ({inst_list})
              AND started_at >= TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}'
              AND started_at <  TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'
            GROUP BY 1, 2
        """).fetchdf()

    def _f1d_window_instances(chute_id, start, end):
        cim = CIM[chute_id]
        sl = cim[(cim["min_ts"] >= start) & (cim["min_ts"] < end)]
        if len(sl) == 0:
            return []
        first_seen = sl.groupby("instance_id")["min_ts"].min().sort_values()
        if len(first_seen) > F1D_MAX_INST:
            top = (sl.groupby("instance_id")["req_count"].sum()
                      .nlargest(F1D_MAX_INST).index)
            first_seen = first_seen.loc[top].sort_values()
        return list(first_seen.index)

    # ─────────────────────────────────────────────────────────────────────
    # F1e helpers needed by F1f (source cell 20, helper block)
    # F1E_REQ_COLOR and the whole-model lookahead diagnostic
    # _f1e_model_window_stats; the F1e figure itself is omitted.
    # ─────────────────────────────────────────────────────────────────────
    F1E_FIGSIZE         = (3.5, 3.0)

    F1E_FIGSIZE_BY_TIER = {
        "high_top": None,   # e.g. (6.5, 6.0)
        "high_min": None,
        "med_top":  None,
        "med_min":  None,
        "low_top":  None,
        "low_min":  None,
    }

    # Relative widths: share panel | cache % panel.
    F1E_WIDTH_RATIOS    = [1.5, 1]

    F1E_REQ_COLOR       = "C0"
    F1E_UTIL_COLOR      = "C2"

    # Whole-model lookahead windows for diagnostic prints.
    F1E_LOOKAHEAD = [
        ("15m", pd.Timedelta(minutes=15)),
        ("30m", pd.Timedelta(minutes=30)),
        ("1h",  pd.Timedelta(hours=1)),
    ]

    def _f1e_model_window_stats(chute_id, start):
        """Whole-model n_req + cacheability over 15m/30m/1h after `start`."""
        out = []
        start_ts = pd.Timestamp(start).strftime("%Y-%m-%d %H:%M:%S")
        for label, delta in F1E_LOOKAHEAD:
            win_end = pd.Timestamp(start) + delta
            end_ts  = win_end.strftime("%Y-%m-%d %H:%M:%S")
            row = con.sql(f"""
                SELECT
                    COUNT(*) AS n_req,
                    AVG(LEAST(GREATEST(CAST(ct AS DOUBLE) / it, 0.0), 1.0))
                        FILTER (WHERE ct IS NOT NULL
                                AND it IS NOT NULL AND it > 0) AS avg_ratio,
                    SUM(ct) FILTER (WHERE it IS NOT NULL AND it > 0) /
                        NULLIF(SUM(it) FILTER (WHERE ct IS NOT NULL
                                               AND it IS NOT NULL AND it > 0), 0)
                        AS sum_ratio
                FROM {TABLE_NAME}
                WHERE chute_id = '{chute_id}'
                  AND started_at >= TIMESTAMP '{start_ts}'
                  AND started_at <  TIMESTAMP '{end_ts}'
                  AND it is not null
                  AND (function_name = 'chat' OR function_name = 'chat_stream')
            """).fetchone()
            out.append((label, row[0], row[1], row[2]))
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Figure: f1f_user_zoom_high_top / high_min (source cell 22, F1f)
    # Single-panel twin-y bars per instance — % of the user's requests
    # (left) vs. per-instance token hit ratio (right) — for the busiest
    # hour of one picked user per (tier, variant). The paper uses the
    # high_top and high_min picks for DeepSeek-V3.2-TEE; the loop also
    # emits the other tiers/models.
    # ─────────────────────────────────────────────────────────────────────
    F1F_FIGSIZE         = (3.5, 3.0)

    F1F_FIGSIZE_BY_TIER = {
        "high_top": None,
        "high_min": None,
        "med_top":  None,
        "med_min":  None,
        "low_top":  None,
        "low_min":  None,
    }

    F1F_REQ_COLOR = F1E_REQ_COLOR  # blue
    F1F_HIT_COLOR = "C3"           # red

    def fig1f_one_pick(chute_id, tier, variant, row):
        if row is None:
            print(f"  {CHOSEN_SHORT[chute_id]} {tier}_{variant}: no qualifying user")
            return

        name = CHOSEN_SHORT[chute_id]
        slug = CHOSEN_SLUG[chute_id]

        user_id = row["user_id"]
        busy_hr = pd.Timestamp(row["busy_hr"])
        win_hit = float(row["hit_rate"])
        win_n   = int(row["n_req"])

        start = busy_hr
        end   = busy_hr + pd.Timedelta(hours=F1D_WIN_HOURS)

        print(f"  [{name}] {tier}_{variant} user={user_id} start={start}")
        for label, n_req, avg_ratio, sum_ratio in _f1e_model_window_stats(chute_id, start):
            avg_s = f"{avg_ratio:.4f}" if avg_ratio is not None else "n/a"
            sum_s = f"{sum_ratio:.4f}" if sum_ratio is not None else "n/a"
            print(f"    +{label}: model_n_req={n_req:,}  "
                  f"avg(ct/it)={avg_s}  sum(ct)/sum(it)={sum_s}")

        order = _f1d_window_instances(chute_id, start, end)
        if not order:
            print(f"  {name} {tier}_{variant}: no instances active in window")
            return
        inst_to_row = {iid: i for i, iid in enumerate(order)}
        n_inst = len(order)

        win_s   = (end - start).total_seconds()
        n_bins  = max(1, int(round(win_s / F1D_GRAN_S)))

        ua = _f1d_user_acc(chute_id, user_id, start, end)
        if len(ua) == 0:
            print(f"  {name} {tier}_{variant}: empty hour")
            return
        ua["started_at"] = pd.to_datetime(ua["started_at"])
        rows_u = ua["instance_id"].map(inst_to_row)
        bin_u  = ((ua["started_at"] - start).dt.total_seconds()
                  / F1D_GRAN_S).astype(np.int64)
        valid  = (rows_u.notna() & (bin_u >= 0) & (bin_u < n_bins)).to_numpy()
        rows_arr = rows_u[valid].to_numpy(dtype=np.int64)
        cols_arr = bin_u[valid].to_numpy(dtype=np.int64)
        it_arr   = ua.loc[valid, "it"].to_numpy(dtype=np.float64)
        ct_arr   = ua.loc[valid, "ct"].to_numpy(dtype=np.float64)

        n_inst_shape = (n_inst, n_bins)
        count_mat = np.zeros(n_inst_shape, dtype=np.float64)
        it_mat    = np.zeros(n_inst_shape, dtype=np.float64)
        ct_mat    = np.zeros(n_inst_shape, dtype=np.float64)
        np.add.at(count_mat, (rows_arr, cols_arr), 1.0)
        np.add.at(it_mat,    (rows_arr, cols_arr), it_arr)
        np.add.at(ct_mat,    (rows_arr, cols_arr), ct_arr)

        n_per_inst   = count_mat.sum(axis=1)
        it_per_inst  = it_mat.sum(axis=1)
        ct_per_inst  = ct_mat.sum(axis=1)

        pct_req = (100.0 * n_per_inst / n_per_inst.sum()
                   if n_per_inst.sum() > 0 else n_per_inst)
        with np.errstate(invalid="ignore", divide="ignore"):
            hit_pct_per_inst = np.where(it_per_inst > 0,
                                        100.0 * ct_per_inst / it_per_inst, 0.0)
        hit_pct_per_inst = np.clip(hit_pct_per_inst, 0.0, 100.0)

        centers = np.arange(n_inst)  # ticks land on the bar-group centre.

        figsize = F1F_FIGSIZE_BY_TIER.get(f"{tier}_{variant}") or F1F_FIGSIZE
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax_hit = ax.twinx()

        bar_w = 0.4
        off   = 0.2
        bars_req = ax.bar(centers - off, pct_req, width=bar_w,
                          color=F1F_REQ_COLOR, edgecolor="black", linewidth=0.25,
                          label="User request")
        bars_hit = ax_hit.bar(centers + off, hit_pct_per_inst, width=bar_w,
                              color=F1F_HIT_COLOR, edgecolor="black", linewidth=0.25,
                              label="Cache hit")

        ax.set_xlim(-0.5, n_inst - 0.5)
        ax.set_xlabel("Instance index")

        # Left y: % user util (blue, auto-scaled).
        share_ymax = max(8.0, float(pct_req.max() if len(pct_req) else 0.0) * 1.25)
        ax.set_ylim(0, share_ymax)
        ax.set_ylabel("User request (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))

        # Right y: cache hit %, fixed 0-100, red.
        ax_hit.set_ylim(0, 100)
        ax_hit.set_ylabel("Token hit (%)")
        ax_hit.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax_hit.spines["right"].set_visible(True)
        ax_hit.spines["right"].set_color("black")
        ax_hit.spines["right"].set_linestyle("-")

        # X-axis ticks: major at sensible step, minor every 1 as helper grid.
        if n_inst <= 12:
            x_major = 1
        elif n_inst <= 30:
            x_major = 5
        else:
            x_major = 10
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))

        ax.grid(True, axis="x", which="major", alpha=0.35,
                linestyle="--", linewidth=0.5)
        ax.grid(True, axis="x", which="minor", alpha=0.15,
                linestyle=":",  linewidth=0.4)
        ax.grid(True, axis="y", which="major", alpha=0.3,
                linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        ax.legend(
            handles=[bars_req, bars_hit],
            loc="upper left",
            frameon=False, fontsize=7,
            handlelength=1.2, handletextpad=0.4, borderaxespad=0.3,
        )

        save_fig(fig, f"f1f_user_zoom_{tier}_{variant}_{slug}.pdf", subdir=slug)
        plt.close(fig)

    def fig1f_user_zoom(chute_id):
        name = CHOSEN_SHORT[chute_id]
        df = _f1d_user_busy_hours(chute_id)
        if len(df) == 0:
            print(f"  skip {name}: no users with busy-hour >= {F1D_MIN_REQ_HOUR} reqs")
            return
        for tier, variant, row in _f1d_pick_users(df):
            fig1f_one_pick(chute_id, tier, variant, row)

    for c in CHOSEN_CHUTES:
        fig1f_user_zoom(c)

    con.close()


# ════════════════════════════════════════════════════════════════════════════
# Part B — Simulator sweep figures
# (source: pipeline/routing/)
#
# Part B does NOT read the trace. It plots the routing simulator's output:
# the sweep replays preprocessed request sessions across a fleet of simulated
# instances under four routing policies (round_robin, load_first, cache_first,
# sticky).
#
# It reads data/routing/figure20_sweep.csv, which ships with the artifact, and
# falls back to the raw sweep directory (one subdirectory per (model, N,
# cache_size, load_metric) cell, each with a summary.csv and
# policy=*/instance_metrics.parquet) when that is present.
#
# Rebuild either with:
#   make dataset    sessions.parquet from the trace
#   make routing    the 72-cell sweep, the CSV, and the figure
# ════════════════════════════════════════════════════════════════════════════
def run_part_b():
    # ─────────────────────────────────────────────────────────────────────
    # Part B setup: plotting rc + legend helper (source cell 0)
    # Re-applies the routing figures' rcParams (they differ slightly
    # from Part A's, matching how each source script rendered its own
    # figures).
    # ─────────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.5,
    })

    def style_grid(ax):
        """Dashed major + minor grid on both axes (matches the paper style)."""
        ax.set_axisbelow(True)
        if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linestyle="--", linewidth=0.4)

    def save_legend(handles, labels, path, ncol=None, figsize=None):
        """Render a standalone legend as its own PDF."""
        if ncol is None:
            ncol = len(labels)
        if figsize is None:
            figsize = (max(1.5, 1.1 * len(labels)) + 0.5, 0.5)
        fig_l = plt.figure(figsize=figsize)
        fig_l.legend(
            handles, labels,
            loc="center", ncol=ncol, frameon=False,
            columnspacing=1.2, handlelength=1.8,
        )
        fig_l.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig_l)

    # Flat output for this repro script: reuse PDF_DIR from Setup.
    FINAL_OUT_DIR = PDF_DIR

    # ─────────────────────────────────────────────────────────────────────
    # Load sweep summaries (source cell 2)
    # ─────────────────────────────────────────────────────────────────────
    # This artifact ships the rolled-up sweep as data/routing/figure20_sweep.csv
    # (1,152 rows = the full 288-cell grid x 4 policies), which already carries
    # both the summary.csv columns and the derived max/min columns. Prefer it;
    # fall back to scanning a raw sweep directory when one is given or the CSV
    # is absent. --sweep-dir overrides, --recompute forces the rescan.
    SWEEP_DIR = Path(args.sweep_dir) if args.sweep_dir else ROUTING_SWEEP_DIR
    SWEEP_CSV = ROUTING_SWEEP_CSV
    USE_SWEEP_CSV = (not args.sweep_dir) and (not args.recompute) and SWEEP_CSV.exists()
    SWEEP_CACHE = CACHE_DIR / "sweep_summary.parquet"
    SWEEP_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if USE_SWEEP_CSV:
        sweep_df = pd.read_csv(SWEEP_CSV)
        print(f"Loaded {len(sweep_df):,} rows from {SWEEP_CSV}")
    elif not CACHED or not SWEEP_CACHE.exists():
        rows = []
        for cell in sorted(SWEEP_DIR.iterdir()):
            if not cell.is_dir():
                continue
            name = cell.name
            try:
                model, rest = name.split("_N", 1)
                N_s, rest = rest.split("_cache", 1)
                cache_s, lm = rest.split("_lm_", 1)
            except ValueError:
                print(f"skip malformed: {name}")
                continue
            csv = cell / "summary.csv"
            if not csv.exists():
                print(f"missing summary.csv: {cell}")
                continue
            df = pd.read_csv(csv)
            df["model"] = model
            df["num_instances"] = int(N_s)
            df["cache_size"] = int(cache_s)
            df["load_metric"] = lm
            rows.append(df)
        sweep_df = pd.concat(rows, ignore_index=True)
        sweep_df.to_parquet(SWEEP_CACHE, engine="pyarrow")
        print(f"Saved {len(sweep_df):,} rows -> {SWEEP_CACHE}")
    else:
        sweep_df = pd.read_parquet(SWEEP_CACHE)
        print(f"Loaded {len(sweep_df):,} rows from {SWEEP_CACHE}")

    print(
        sweep_df[
            ["model", "num_instances", "cache_size", "load_metric", "policy_label"]
        ].nunique()
    )

    # ─────────────────────────────────────────────────────────────────────
    # Derive max/min token load from instance_metrics.parquet (source cell 3)
    # The summary.csv only stores max/mean, CV, and Jain — but for the
    # "cache_first sits between LB and sticky" story we also want max/min
    # (peak vs. coldest instance). Computed on active_tokens_per_sec_mean
    # to match the signal max_mean_token_load already uses.
    # ─────────────────────────────────────────────────────────────────────
    MAXMIN_CACHE = CACHE_DIR / "sweep_maxmin.parquet"
    MAXMIN_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if USE_SWEEP_CSV:
        # figure20_sweep.csv already carries the max/min columns, derived from
        # the same instance_metrics.parquet with the same +1 smoothing.
        derived = sweep_df[[
            "model", "num_instances", "cache_size", "load_metric", "policy_label",
            "max_min_token_load", "max_min_req_count", "max_min_input_tokens",
        ]].copy()
        print(f"Took {len(derived):,} max/min rows from {SWEEP_CSV}")
    elif not CACHED or not MAXMIN_CACHE.exists():
        rows = []
        for cell in sorted(SWEEP_DIR.iterdir()):
            if not cell.is_dir():
                continue
            name = cell.name
            try:
                model, rest = name.split("_N", 1)
                N_s, rest = rest.split("_cache", 1)
                cache_s, lm = rest.split("_lm_", 1)
            except ValueError:
                continue
            for pol_dir in cell.glob("policy=*"):
                policy = pol_dir.name.split("=", 1)[1]
                ip = pol_dir / "instance_metrics.parquet"
                if not ip.exists():
                    continue
                inst = pd.read_parquet(ip, engine="pyarrow")
                atps = inst["active_tokens_per_sec_mean"].astype(float)
                rc   = inst["request_count"].astype(float)
                in_tok = inst["input_tokens"].astype(float)
                rows.append({
                    "model": model,
                    "num_instances": int(N_s),
                    "cache_size": int(cache_s),
                    "load_metric": lm,
                    "policy_label": policy,
                    # +1 smoothing so a fully-starved instance (min=0, common for sticky)
                    # produces a finite-but-huge ratio instead of NaN, keeping the line plottable.
                    "max_min_token_load":   float((atps.max()   + 1.0) / (atps.min()   + 1.0)),
                    "max_min_req_count":    float((rc.max()     + 1.0) / (rc.min()     + 1.0)),
                    "max_min_input_tokens": float((in_tok.max() + 1.0) / (in_tok.min() + 1.0)),
                })
        derived = pd.DataFrame(rows)
        derived.to_parquet(MAXMIN_CACHE, engine="pyarrow")
        print(f"Saved {len(derived):,} rows -> {MAXMIN_CACHE}")
    else:
        derived = pd.read_parquet(MAXMIN_CACHE)
        print(f"Loaded {len(derived):,} rows from {MAXMIN_CACHE}")

    JOIN_KEYS = ["model", "num_instances", "cache_size", "load_metric", "policy_label"]
    # Drop any prior max/min columns so re-running this cell stays idempotent.
    sweep_df = sweep_df.drop(
        columns=[c for c in ("max_min_token_load", "max_min_req_count", "max_min_input_tokens")
                 if c in sweep_df.columns],
        errors="ignore",
    ).merge(derived, on=JOIN_KEYS, how="left")

    print("\nmax/min stats by policy:")
    print(
        sweep_df.groupby("policy_label")[
            ["max_min_token_load", "max_min_req_count", "max_min_input_tokens"]
        ].agg(["mean", "median", "max"]).round(2)
    )

    # ─────────────────────────────────────────────────────────────────────
    # Style maps (source cell 4, style-map block)
    # Policy/load-metric/model style dictionaries plus the sorted sweep
    # axes. The 2x2 facet ablation helpers from the source cell are not
    # needed for the paper figures and are omitted.
    # ─────────────────────────────────────────────────────────────────────
    POLICY_ORDER  = ["round_robin", "load_first", "cache_first", "sticky"]
    POLICY_LABELS = {
        "round_robin": "Round-robin",
        "load_first":  "Load-first",
        "cache_first": "Cache-first",
        "sticky":      "Sticky",
    }
    POLICY_COLORS = {
        "round_robin": "#7f7f7f",
        "load_first":  "#1f77b4",
        "cache_first": "#d62728",
        "sticky":      "#2ca02c",
    }
    POLICY_MARKERS = {
        "round_robin": "o",
        "load_first":  "s",
        "cache_first": "^",
        "sticky":      "D",
    }

    LOAD_METRIC_ORDER  = ["request_count", "input_tokens", "total_tokens", "request_duration"]
    LOAD_METRIC_LABELS = {
        "request_count":    "load = req count",
        "input_tokens":     "load = input tokens",
        "total_tokens":     "load = total tokens",
        "request_duration": "load = duration",
    }

    MODEL_LABELS = {
        "minimax": "MiniMax-M2.5-TEE",
        "v32":     "DeepSeek-V3.2-TEE",
    }

    # Pre-computed sorted axes for convenient reuse.
    CACHE_SIZES_SORTED = sorted(sweep_df["cache_size"].unique())
    N_VALUES_SORTED    = sorted(sweep_df["num_instances"].unique())
    print("cache sizes:", CACHE_SIZES_SORTED)
    print("N values:   ", N_VALUES_SORTED)

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_paper_legend.pdf
    # Standalone policy legend shared by all Part B panels. The current
    # source notebook no longer carries the cell that wrote it, so it is
    # rebuilt here from the style maps with the source notebook's own
    # save_legend helper (output matches the fig_paper_legend.pdf shipped
    # with the paper).
    # ─────────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0],
               color=POLICY_COLORS[p], marker=POLICY_MARKERS[p],
               markersize=6, label=POLICY_LABELS[p])
        for p in POLICY_ORDER
    ]
    legend_labels = [POLICY_LABELS[p] for p in POLICY_ORDER]
    save_legend(legend_handles, legend_labels,
                PDF_DIR / "fig_paper_legend.pdf", ncol=len(legend_labels))
    print("saved", PDF_DIR / "fig_paper_legend.pdf")

    # ─────────────────────────────────────────────────────────────────────
    # Paper figure helper (source cell 15)
    # Single-panel line plot fixed to load_metric = total_tokens; no titles.
    # ─────────────────────────────────────────────────────────────────────
    PAPER_LOAD_METRIC = "total_tokens"
    # Flat output for this repro script.
    PAPER_OUT_DIR = PDF_DIR

    def _paper_lines(
        df, *, x_col, y_col, xlabel, ylabel, file_stem,
        xscale="linear", yscale="linear",
        xlim=None, ylim=None, ypad=0.06,
        legend_loc="best", show_legend=False,
        save=True,
        policies=None,  # optional subset of POLICY_ORDER; None = all
    ):
        """Single-panel line plot, one line per policy.

        Dynamic limits: if ``xlim`` is set we restrict the plotted dataframe
        to that window first, then auto-derive ``ylim`` from the y-values
        inside that window (unless ``ylim`` is also passed).
        Set ``show_legend=False`` to suppress the on-figure legend.
        """
        # make dir file_stem
        if "/" in file_stem:
            (FINAL_OUT_DIR / file_stem.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
        plot_df = df
        if xlim is not None:
            lo, hi = xlim
            plot_df = plot_df[(plot_df[x_col] >= lo) & (plot_df[x_col] <= hi)]

        if ylim is None and len(plot_df) > 0:
            y = plot_df[y_col].dropna()
            if len(y) > 0:
                ymin, ymax = float(y.min()), float(y.max())
                span = ymax - ymin
                if span <= 0:
                    pad = max(abs(ymax) * 0.05, 1e-9)
                    ylim = (ymin - pad, ymax + pad)
                elif yscale == "log":
                    ylim = (ymin / (1 + ypad), ymax * (1 + ypad))
                else:
                    ylim = (ymin - ypad * span, ymax + ypad * span)

        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        plotted_policies = policies if policies is not None else POLICY_ORDER
        for policy in plotted_policies:
            sub = plot_df[plot_df["policy_label"] == policy].sort_values(x_col)
            if sub.empty:
                continue
            ax.plot(
                sub[x_col], sub[y_col],
                marker=POLICY_MARKERS[policy], markersize=6,
                color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xscale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        if yscale == "log":
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.tick_params(axis="both", which="major", length=4.5, width=0.8)
        ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)
        style_grid(ax)

        if show_legend:
            ax.legend(loc=legend_loc, frameon=False, fontsize=10,
                      handlelength=1.8, columnspacing=1.0)

        fig.tight_layout(pad=0.4)
        out = PAPER_OUT_DIR / f"{file_stem}.pdf"
        if save:
            fig.savefig(out, bbox_inches="tight", dpi=300)
            print(f"saved {out}")
        plt.close(fig)
        return out if save else None

    paper_df = sweep_df[sweep_df["load_metric"] == PAPER_LOAD_METRIC]
    print(f"paper_df: {len(paper_df):,} rows  ({sorted(paper_df['model'].unique())})")

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_paper_hitrate_vs_cache_FINAL_{v32,minimax}_N20_.pdf
    # (source cell 16)
    # Token hit ratio vs. per-instance cache size at N = 20. The sweep loop
    # (save=False) prints every (model, N) combination for constant-picking;
    # only the final pair is saved.
    # ─────────────────────────────────────────────────────────────────────
    # ── Paper Figure 1: token hit ratio vs cache size ─────────────────────
    # Sweep: scan every (model, N) so you can pick the constant for the
    # final pair below.
    for model in sorted(paper_df["model"].unique()):
        for N in N_VALUES_SORTED:
            sub = paper_df[(paper_df["model"] == model) & (paper_df["num_instances"] == N)]
            print(f"[Fig 1 sweep] hit ratio vs cache  ·  model={model}  ·  N={N}")
            _paper_lines(
                sub,
                x_col="cache_size", y_col="token_hit_rate",
                xlabel="Instance cache size",
                ylabel="Token hit ratio",
                file_stem=f"fig_paper_hitrate_vs_cache_{model}_N{N}",
                xscale="log",
                legend_loc="lower right",
                save=False,
            )

    # ── Final pair: one hardcoded constant drives both the filter and the filename ─
    print("\n=== Final pair ===")
    FINAL_N = 20

    sub = paper_df[(paper_df["model"] == "minimax") & (paper_df["num_instances"] == FINAL_N)]
    print(f"[Fig 1 final] minimax · N={FINAL_N}")
    _paper_lines(
        sub,
        x_col="cache_size", y_col="token_hit_rate",
        xlabel="Instance cache size",
        ylabel="Token hit ratio",
        file_stem=f"fig_paper_hitrate_vs_cache_FINAL_minimax_N{FINAL_N}_",  # trailing _ matches the paper-embedded filename
        xscale="log",
        show_legend=False,
    )

    sub = paper_df[(paper_df["model"] == "v32") & (paper_df["num_instances"] == FINAL_N)]
    print(f"[Fig 1 final] v32 · N={FINAL_N}")
    _paper_lines(
        sub,
        x_col="cache_size", y_col="token_hit_rate",
        xlabel="Instance cache size",
        ylabel="Token hit ratio",
        file_stem=f"fig_paper_hitrate_vs_cache_FINAL_v32_N{FINAL_N}_",  # trailing _ matches the paper-embedded filename
        xscale="log",
        show_legend=False,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_paper_repl_vs_N_FINAL_{v32,minimax}_cache25000.pdf
    # (source cell 17)
    # Effective KV replication ratio vs. instance count at cache = 25,000.
    # ─────────────────────────────────────────────────────────────────────
    # ── Paper Figure 2: replication ratio vs # instances ─────────────────
    for model in sorted(paper_df["model"].unique()):
        for C in CACHE_SIZES_SORTED:
            sub = paper_df[(paper_df["model"] == model) & (paper_df["cache_size"] == C)]
            print(f"[Fig 2 sweep] replication vs N  ·  model={model}  ·  cache={C:,}")
            _paper_lines(
                sub,
                x_col="num_instances", y_col="effective_replication_ratio",
                xlabel="# instances",
                ylabel="Replication ratio",
                file_stem=f"fig_paper_repl_vs_N_{model}_cache{C}",
                legend_loc="best",
                save=False,
            )

    # ── Final pair: one hardcoded constant drives both the filter and the filename ─
    print("\n=== Final pair ===")
    FINAL_CACHE = 25000

    sub = paper_df[(paper_df["model"] == "minimax") & (paper_df["cache_size"] == FINAL_CACHE)]
    print(f"[Fig 2 final] minimax · cache={FINAL_CACHE}")
    _paper_lines(
        sub,
        x_col="num_instances", y_col="effective_replication_ratio",
        xlabel="# instances",
        ylabel="Replication ratio",
        file_stem=f"fig_paper_repl_vs_N_FINAL_minimax_cache{FINAL_CACHE}",
        show_legend=False,
    )

    sub = paper_df[(paper_df["model"] == "v32") & (paper_df["cache_size"] == FINAL_CACHE)]
    print(f"[Fig 2 final] v32 · cache={FINAL_CACHE}")
    _paper_lines(
        sub,
        x_col="num_instances", y_col="effective_replication_ratio",
        xlabel="# instances",
        ylabel="Replication ratio",
        file_stem=f"fig_paper_repl_vs_N_FINAL_v32_cache{FINAL_CACHE}",
        show_legend=False,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_paper_repl_vs_cache_FINAL_{v32,minimax}_N20.pdf
    # (source cell 18)
    # Effective KV replication ratio vs. cache size at N = 20.
    # ─────────────────────────────────────────────────────────────────────
    # ── Paper Figure 3: replication ratio vs cache size ──────────────────
    for model in sorted(paper_df["model"].unique()):
        for N in N_VALUES_SORTED:
            sub = paper_df[(paper_df["model"] == model) & (paper_df["num_instances"] == N)]
            print(f"[Fig 3 sweep] replication vs cache  ·  model={model}  ·  N={N}")
            _paper_lines(
                sub,
                x_col="cache_size", y_col="effective_replication_ratio",
                xlabel="Instance cache size",
                ylabel="Replication ratio",
                file_stem=f"fig_paper_repl_vs_cache_{model}_N{N}",
                xscale="log",
                legend_loc="upper right",
                save=False,
            )

    # ── Final pair: one hardcoded constant drives both the filter and the filename ─
    print("\n=== Final pair ===")
    FINAL_N = 20

    sub = paper_df[(paper_df["model"] == "minimax") & (paper_df["num_instances"] == FINAL_N)]
    print(f"[Fig 3 final] minimax · N={FINAL_N}")
    _paper_lines(
        sub,
        x_col="cache_size", y_col="effective_replication_ratio",
        xlabel="Instance cache size",
        ylabel="Replication ratio",
        file_stem=f"fig_paper_repl_vs_cache_FINAL_minimax_N{FINAL_N}",
        xscale="log",
        show_legend=False,
    )

    sub = paper_df[(paper_df["model"] == "v32") & (paper_df["num_instances"] == FINAL_N)]
    print(f"[Fig 3 final] v32 · N={FINAL_N}")
    _paper_lines(
        sub,
        x_col="cache_size", y_col="effective_replication_ratio",
        xlabel="Instance cache size",
        ylabel="Replication ratio",
        file_stem=f"fig_paper_repl_vs_cache_FINAL_v32_N{FINAL_N}",
        xscale="log",
        show_legend=False,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Figure: fig_paper_imbalance_vs_N_FINAL_v32_cache5000000.pdf +
    #         fig_lb_minimax_imbalance.pdf (source cell 20)
    # Load imbalance (%) vs. instance count at cache = 5,000,000, sticky
    # excluded. The MiniMax panel is the plot the paper embeds as
    # minimaximbal.png, so it is additionally copied to its canonical repro
    # name fig_lb_minimax_imbalance.pdf.
    #
    # Reproducibility note: against the checked-in sweep output, the
    # round-robin and load-first lines match the paper screenshot exactly;
    # the cache-first line differs slightly (the screenshot appears to have
    # been exported from an earlier in-notebook state of the cache-first
    # run). This block reproduces the current source notebook faithfully.
    # ─────────────────────────────────────────────────────────────────────
    # ── Paper Figure 5: load imbalance vs # instances ─────────────────────
    # Load imbalance is expressed as a percentage: (max/min - 1) * 100, so a
    # perfectly balanced fleet is 0% and a 2% reading means the busiest
    # instance carries 2% more token load than the least-busy one. Source
    # ratio comes from instance_metrics.parquet via ablate-derive-maxmin:
    #   (max(active_tokens_per_sec_mean) + 1) / (min(active_tokens_per_sec_mean) + 1)
    # Sticky is excluded — its tail-starvation dwarfs every balanced policy
    # and makes the figure unreadable.
    FIG5_POLICIES = [p for p in POLICY_ORDER if p != "sticky"]

    def _add_imbalance_pct(df):
        out = df.copy()
        out["load_imbalance_pct"] = (out["max_mean_token_load"] - 1.0) * 100.0
        return out

    for model in sorted(paper_df["model"].unique()):
        for C in CACHE_SIZES_SORTED:
            sub = paper_df[(paper_df["model"] == model) & (paper_df["cache_size"] == C)]
            print(f"[Fig 5 sweep] load imbalance vs N  ·  model={model}  ·  cache={C:,}")
            sub = sub[paper_df["policy_label"] != "sticky"]
            sub = _add_imbalance_pct(sub)

            _paper_lines(
                sub,
                x_col="num_instances", y_col="load_imbalance_pct",
                xlabel="# instances",
                ylabel="Load imbalance (%)",
                file_stem=f"fig_paper_imbalance_vs_N_{model}_cache{C}",
                policies=FIG5_POLICIES,
                legend_loc="best",
                save=False,
            )

    # ── Final pair: one hardcoded constant drives both the filter and the filename ─
    print("\n=== Final pair ===")
    FINAL_CACHE = 5000000

    sub = paper_df[(paper_df["model"] == "minimax") & (paper_df["cache_size"] == FINAL_CACHE)]
    sub = sub[paper_df["policy_label"] != "sticky"]
    sub = _add_imbalance_pct(sub)
    print(f"[Fig 5 final] minimax · cache={FINAL_CACHE}")
    out_minimax = _paper_lines(
        sub,
        x_col="num_instances", y_col="load_imbalance_pct",
        xlabel="# instances",
        ylabel="Load imbalance (%)",
        file_stem=f"fig_paper_imbalance_vs_N_FINAL_minimax_cache{FINAL_CACHE}",
        show_legend=False,
        policies=FIG5_POLICIES,
    )

    sub = paper_df[(paper_df["model"] == "v32") & (paper_df["cache_size"] == FINAL_CACHE)]
    sub = sub[paper_df["policy_label"] != "sticky"]
    sub = _add_imbalance_pct(sub)
    print(f"[Fig 5 final] v32 · cache={FINAL_CACHE}")
    _paper_lines(
        sub,
        x_col="num_instances", y_col="load_imbalance_pct",
        xlabel="# instances",
        ylabel="Load imbalance (%)",
        file_stem=f"fig_paper_imbalance_vs_N_FINAL_v32_cache{FINAL_CACHE}",
        show_legend=False,
        policies=FIG5_POLICIES,
    )

    # The paper embeds the minimax panel as minimaximbal.png; keep a copy
    # under its canonical repro name as well.
    shutil.copyfile(out_minimax, PDF_DIR / "fig_lb_minimax_imbalance.pdf")
    print("saved", PDF_DIR / "fig_lb_minimax_imbalance.pdf")


# ════════════════════════════════════════════════════════════════════════════
# Part gating
# ════════════════════════════════════════════════════════════════════════════
if args.part in ("a", "all"):
    print("=== Part A — trace analysis ===")
    run_part_a()
if args.part in ("b", "all"):
    print("=== Part B — simulator sweep figures ===")
    run_part_b()
print("done.")
