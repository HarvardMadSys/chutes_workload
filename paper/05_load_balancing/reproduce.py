#!/usr/bin/env python3
"""Load balancing in production: the trace figures of the paper's case study.

Every figure covers the trace's last two months, days 306 to 364. Time axes
count days from the start of the trace (day 0).

DeepSeek-V3-0324-TEE:
  fig_lb_request_rate.pdf        requests per hour
  fig_lb_active_instances.pdf    active instances per minute, median of each hour
  fig_lb_maxmean_tokens.pdf      max/mean token load across instances per minute,
                                 mean of each hour
  fig_lb_concurrency.pdf         requests per instance and minute over the 24 hours
                                 with the most active instances

DeepSeek-V3.2-TEE:
  fig_lb_ttft_p90_vs_load.pdf    P90 TTFT by instance load and input tokens
  fig_lb_decode_p90_vs_load.pdf  P90 decode time by instance load and output tokens
  fig_lb_user_zoom_top.pdf       the busiest hour of the busiest user whose token hit
                                 ratio that hour is at least 0.4: per instance, the
                                 user's share of requests and its token hit ratio
  fig_lb_user_zoom_min.pdf       the same for the least busy such user

Query results are cached in output/paper/cache/ and reused unless --recompute is
given. Each user zoom also runs one small trace query on every run.

Usage:
    python paper/05_load_balancing/reproduce.py [--db PATH] [--recompute]
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import LazyDB, trace_day  # noqa: E402
from _paths import CHUTE_MODELS_CSV, DB_DEFAULT, section_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the trace (scripts/build_trace_view.py)")
parser.add_argument("--recompute", action="store_true",
                    help="rerun the queries even if their cache files exist")
args = parser.parse_args()
CACHED = not args.recompute   # True: reuse cache files and only replot

PDF_DIR, CACHE_DIR = section_paths("05_load_balancing")
con = LazyDB(args.db)

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

# The trace's last two months, days 306 to 364 (WINDOW_START <= started_at < WINDOW_END).
# The bounds have the same time of day as the model windows in config/workload.json;
# the paper's picks (e.g. the two zoomed user-hours) depend on these exact bounds.
WINDOW_START = "1970-11-02 19:59:59.984705"
WINDOW_END = "1970-12-31 19:59:59.984705"

# Key used in cache file names -> model name in data/paper/chute_models.csv.
MODELS = {
    "deepseek_v3_0324": "deepseek-ai/DeepSeek-V3-0324-TEE",
    "deepseek_v32": "deepseek-ai/DeepSeek-V3.2-TEE",
}
_chutes = pd.read_csv(CHUTE_MODELS_CSV)
_chute_of = dict(zip(_chutes["name"], _chutes["chute_id"]))
CHUTE = {key: _chute_of[name] for key, name in MODELS.items()}
TITLE = {key: name.split("/", 1)[1] for key, name in MODELS.items()}  # "DeepSeek-V3.2-TEE"


def cached_query(name, sql):
    """Run `sql` on the trace and store the result as CACHE_DIR/name; later runs read the file."""
    path = CACHE_DIR / name
    if CACHED and path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    df = con.sql(sql).fetchdf()
    df.to_parquet(path)
    print(f"  {name}: {len(df):,} rows ({time.time() - t0:.0f} s)")
    return df


def save_fig(fig, name):
    path = PDF_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {path}")


def style_grid(ax):
    """Dashed major and minor grid lines behind the data."""
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linestyle="--", linewidth=0.4)


def abbr_count(x, _pos):
    """Tick label: 150000 -> 150k."""
    if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}k"
    return f"{int(x)}"


def day_of(ts):
    """Days since the trace start, at hourly resolution: 06:00 on day 303 -> 303.25."""
    return trace_day(ts) + ts.hour / 24


def index_tick_step(n):
    """Major tick step for an axis with one position per instance."""
    return 1 if n <= 12 else 5 if n <= 30 else 10


def instances_by_first_request(stats):
    """The instances in `stats`, in order of their first active minute."""
    return list(stats.groupby("instance_id")["min_ts"].min().sort_values().index)


# ════════════════════════════════════════════════════════════════════════════
# DeepSeek-V3-0324: request rate, active instances, max/mean load, concurrency
# ════════════════════════════════════════════════════════════════════════════
def instance_minute_stats(model):
    """Per instance and minute of the window: requests, token sums, TTFT and
    duration percentiles, distinct users."""
    return cached_query(f"instance_minute_stats_{model}.parquet", f"""
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
        FROM all_metrics_user
        WHERE chute_id    = '{CHUTE[model]}'
          AND instance_id IS NOT NULL
          AND started_at >= TIMESTAMP '{WINDOW_START}'
          AND started_at <  TIMESTAMP '{WINDOW_END}'
        GROUP BY 1, 2, 3
    """)


def per_minute(stats):
    """Per minute: active instances (those with a request), requests, and the
    busiest instance's token load divided by the mean instance's."""
    g = stats.groupby("min_ts")
    return pd.DataFrame({
        "n_active_instances": g.size(),
        "total_requests": g["req_count"].sum(),
        "max_mean_total_tokens": g["total_tokens"].max() / g["total_tokens"].mean(),
    })


def plot_hourly(series, ylabel, name, color, linewidth, yformatter=None):
    """One hourly series of DeepSeek-V3-0324 over the window."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.plot(day_of(series.index), series.values, color=color, linewidth=linewidth)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(TITLE["deepseek_v3_0324"])
    if yformatter is not None:
        ax.yaxis.set_major_formatter(yformatter)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    style_grid(ax)
    fig.tight_layout()
    save_fig(fig, name)


stats_v3 = instance_minute_stats("deepseek_v3_0324")
minutes_v3 = per_minute(stats_v3)

# fig_lb_request_rate.pdf: requests per hour.
plot_hourly(minutes_v3["total_requests"].resample("1h").sum(), "Requests / hour",
            "fig_lb_request_rate.pdf", color="C0", linewidth=1.3,
            yformatter=FuncFormatter(abbr_count))

# fig_lb_active_instances.pdf: active instances per minute, median of each hour
# (a count, not a rate; the paper's caption gives the aggregation).
plot_hourly(minutes_v3["n_active_instances"].resample("1h").median(), "Active instances",
            "fig_lb_active_instances.pdf", color="C2", linewidth=1.4)

# fig_lb_maxmean_tokens.pdf: max/mean token load per minute, mean of each hour,
# over the minutes with at least 2 active instances and 10 requests.
balanced = minutes_v3[(minutes_v3["n_active_instances"] >= 2)
                      & (minutes_v3["total_requests"] >= 10)]
plot_hourly(balanced["max_mean_total_tokens"].resample("1h").mean(), "Max/mean token load",
            "fig_lb_maxmean_tokens.pdf", color="C0", linewidth=1.6)


# fig_lb_concurrency.pdf: requests per instance and minute over the 24 hours
# with the most active instances per minute (on average, idle minutes count as 0).
# Rows are the instances in order of their first request in that window.
def densest_window_end(stats, hours):
    """End of the `hours`-wide window with the highest mean number of active
    instances per minute."""
    per_min = stats.groupby("min_ts")["instance_id"].nunique().sort_index()
    per_min = per_min.reindex(pd.date_range(per_min.index.min(), per_min.index.max(),
                                            freq="1min"), fill_value=0)
    return per_min.rolling(window=f"{hours}h", min_periods=hours * 60).mean().idxmax()


def fig_concurrency(stats, hours=24):
    win_start = (densest_window_end(stats, hours) - pd.Timedelta(hours=hours)).floor("h")
    win_end = win_start + pd.Timedelta(hours=hours)
    win = stats[(stats["min_ts"] >= win_start) & (stats["min_ts"] < win_end)]
    order = instances_by_first_request(win)
    counts = (win.groupby(["instance_id", "min_ts"])["req_count"].sum()
                 .unstack(fill_value=0)
                 .reindex(index=order,
                          columns=pd.date_range(win_start, win_end - pd.Timedelta("1min"),
                                                freq="1min"),
                          fill_value=0))
    arr = counts.to_numpy(dtype=float)
    arr = np.where(arr > 0, arr, np.nan)          # idle minutes stay blank
    vmin = max(1.0, np.nanpercentile(arr, 5))
    vmax = np.nanpercentile(arr, 99)
    n_inst = len(order)
    day0 = day_of(win_start)
    day1 = day0 + hours / 24

    fig, ax = plt.subplots(figsize=(4.2, 3.5), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis",
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   extent=[day0, day1, 0, n_inst], interpolation="nearest")
    ax.set_ylabel("Instance index")
    ax.set_xlabel("Day")
    ax.set_title(TITLE["deepseek_v3_0324"])

    # Colorbar ticks: vmin + 9, vmax and their midpoint, each rounded to a multiple of 10.
    cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.4)
    cb.set_label("Requests / min")
    vmin_tick = int(round((vmin + 9) / 10) * 10)
    vmax_tick = int(round(vmax / 10) * 10)
    vmid_tick = int(round(((vmin_tick + vmax_tick) / 2) / 10) * 10)
    cb.set_ticks([max(1, t) for t in sorted({vmin_tick, vmid_tick, vmax_tick})])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    cb.ax.yaxis.set_minor_locator(NullLocator())

    ax.set_ylim(0, n_inst)
    yticks = np.arange(0, n_inst + 1, index_tick_step(n_inst))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(t)) for t in yticks])
    ax.yaxis.set_minor_locator(NullLocator())

    ax.xaxis.set_major_locator(MultipleLocator(0.25))    # every 6 hours
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(MultipleLocator(1 / 12))  # every 2 hours
    ax.set_xlim(day0, day1)
    ax.grid(False, which="both")
    save_fig(fig, "fig_lb_concurrency.pdf")


fig_concurrency(stats_v3)


# ════════════════════════════════════════════════════════════════════════════
# DeepSeek-V3.2: P90 TTFT and decode time vs. instance load
# ════════════════════════════════════════════════════════════════════════════
def request_latency_vs_load(model):
    """One row per request: input and output tokens, TTFT and decode time, and
    the instance load: the tokens (it + ot) of all requests that started on the
    same instance in the same second."""
    return cached_query(f"request_latency_vs_load_{model}.parquet", f"""
    WITH filtered AS (
        SELECT instance_id,
               started_at,
               ttft,
               it,
               ot,
               EXTRACT(EPOCH FROM (completed_at - started_at)) AS dur_s,
               DATE_TRUNC('second', started_at)                AS sec_ts
        FROM all_metrics_user
        WHERE chute_id    = '{CHUTE[model]}'
          AND instance_id IS NOT NULL
          AND started_at >= TIMESTAMP '{WINDOW_START}'
          AND started_at <  TIMESTAMP '{WINDOW_END}'
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
    """)


def p90(values):
    return np.percentile(values, 90)


def plot_p90_hexbin(df, y_col, y_label, c_col, c_label, name):
    """Hexbin of instance load x `y_col`, colored by the P90 of `c_col` in each
    cell with at least 30 requests. The color scale is clipped to the 5th-95th
    percentile of `c_col` over all requests."""
    x = df["bucket_total_tokens"].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    c = df[c_col].to_numpy(dtype=float)
    keep = (x > 0) & (y > 0) & np.isfinite(c)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    hb = ax.hexbin(x[keep], y[keep], C=c[keep], reduce_C_function=p90,
                   xscale="log", yscale="log", gridsize=36, mincnt=30, cmap="magma_r",
                   vmin=np.nanpercentile(c[keep], 5), vmax=np.nanpercentile(c[keep], 95))
    cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.046)
    cb.set_label(c_label)
    ax.set_xlabel("Instance load (tokens/s)")
    ax.set_ylabel(y_label)
    ax.set_title(TITLE["deepseek_v32"])
    style_grid(ax)
    fig.tight_layout()
    save_fig(fig, name)


requests_v32 = request_latency_vs_load("deepseek_v32")

# fig_lb_ttft_p90_vs_load.pdf: P90 TTFT by instance load and input tokens.
plot_p90_hexbin(requests_v32, "req_input_tokens", "Input tokens",
                "req_ttft", "P90 TTFT (s)", "fig_lb_ttft_p90_vs_load.pdf")

# fig_lb_decode_p90_vs_load.pdf: P90 decode time (duration - TTFT) by instance
# load and output tokens.
plot_p90_hexbin(requests_v32, "req_output_tokens", "Output tokens",
                "req_gen_time", "P90 decode time (s)", "fig_lb_decode_p90_vs_load.pdf")


# ════════════════════════════════════════════════════════════════════════════
# DeepSeek-V3.2: one user's busiest hour, request share vs. token hit ratio
# ════════════════════════════════════════════════════════════════════════════
MIN_BUSY_HOUR_REQUESTS = 200
HIGH_HIT_RATIO = 0.4


def user_busiest_hours(model):
    """Per user: the clock hour with the most requests (the earliest on ties) and
    its token hit ratio sum(ct) / sum(it). Users whose busiest hour has fewer than
    MIN_BUSY_HOUR_REQUESTS requests are left out."""
    return cached_query(f"user_busiest_hour_{model}.parquet", f"""
        WITH per_user_hour AS (
            SELECT user_id,
                   DATE_TRUNC('hour', started_at) AS hr_ts,
                   COUNT(*)                          AS n_req,
                   COALESCE(SUM(it), 0)::BIGINT      AS it_sum,
                   COALESCE(SUM(ct), 0)::BIGINT      AS ct_sum
            FROM all_metrics_user
            WHERE chute_id = '{CHUTE[model]}'
              AND user_id  IS NOT NULL
              AND started_at >= TIMESTAMP '{WINDOW_START}'
              AND started_at <  TIMESTAMP '{WINDOW_END}'
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
          AND n_req >= {MIN_BUSY_HOUR_REQUESTS}
    """)


def user_requests(model, user_id, start, end):
    """The user's requests in [start, end): start time, instance, input and cached tokens."""
    return con.sql(f"""
        SELECT started_at, instance_id,
               COALESCE(it, 0)::DOUBLE AS it,
               COALESCE(ct, 0)::DOUBLE AS ct
        FROM all_metrics_user
        WHERE chute_id    = '{CHUTE[model]}'
          AND user_id     = '{user_id}'
          AND instance_id IS NOT NULL
          AND started_at >= TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}'
          AND started_at <  TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'
    """).fetchdf()


def fig_user_zoom(model, stats, pick, name):
    """Bars per instance active in the user's busiest hour (in order of first
    request): the share of the user's requests it served (left axis) and the
    user's token hit ratio sum(ct) / sum(it) on it (right axis)."""
    start = pd.Timestamp(pick["busy_hr"])
    end = start + pd.Timedelta(hours=1)
    in_hour = stats[(stats["min_ts"] >= start) & (stats["min_ts"] < end)]
    order = instances_by_first_request(in_hour)
    print(f"  {name}: user {pick['user_id']}, day {trace_day(start)} from {start:%H}:00, "
          f"{len(order)} instances")

    per_inst = (user_requests(model, pick["user_id"], start, end)
                .groupby("instance_id")
                .agg(n_req=("it", "size"), it=("it", "sum"), ct=("ct", "sum"))
                .reindex(order, fill_value=0))
    n_req, it, ct = (per_inst[c].to_numpy(dtype=float) for c in ("n_req", "it", "ct"))
    pct_req = 100.0 * n_req / n_req.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        hit_pct = np.clip(np.where(it > 0, 100.0 * ct / it, 0.0), 0.0, 100.0)
    n_inst = len(order)
    centers = np.arange(n_inst)

    # Left and right axes (twinx): request share and token hit ratio per instance.
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax_hit = ax.twinx()
    bars_req = ax.bar(centers - 0.2, pct_req, width=0.4,
                      color="C0", edgecolor="black", linewidth=0.25, label="User request")
    bars_hit = ax_hit.bar(centers + 0.2, hit_pct, width=0.4,
                          color="C3", edgecolor="black", linewidth=0.25, label="Cache hit")

    ax.set_xlim(-0.5, n_inst - 0.5)
    ax.set_xlabel("Instance index")
    ax.set_ylim(0, max(8.0, pct_req.max() * 1.25))
    ax.set_ylabel("User request (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))

    ax_hit.set_ylim(0, 100)
    ax_hit.set_ylabel("Token hit (%)")
    ax_hit.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax_hit.spines["right"].set_visible(True)
    ax_hit.spines["right"].set_color("black")
    ax_hit.spines["right"].set_linestyle("-")

    ax.xaxis.set_major_locator(MultipleLocator(index_tick_step(n_inst)))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.grid(True, axis="x", which="major", alpha=0.35, linestyle="--", linewidth=0.5)
    ax.grid(True, axis="x", which="minor", alpha=0.15, linestyle=":", linewidth=0.4)
    ax.grid(True, axis="y", which="major", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[bars_req, bars_hit], loc="upper left", frameon=False, fontsize=7,
              handlelength=1.2, handletextpad=0.4, borderaxespad=0.3)
    save_fig(fig, name)


stats_v32 = instance_minute_stats("deepseek_v32")
busy = user_busiest_hours("deepseek_v32")
# Ties in n_req are broken by the busiest hour, so a --recompute picks the same users.
high_hit = busy[busy["hit_rate"] >= HIGH_HIT_RATIO].sort_values(
    ["n_req", "busy_hr"], ascending=[False, True], kind="stable")

# fig_lb_user_zoom_top.pdf: the user with the most requests in its busiest hour.
fig_user_zoom("deepseek_v32", stats_v32, high_hit.iloc[0], "fig_lb_user_zoom_top.pdf")
# fig_lb_user_zoom_min.pdf: the user with the fewest (still >= MIN_BUSY_HOUR_REQUESTS).
fig_user_zoom("deepseek_v32", stats_v32, high_hit.iloc[-1], "fig_lb_user_zoom_min.pdf")
