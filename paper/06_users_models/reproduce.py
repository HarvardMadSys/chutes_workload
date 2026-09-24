#!/usr/bin/env python3
"""§4 Users and Models: the section's ten paper figures.

Figures (figures/paper/06_users_models/):
   1 fig_model_requests_vs_users_density.pdf        models: requests vs. distinct users
   2 fig_user_requests_distinct_models_heatmap.pdf  users: requests vs. distinct models
   3 fig_user_median_input_output_heatmap.pdf       users: median input vs. median output tokens
   4 fig_model_median_input_output_heatmap.pdf      models: median input vs. median output tokens
   5 fig_user_raster_periodic.pdf                   31 days of one user's requests, one row per model
   6 fig_user_raster_explore.pdf                    the same for a second user
   7 fig_model_raster_qwen3_next_80b.pdf            31 days of one model's requests, one row per user
   8 fig_model_raster_gpt_oss_20b.pdf               the same for a second model
   9 fig_model_heatmap_fixedxy_density.pdf          models: requests per hour vs. hourly CV of
                                                    inter-arrival times (IATs)
  10 fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf
                                                    the same axes, colored by lag-1 IAT autocorrelation

Query results are cached in output/paper/cache/ and reused on the next run;
--recompute reruns every query against the trace.

usage: python paper/06_users_models/reproduce.py [--db PATH] [--recompute]
"""
import matplotlib
matplotlib.use("Agg")  # headless; before pyplot is imported

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import LazyDB, trace_day  # noqa: E402
from _paths import CHUTE_MODELS_CSV, DB_DEFAULT, section_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--db", default=DB_DEFAULT,
                    help="DuckDB view over the trace (scripts/build_trace_view.py; table all_metrics_user)")
parser.add_argument("--recompute", action="store_true",
                    help="rerun every query even if its cache file exists")
args = parser.parse_args()
CACHED = not args.recompute

PDF_DIR, CACHE_DIR = section_paths("06_users_models")
con = LazyDB(args.db)  # the trace is opened by the first query that is not cached
TABLE_NAME = "all_metrics_user"
chute_models = pd.read_csv(CHUTE_MODELS_CSV)
CHUTE_TO_MODEL = dict(zip(chute_models["chute_id"], chute_models["name"]))

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


# ── Helpers ─────────────────────────────────────────────────────────────────

def query(sql):
    return con.sql(sql).fetchdf()


def cached(name, compute):
    """The DataFrame in CACHE_DIR/name; if it is missing (or --recompute), compute() and save it there."""
    path = CACHE_DIR / name
    if CACHED and path.exists():
        df = pd.read_parquet(path)
        print(f"Loaded {len(df):,} rows from {name}")
        return df
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Saved {len(df):,} rows to {name}")
    return df


def cached_query(name, sql):
    return cached(name, lambda: query(sql))


def sql_in(values):
    """A SQL list of string literals, e.g. ('a', 'b'), for `x IN ...`."""
    return "(" + ", ".join("'" + str(v).replace("'", "''") + "'" for v in values) + ")"


def batch_plan(ids, sizes):
    """Split ids into consecutive batches. sizes is [(limit, size), ...]: a batch that starts
    before index `limit` holds `size` ids; the last limit is None."""
    batches, start = [], 0
    while start < len(ids):
        size = next(s for limit, s in sizes if limit is None or start < limit)
        batches.append(ids[start:start + size])
        start += size
    return batches


def request_rank(col):
    """Every user_id or chute_id (col) with its request count, busiest first. Used to batch queries."""
    entity = {"user_id": "user", "chute_id": "model"}[col]
    return cached_query(f"{entity}_request_rank.parquet", f"""
        SELECT {col}, COUNT(*) AS n_requests
        FROM {TABLE_NAME}
        WHERE {col} IS NOT NULL
        GROUP BY {col}
        ORDER BY n_requests DESC
    """)


def save_fig(fig, name, pad=0.4):
    fig.tight_layout(pad=pad)
    fig.savefig(PDF_DIR / name, bbox_inches="tight", dpi=300)
    print(f"Saved {PDF_DIR / name}")
    plt.close(fig)


def log_log_grid(ax):
    """Log ticks on both axes: labelled decades, unlabelled minor ticks, dashed grid."""
    ax.set_axisbelow(True)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=10))
        axis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.35)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.4, alpha=0.12)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)


def median_tokens(col, batch_size, min_requests):
    """Median input and output tokens per user_id or chute_id (col), over requests with both > 0.
    The 10 busiest are queried one at a time, the rest batch_size at a time (all together if None)."""
    ids = request_rank(col)[col].tolist()
    batches = batch_plan(ids, [(10, 1), (None, batch_size or len(ids))])
    parts = []
    for k, batch in enumerate(batches, 1):
        parts.append(query(f"""
            SELECT {col},
                   COUNT(*) AS n_requests,
                   approx_quantile(it, 0.5) AS input_median,
                   approx_quantile(ot, 0.5) AS output_median
            FROM {TABLE_NAME}
            WHERE {col} IN {sql_in(batch)}
              AND it IS NOT NULL AND it > 0
              AND ot IS NOT NULL AND ot > 0
            GROUP BY {col}
            HAVING COUNT(*) > {min_requests}
        """))
        print(f"  batch {k}/{len(batches)}: {len(batch):,} ids -> {len(parts[-1]):,} rows")
    return pd.concat(parts, ignore_index=True)


# ── Figure 1: fig_model_requests_vs_users_density.pdf (fig:volume_breadth_models) ──
# Models: requests vs. distinct users, hexbin colored by the number of models.

def iat_per_model():
    """Inter-arrival-time stats of every model with at least 100 positive IATs. Busy models make the
    LAG window memory-heavy, so ranks 1-5 are queried alone, 6-20 in pairs, the rest 100 at a time."""
    ids = request_rank("chute_id")["chute_id"].tolist()
    batches = batch_plan(ids, [(5, 1), (20, 2), (None, 100)])
    parts = []
    for k, batch in enumerate(batches, 1):
        parts.append(query(f"""
            WITH iats AS (
                SELECT chute_id,
                       EXTRACT(EPOCH FROM (started_at - LAG(started_at) OVER (
                           PARTITION BY chute_id ORDER BY started_at
                       ))) AS iat_sec
                FROM {TABLE_NAME}
                WHERE chute_id IN {sql_in(batch)}
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
        """))
        print(f"  IAT batch {k}/{len(batches)}: {len(batch)} models -> {len(parts[-1])} rows")
    return pd.concat(parts, ignore_index=True)


model_counts = cached_query("model_requests_vs_users.parquet", f"""
    SELECT chute_id,
           COUNT(*)                AS n_requests,
           COUNT(DISTINCT user_id) AS n_users
    FROM {TABLE_NAME}
    WHERE chute_id IS NOT NULL AND user_id IS NOT NULL
    GROUP BY chute_id
""")
# Only models in the IAT table are drawn, i.e. models with at least 100 positive inter-arrival times.
df = model_counts.merge(cached("iat_per_model.parquet", iat_per_model), on="chute_id")

fig, ax = plt.subplots(figsize=(3.8, 3.0))
hb = ax.hexbin(df['n_requests'], df['n_users'], xscale='log', yscale='log', gridsize=22, mincnt=1,
               cmap='viridis', norm=LogNorm(), linewidths=0)
ax.set_xlabel('Requests per model')
ax.set_ylabel('Distinct users')
ax.set_ylim(1, df['n_users'].max() * 1.5)
fig.colorbar(hb, ax=ax, pad=0.02).set_label('Models')
save_fig(fig, 'fig_model_requests_vs_users_density.pdf')


# ── Figure 2: fig_user_requests_distinct_models_heatmap.pdf (fig:volume_breadth_users) ──
# Users with more than 10 requests and more than one model: requests vs. distinct models.

users = cached_query("user_requests_distinct_models.parquet", f"""
    SELECT user_id,
           COUNT(*) AS n_requests,
           approx_count_distinct(chute_id) AS n_models
    FROM {TABLE_NAME}
    WHERE user_id IS NOT NULL AND chute_id IS NOT NULL
    GROUP BY user_id
    HAVING COUNT(*) > 10 AND approx_count_distinct(chute_id) > 1
""")

fig, ax = plt.subplots(figsize=(3.8, 3.0))
hb = ax.hexbin(users['n_requests'], users['n_models'], xscale='log', yscale='log', gridsize=24, mincnt=1,
               cmap='viridis', norm=LogNorm(), linewidths=0)
ax.set_xlabel('Requests per user')
ax.set_ylabel('Distinct models')
log_log_grid(ax)
fig.colorbar(hb, ax=ax, pad=0.02).set_label('Users')
save_fig(fig, 'fig_user_requests_distinct_models_heatmap.pdf')


# ── Figure 3: fig_user_median_input_output_heatmap.pdf (fig:token_shape_users) ──
# Users with more than 10 requests: median input vs. median output tokens, with the line y = x.

user_tokens = cached("user_median_input_output.parquet",
                     lambda: median_tokens("user_id", batch_size=7_500, min_requests=10))

fig, ax = plt.subplots(figsize=(3.8, 3.0))
hb = ax.hexbin(user_tokens['input_median'], user_tokens['output_median'], xscale='log', yscale='log',
               gridsize=24, mincnt=1, cmap='viridis', norm=LogNorm(), linewidths=0)
ax.set_xlabel('Median user input')
ax.set_ylabel('Median user output')
log_log_grid(ax)
ax.set_xlim(1, 1_000_000)
ax.set_ylim(1, 100_000)
ax.plot([1, 100_000], [1, 100_000], color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
fig.colorbar(hb, ax=ax, pad=0.02).set_label('Users')
save_fig(fig, 'fig_user_median_input_output_heatmap.pdf')


# ── Figure 4: fig_model_median_input_output_heatmap.pdf (fig:token_shape_models) ──
# Models: median input vs. median output tokens, with the line y = x.

model_tokens = cached("model_median_input_output.parquet",
                      lambda: median_tokens("chute_id", batch_size=None, min_requests=0))

fig, ax = plt.subplots(figsize=(3.8, 3.0))
hb = ax.hexbin(model_tokens['input_median'], model_tokens['output_median'], xscale='log', yscale='log',
               gridsize=24, mincnt=1, cmap='viridis', norm=LogNorm(), linewidths=0)
ax.set_xlabel('Median model input')
ax.set_ylabel('Median model output')
log_log_grid(ax)
top = max(ax.get_xlim()[1], ax.get_ylim()[1])  # the same range on both axes, so y = x is the diagonal
ax.set_xlim(1, top)
ax.set_ylim(1, top)
ax.plot([1, 100_000], [1, 100_000], color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
fig.colorbar(hb, ax=ax, pad=0.02).set_label('Models')
save_fig(fig, 'fig_model_median_input_output_heatmap.pdf')


# ── Access rasters (figures 5-8) ────────────────────────────────────────────
# One dot per request over the 31 days after the first request in the window: a user's requests,
# one row per model (5, 6), or a model's requests, one row per user (7, 8). Rows are numbered in
# order of first use. Timestamps count from the start of the trace, 1970-01-01 (day 0).
# The windows start and end at the same time of day as the model windows in
# config/workload.json; the paper's user and model picks depend on these exact bounds.
USER_WINDOW = ("1970-06-16 19:59:59.984705", "1970-07-23 19:59:59.984705")   # trace days 166-203
MODEL_WINDOW = ("1970-06-16 19:59:59.984705", "1970-08-16 19:59:59.984705")  # trace days 166-227
RASTER_DAYS = 31
MODEL_RASTER_CAP = 800
MODEL_RASTER_SEED = 20260504


def day_tag(window):
    """'d166_d203': the trace days of a window's bounds, for cache names."""
    return "_".join(f"d{trace_day(t)}" for t in window)


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")[:90] or "unknown"


def pick(df, column, value):
    """The one row of df whose `column` equals value."""
    rows = df[df[column] == value]
    if len(rows) != 1:
        raise SystemExit(f"expected one row with {column} == {value!r}, found {len(rows)}")
    return rows.iloc[0]


def first_days(df):
    """The requests of the RASTER_DAYS days after df's first request, with `day` = days since it."""
    day = (df["started_at"] - df["started_at"].min()).dt.total_seconds() / 86400
    return df.assign(day=day)[day < RASTER_DAYS]


def plot_raster(df, y, ylabel, title, name):
    """One dot per request: day (x) vs. raster row y (a model or user index)."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    # Integer row indices take tab20's 20 colors in turn.
    ax.scatter(df["day"], df[y], s=4, alpha=0.6, linewidths=0,
               c=plt.cm.tab20(df[y].astype(int) % 20), rasterized=True)
    ax.set_xlim(0, RASTER_DAYS)
    ax.set_xticks(np.arange(0, RASTER_DAYS + 1, 6))
    ax.set_xlabel("Days")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    ax.set_title(title)
    save_fig(fig, name)


def top_users_in_window():
    """The 200 users with the most requests in USER_WINDOW, ranked 1-200."""
    df = query(f"""
        SELECT user_id,
               COUNT(*) AS n_requests,
               approx_count_distinct(chute_id) AS n_models,
               MIN(started_at) AS first_seen
        FROM {TABLE_NAME}
        WHERE user_id IS NOT NULL
          AND chute_id IS NOT NULL
          AND started_at >= '{USER_WINDOW[0]}'
          AND started_at < '{USER_WINDOW[1]}'
        GROUP BY user_id
        ORDER BY n_requests DESC
        LIMIT 200
    """)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def user_raster(rank, user_id):
    """The user's requests in USER_WINDOW; model_idx numbers the user's models in order of first use."""
    def compute():
        models = query(f"""
            SELECT chute_id, MIN(started_at) AS first_seen, COUNT(*) AS model_requests
            FROM {TABLE_NAME}
            WHERE user_id IN {sql_in([user_id])}
              AND chute_id IS NOT NULL
              AND started_at >= '{USER_WINDOW[0]}'
              AND started_at < '{USER_WINDOW[1]}'
            GROUP BY chute_id
            ORDER BY first_seen, chute_id
        """)
        models["model_idx"] = np.arange(len(models))
        requests = query(f"""
            SELECT started_at, chute_id
            FROM {TABLE_NAME}
            WHERE user_id IN {sql_in([user_id])}
              AND chute_id IS NOT NULL
              AND started_at >= '{USER_WINDOW[0]}'
              AND started_at < '{USER_WINDOW[1]}'
            ORDER BY started_at, invocation_id
        """)
        return requests.merge(models[["chute_id", "model_idx", "model_requests"]], on="chute_id", how="left")
    name = f"user_raster_rank{rank}_{slug(user_id)}_{day_tag(USER_WINDOW)}.parquet"
    return cached(f"user_usage_rasters/{name}", compute)


def draw_user_raster(rank, name):
    """User ids are anonymized, so a user is addressed by its request rank in USER_WINDOW."""
    user = pick(top_users, "rank", rank)
    df = first_days(user_raster(rank, user["user_id"]))
    plot_raster(df, "model_idx", "Model index",
                f"User rank {rank}\n{len(df):,} requests, {df['model_idx'].nunique():,} models", name)


def top_models_in_window():
    """The 200 models with the most requests in MODEL_WINDOW, ranked 1-200, with their names."""
    df = query(f"""
        SELECT chute_id,
               COUNT(*) AS n_requests,
               approx_count_distinct(user_id) AS n_users
        FROM {TABLE_NAME}
        WHERE chute_id IS NOT NULL
          AND user_id IS NOT NULL
          AND started_at >= '{MODEL_WINDOW[0]}'
          AND started_at < '{MODEL_WINDOW[1]}'
        GROUP BY chute_id
        ORDER BY n_requests DESC
        LIMIT 200
    """)
    df["model_name"] = df["chute_id"].map(CHUTE_TO_MODEL).fillna(df["chute_id"])
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def model_raster(rank, chute_id, model_name):
    """Requests to the model in MODEL_WINDOW from at most MODEL_RASTER_CAP of its users;
    user_idx numbers those users in order of first request."""
    def compute():
        users = query(f"""
            SELECT user_id, MIN(started_at) AS first_seen, COUNT(*) AS user_requests
            FROM {TABLE_NAME}
            WHERE chute_id IN {sql_in([chute_id])}
              AND user_id IS NOT NULL
              AND started_at >= '{MODEL_WINDOW[0]}'
              AND started_at < '{MODEL_WINDOW[1]}'
            GROUP BY user_id
            ORDER BY first_seen, user_id
        """)
        # A cap, not sampling: a seeded shuffle picks which MODEL_RASTER_CAP users stay (all, if fewer).
        users = (users.sample(frac=1, random_state=MODEL_RASTER_SEED + rank)
                 .head(MODEL_RASTER_CAP)
                 .sort_values(["first_seen", "user_id"])
                 .reset_index(drop=True))
        users["user_idx"] = np.arange(len(users))
        requests = query(f"""
            SELECT started_at, user_id
            FROM {TABLE_NAME}
            WHERE chute_id IN {sql_in([chute_id])}
              AND user_id IN {sql_in(users["user_id"])}
              AND started_at >= '{MODEL_WINDOW[0]}'
              AND started_at < '{MODEL_WINDOW[1]}'
            ORDER BY started_at, invocation_id
        """)
        return (requests.merge(users[["user_id", "user_idx", "user_requests"]], on="user_id", how="left")
                .drop(columns=["user_id"]))
    name = f"model_raster_rank{rank}_{slug(model_name)}_{day_tag(MODEL_WINDOW)}.parquet"
    return cached(f"user_usage_rasters/{name}", compute)


def draw_model_raster(model_name, name):
    """The title gives the model's request rank in MODEL_WINDOW."""
    model = pick(top_models, "model_name", model_name)
    rank = int(model["rank"])
    df = first_days(model_raster(rank, model["chute_id"], model_name))
    plot_raster(df, "user_idx", "User index", f"Model rank {rank}", name)


top_users = cached(f"user_raster_top200_users_{day_tag(USER_WINDOW)}.parquet", top_users_in_window)
top_models = cached(f"model_raster_top200_models_{day_tag(MODEL_WINDOW)}.parquet", top_models_in_window)

# ── Figure 5: fig_user_raster_periodic.pdf (fig:user_access_persistent) ──
# The rank-190 user: a stable set of models, called again and again through the month.
draw_user_raster(190, "fig_user_raster_periodic.pdf")

# ── Figure 6: fig_user_raster_explore.pdf (fig:user_access_exploration) ──
# The rank-134 user.
draw_user_raster(134, "fig_user_raster_explore.pdf")

# ── Figure 7: fig_model_raster_qwen3_next_80b.pdf (fig:model_access_power_user) ──
# Qwen3-Next-80B: long horizontal bands of power users who come back at regular intervals.
draw_model_raster("Qwen/Qwen3-Next-80B-A3B-Instruct", "fig_model_raster_qwen3_next_80b.pdf")

# ── Figure 8: fig_model_raster_gpt_oss_20b.pdf (fig:model_access_correlated) ──
# gpt-oss-20b: sparse at first, then dense vertical clusters of users active at the same time.
draw_model_raster("openai/gpt-oss-20b", "fig_model_raster_gpt_oss_20b.pdf")


# ── Burstiness table (figures 9, 10) ────────────────────────────────────────
# Per model: mean requests per hour and mean coefficient of variation (CV) of the inter-arrival
# times (IATs) within an hour, both over the model's valid hours (>= 10 requests, >= 5 positive
# IATs), for models with > 100 requests and >= 5 valid hours; plus the Spearman correlation of
# each IAT with the next one (lag-1 autocorrelation).
MIN_MODEL_REQUESTS = 100
MIN_HOUR_REQUESTS = 10
MIN_HOUR_IATS = 5
MIN_VALID_HOURS = 5
AUTOCORR_MAX_REQUESTS = 100_000_000  # busier models are too heavy for the autocorrelation; fig. 10 omits them


def hourly_burstiness():
    """One row per (model, hour with >= MIN_HOUR_REQUESTS requests): request count and IAT CV.
    Computed one trace month at a time; each month is its own parquet, so an interrupted run resumes."""
    batch_dir = CACHE_DIR / "_cache_model_hourly_burstiness_batches"
    batch_dir.mkdir(exist_ok=True)
    first, last = con.sql(f"""
        SELECT DATE_TRUNC('month', MIN(started_at))::DATE AS min_month,
               DATE_TRUNC('month', MAX(started_at))::DATE AS max_month
        FROM {TABLE_NAME}
        WHERE started_at IS NOT NULL
    """).fetchone()
    months = pd.date_range(pd.Timestamp(first), pd.Timestamp(last) + pd.offsets.MonthBegin(1), freq="MS")
    for start, end in zip(months[:-1], months[1:]):
        path = batch_dir / f"model_hourly_burstiness_{start:%Y%m}.parquet"
        if CACHED and path.exists():
            continue
        print(f"  hourly burstiness {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
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
                      AND started_at >= TIMESTAMP '{start:%Y-%m-%d}'
                      AND started_at <  TIMESTAMP '{end:%Y-%m-%d}'
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
    return query(f"""
        SELECT *
        FROM read_parquet('{batch_dir.as_posix()}/model_hourly_burstiness_*.parquet')
        ORDER BY chute_id, hour
    """)


def summarize_models(hourly):
    """Per model: request and user counts, and requests/hour and IAT CV averaged over its valid hours."""
    valid = hourly.dropna(subset=["cv_iat"]).replace([np.inf, -np.inf], np.nan)
    valid = valid[valid["cv_iat"] > 0]
    per_model = (valid.groupby("chute_id")
                 .agg(n_valid_hours=("hour", "count"),
                      cv_iat_mean=("cv_iat", "mean"),
                      cv_iat_median=("cv_iat", "median"),
                      mean_req_per_hour=("req_per_hour", "mean"))
                 .reset_index())
    out = model_counts.rename(columns={"n_requests": "total_requests"})
    out = out.merge(per_model, on="chute_id", how="left")
    out["model_name"] = out["chute_id"].map(CHUTE_TO_MODEL).fillna("unknown")
    out = out[(out["total_requests"] > MIN_MODEL_REQUESTS) & (out["n_valid_hours"] >= MIN_VALID_HOURS)]
    return out.sort_values("total_requests", ascending=False).reset_index(drop=True)


def lag1_autocorr(summary):
    """Per model: the Pearson and Spearman correlation of each IAT with the next one (figure 10 uses
    Spearman). Models are batched busiest first: the first 25 one at a time, then by 5 up to 100, by 100
    up to 1,000, then by 1,000. Each batch is its own parquet, so an interrupted run resumes."""
    batch_dir = CACHE_DIR / "_cache_model_iat_lag1_autocorr_where_batches"
    batch_dir.mkdir(exist_ok=True)
    ids = summary.loc[summary["total_requests"] < AUTOCORR_MAX_REQUESTS, "chute_id"].tolist()
    batches = batch_plan(ids, [(25, 1), (100, 5), (1000, 100), (None, 1000)])
    parts, start = [], 0
    for k, batch in enumerate(batches, 1):
        tag = f"batch_{k:04d}_models_{start + 1:05d}_{start + len(batch):05d}_n{len(batch):04d}"
        path = batch_dir / f"{tag}.parquet"
        start += len(batch)
        if not (CACHED and path.exists() and path.stat().st_size > 0):
            t0 = time.time()
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
                          AND chute_id IN {sql_in(batch)}
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
                ) TO '{path.as_posix()}' (FORMAT 'parquet')
            """)
            print(f"  autocorr batch {k}/{len(batches)}: {len(batch)} models, {time.time() - t0:.1f}s")
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


def model_burstiness():
    """The per-model table of figures 9 and 10 (see the comment above)."""
    summary = cached("_cache_model_burstiness_summary.parquet", lambda: summarize_models(hourly_burstiness()))
    autocorr = lag1_autocorr(summary)
    autocorr = autocorr[["chute_id", "n_iat_pairs", "spearman_lag1_autocorr"]]
    table = summary.merge(autocorr, on="chute_id", how="left")
    return table[["chute_id", "model_name", "mean_req_per_hour", "cv_iat_mean", "spearman_lag1_autocorr",
                  "total_requests", "n_users", "cv_iat_median", "n_iat_pairs"]]


def compact_number(x, _pos=None):
    """Tick label: 1k, 2M, 3B, ..."""
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


def style_grid_heatmap(ax):
    """The dashed major and minor grid of figures 9 and 10."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.32, linestyle="--", linewidth=0.55)
    ax.grid(True, which="minor", alpha=0.12, linestyle="--", linewidth=0.35)


def burstiness_axes(ax):
    """Labels, tick labels and grid of figures 9 and 10 (log x and log y)."""
    ax.set_xlabel("Requests/hour")
    ax.set_ylabel("Mean CV(IAT)")
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
    style_grid_heatmap(ax)


burstiness = cached("_cache_model_metric_autocorr_suite.parquet", model_burstiness)
# Both axes are logarithmic, so keep positive values only.
burstiness = burstiness.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_req_per_hour", "cv_iat_mean"])
burstiness = burstiness[(burstiness["mean_req_per_hour"] > 0) & (burstiness["cv_iat_mean"] > 0)]

# ── Figure 9: fig_model_heatmap_fixedxy_density.pdf (fig:model_burstiness_density) ──
# Models: requests per hour vs. mean hourly CV(IAT), hexbin colored by the number of models.

fig, ax = plt.subplots(figsize=(3.7, 3.2))
hb = ax.hexbin(burstiness["mean_req_per_hour"], burstiness["cv_iat_mean"], xscale="log", yscale="log",
               gridsize=32, mincnt=1, cmap="viridis", norm=LogNorm(), linewidths=0)
burstiness_axes(ax)
fig.colorbar(hb, ax=ax, pad=0.015).set_label("Models")
save_fig(fig, "fig_model_heatmap_fixedxy_density.pdf", pad=0.45)

# ── Figure 10: fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf (fig:model_burstiness_autocorr) ──
# The same axes, one dot per model, colored by the Spearman lag-1 IAT autocorrelation.

scored = burstiness.dropna(subset=["spearman_lag1_autocorr"])
fig, ax = plt.subplots(figsize=(3.7, 3.2))
sc = ax.scatter(scored["mean_req_per_hour"], scored["cv_iat_mean"], c=scored["spearman_lag1_autocorr"],
                s=12, alpha=0.68, linewidth=0, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
ax.set_xscale("log")
ax.set_yscale("log")
burstiness_axes(ax)
fig.colorbar(sc, ax=ax, pad=0.01).set_label("Lag-1 autocorr")
save_fig(fig, "fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf", pad=0.45)

print(f"Done: the §4 figures are in {PDF_DIR}")
