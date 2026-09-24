# Trace-analysis figures

One directory per group of the paper's figures. Each `reproduce.py` queries
the trace view (`make trace`), caches every query result in
`output/paper/cache/`, and draws exactly the figures the paper includes into
`figures/paper/<directory>/`. Each directory's README lists its figures, what
each one shows, and the cache behind it.

| Directory | Paper section | Figures | Redraw without the trace |
|---|---|---|---|
| [`01_workload_overview/`](01_workload_overview/) | §2 Background | 5 | all 5 |
| [`02_token_shape_latency/`](02_token_shape_latency/) | §3 Production Trace Analysis | 8 | all 8 |
| [`03_workload_evolution/`](03_workload_evolution/) | §4 Users and Models, §5 Workload Evolution | 7 | 5 (not the two cohort figures) |
| [`04_prefix_caching/`](04_prefix_caching/) | §5 Workload Evolution, §6 Prefix Caching | 7 | 6 (not the TTL coverage) |
| [`05_load_balancing/`](05_load_balancing/) | §6 Prefix Caching, §7 Load Balancing | 8 | none |
| [`06_users_models/`](06_users_models/) | §4 Users and Models | 10 | 5 (not the heatmap of per-user medians or the four rasters) |

```bash
python paper/02_token_shape_latency/reproduce.py   # one directory
make paper                                         # all six
make paper-check                                   # compare figures/ with config/figures.json
```

Every script takes `--db PATH` (the DuckDB trace view; default
`$CHUTES_DB_PATH`, else `output/trace_view.duckdb`) and `--recompute` (rerun
every query instead of reusing `output/paper/cache/`).

**Cache.** The small query results ship in `data/paper/cache/` and are copied
into `output/paper/cache/` on the first run. A script opens the trace only when
a result it needs is missing, or with `--recompute`, so the figures marked
above redraw without the 91 GB download. The rest need the trace once; later
runs replot from `output/paper/cache/`.

**Time axes.** Trace timestamps count from the first request
(`1970-01-01 00:00:00` is the start), so time axes show trace days (day 0
first) or trace months (1 to 13).

The paper's other three figures come from the simulation studies:
the two §6 token-hit-ratio figures from `pipeline/caching/plot_hit_ratio.py`,
and the §7 routing figure from `pipeline/routing/plot_tradeoff.py`
(`make figures` redraws all three from committed results).
