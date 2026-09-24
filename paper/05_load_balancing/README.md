# §6–§7 Load balancing (production trace)

Paper floats: **`fig:case_lb_demand_replication`** (in §6 Prefix Caching),
**`fig:case_lb_latency`** and **`fig:case_cache_regimes`** (§7 Load Balancing).

```bash
python paper/05_load_balancing/reproduce.py [--db PATH] [--recompute]
```

Window: the last two months of the trace, days 306 to 364: the queries keep
`started_at >= '1970-11-02 19:59:59.984705'` and `< '1970-12-31 19:59:59.984705'`
(the same time of day as the model windows in `config/workload.json`). Time
axes count days from the start of the trace (day 0).

| Output (`figures/paper/05_load_balancing/`) | Paper label | Model | What it shows |
|---|---|---|---|
| `fig_lb_request_rate.pdf` | `fig:case_lb_reqrate` | DeepSeek-V3-0324-TEE | Requests per hour |
| `fig_lb_active_instances.pdf` | `fig:case_lb_active` | DeepSeek-V3-0324-TEE | Active instances per minute, median of each hour |
| `fig_lb_maxmean_tokens.pdf` | `fig:case_lb_maxmean` | DeepSeek-V3-0324-TEE | Per-minute max/mean token load across instances, mean of each hour |
| `fig_lb_concurrency.pdf` | `fig:case_lb_concurrency` | DeepSeek-V3-0324-TEE | Requests per instance and minute over the 24 hours with the most active instances |
| `fig_lb_ttft_p90_vs_load.pdf` | `fig:case_lb_ttft` | DeepSeek-V3.2-TEE | Instance load × input tokens hexbin, coloured by P90 TTFT |
| `fig_lb_decode_p90_vs_load.pdf` | `fig:case_lb_duration` | DeepSeek-V3.2-TEE | Instance load × output tokens hexbin, coloured by P90 decode time |
| `fig_lb_user_zoom_top.pdf` | `fig:case_cache_high_load` | DeepSeek-V3.2-TEE | The busiest hour of the busiest user whose token hit ratio that hour is at least 0.4: per instance, the user's share of requests and its token hit ratio |
| `fig_lb_user_zoom_min.pdf` | `fig:case_cache_low_load` | DeepSeek-V3.2-TEE | The same for the least busy such user (still at least 200 requests in that hour) |

**Data**: the DuckDB trace and `data/paper/chute_models.csv`. The query results
are cached in `output/paper/cache/`:

| Cache file | One row per | Used by |
|---|---|---|
| `instance_minute_stats_deepseek_v3_0324.parquet` (40 MB) | instance and minute | the four DeepSeek-V3-0324 figures |
| `instance_minute_stats_deepseek_v32.parquet` (47 MB) | instance and minute | the user zooms (the instances active in the hour) |
| `request_latency_vs_load_deepseek_v32.parquet` (455 MB) | request | the two hexbins |
| `user_busiest_hour_deepseek_v32.parquet` (44 KB) | user | the user zooms (which user and hour) |

None of them ship with the artifact, so the first run needs the trace and takes
a few minutes. Each user zoom also runs one small trace query on every run.
