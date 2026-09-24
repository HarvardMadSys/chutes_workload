# §4 Users and Models

Paper floats: **`fig:volume_breadth`**, **`fig:entity_token_shape`**,
**`fig:user_access_patterns`**, **`fig:model_access_patterns`**,
**`fig:model_burstiness`**.

```bash
python paper/06_users_models/reproduce.py [--db PATH] [--recompute]
```

The script draws these ten figures into `figures/paper/06_users_models/`. All
ten are in the paper.

| # | Figure | Paper label | What it shows |
|---|---|---|---|
| 1 | `fig_model_requests_vs_users_density.pdf` | `fig:volume_breadth_models` | Per-model hexbin: requests vs. distinct users |
| 2 | `fig_user_requests_distinct_models_heatmap.pdf` | `fig:volume_breadth_users` | Per-user hexbin: requests vs. distinct models |
| 3 | `fig_user_median_input_output_heatmap.pdf` | `fig:token_shape_users` | Per-user median input vs. median output tokens |
| 4 | `fig_model_median_input_output_heatmap.pdf` | `fig:token_shape_models` | Per-model median input vs. median output tokens |
| 5 | `fig_user_raster_periodic.pdf` | `fig:user_access_persistent` | 31-day model-access raster of the rank-190 user |
| 6 | `fig_user_raster_explore.pdf` | `fig:user_access_exploration` | 31-day model-access raster of the rank-134 user |
| 7 | `fig_model_raster_qwen3_next_80b.pdf` | `fig:model_access_power_user` | 31-day user-access raster of Qwen3-Next-80B-A3B-Instruct |
| 8 | `fig_model_raster_gpt_oss_20b.pdf` | `fig:model_access_correlated` | 31-day user-access raster of gpt-oss-20b |
| 9 | `fig_model_heatmap_fixedxy_density.pdf` | `fig:model_burstiness_density` | Per-model hexbin: requests per hour vs. mean hourly CV of inter-arrival times |
| 10 | `fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf` | `fig:model_burstiness_autocorr` | Same axes, one dot per model, colored by the Spearman lag-1 IAT autocorrelation |

**What needs the trace.** The query results behind figures 1, 2, 4, 9 and 10
are committed in `data/paper/cache/`, so these figures replot without the
trace. Figures 3 and 5–8 need the trace on the first run, because their results
are not committed: the per-user medians take 9.6 MB and the four raster tables
29 MB. Every run after that reuses `output/paper/cache/`, and `--recompute`
reruns every query.

**Rasters.** A raster shows the 31 days after the first request in its window.
The window is trace days 166–203 for the user rasters and days 166–227 for the
model rasters, with day 0 the start of the trace. User ids are anonymized per
3-month round, so each user is picked by their request rank in the window.
Models are picked by name through `data/paper/chute_models.csv`, which the
script requires. A model raster shows at most 800 of the model's users; a
seeded shuffle picks which ones.

**Batching.** The heavy queries run in batches to bound DuckDB memory:
- per-model inter-arrival times: the 5 busiest models one at a time, then in pairs up to rank 20, then 100 at a time;
- per-user medians: the 10 busiest users one at a time, then 7,500 at a time;
- per-model medians: the 10 busiest models one at a time, then all the others in one query;
- hourly burstiness: one trace month at a time;
- lag-1 autocorrelation: batches of models that grow from 1 to 1,000.

The last two write each batch to its own parquet, so an interrupted run resumes
where it stopped.
