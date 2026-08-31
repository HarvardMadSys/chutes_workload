# §4 Users and Models

Paper floats: **`fig:volume_breadth`**, **`fig:entity_token_shape`**,
**`fig:user_access_patterns`**, **`fig:model_access_patterns`**,
**`fig:model_burstiness`**.

```bash
python paper/06_users_models/reproduce.py [--db PATH] [--recompute]
```

All ten figures this section draws are in the paper.

| # | Output (`figures/paper/06_users_models/`) | Paper label | What it shows |
|---|---|---|---|
| 1 | `requests_users_model_density.pdf` | `fig:volume_breadth_models` | Per-model hexbin: requests × distinct users, log-log |
| 2 | `fig_user_requests_distinct_models_heatmap.pdf` | `fig:volume_breadth_users` | Per-user hexbin: requests × distinct models |
| 3 | `fig_user_median_input_output_heatmap.pdf` | `fig:token_shape_users` | Per-user median input vs median output tokens |
| 4 | `fig_model_median_input_output_heatmap.pdf` | `fig:token_shape_models` | Per-model median input vs median output tokens |
| 5 | `periodic.pdf` | `fig:user_access_persistent` | 30-day model-access raster for a periodic high-volume user |
| 6 | `user_raster_explore.pdf` | `fig:user_access_exploration` | 30-day model-access raster for a model-exploring user |
| 7 | `standalone_model_rank41_Qwen_Qwen3-Next-80B-A3B-Instruct_…_user_access_raster.pdf` | `fig:model_access_power_user` | 30-day user-access raster, Qwen3-Next-80B — periodic access |
| 8 | `openai_gpt-oss-20b_user_access_raster.pdf` | `fig:model_access_correlated` | 30-day user-access raster, gpt-oss-20b — correlated access |
| 9 | `fig_model_heatmap_fixedxy_density.pdf` | `fig:model_burstiness_density` | Model popularity vs burstiness, hexbin density |
| 10 | `fig_model_heatmap_fixedxy_scatter_colored_metrics.pdf` | `fig:model_burstiness_autocorr` | Same axes, coloured by Spearman lag-1 IAT autocorrelation |

**Data**: the DuckDB trace and `data/paper/chute_models.csv` — the CSV is
load-bearing here, not just validation: figures 7–8 select their models by
`(rank, model_name)`, so without it that selection matches nothing and the two
rasters are silently skipped.

Figures 1–8 depend on caches too large for git (per-user/per-model medians at
9–12 MB each, and 29 MB of per-entity raster tables), so a first run recomputes
them from the trace — see [`../README.md`](../README.md).

The heavy queries are batched to bound DuckDB memory: top-ranked models one at
a time, then in groups; per-user medians in 7,500-user batches; the
autocorrelation suite in growing batches that each write their own parquet, so
an interrupted run resumes where it stopped.
