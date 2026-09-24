# §5–§6 Prefix caching

Paper floats: **`fig:cache_behavior_overview`** and **`fig:iat_overview`**
(in §5 One-Year Workload Evolution), **`fig:pair_iat_cdf`** (§6 Prefix Caching).

```bash
python paper/04_prefix_caching/reproduce.py [--db PATH] [--recompute]
```

| # | Output (`figures/paper/04_prefix_caching/`) | Paper label | What it shows | Needs |
|---|---|---|---|---|
| 1 | `fig_request_cached_fraction_cdf_by_input.pdf` | `fig:request_cache_fraction_cdf` | Token hit ratio CDF: Global vs <1K / >8K input tokens | cached |
| 2 | `fig_request_token_hit_rate_cdf_by_model.pdf` | `fig:model_cache_fraction_cdf` | Token hit ratio CDF: Global + 3 highlighted models | cached |
| 3 | `fig_model_nreq_vs_avg_hitratio_hexbin.pdf` | `fig:model_cache_profiles` | Hexbin: per-model request count vs average token hit ratio | cached |
| 4 | `fig_user_nreq_vs_avg_hitratio_hexbin.pdf` | `fig:user_cache_profiles` | The same per user | cached |
| 5 | `fig_iat_monthly_per_user_band.pdf` | `fig:iat_per_user_monthly_trend` | Per-user inter-arrival time (IAT) by trace month: the median over users of each user's P50, with bands between the medians of each user's P10/P90 and P25/P75 | cached |
| 6 | `fig_per_user_iat_p50_by_model.pdf` | `fig:iat_per_user_by_model` | Boxplot over users of each user's median IAT to each highlighted model | cached |
| 7 | `fig_iat_pair_ttl_coverage.pdf` | `fig:pair_iat_cdf` | Request-weighted TTL coverage of (user, model) reuse gaps | **trace** |

"cached" means the query results ship in `data/paper/cache/`: the script opens
the DuckDB trace only when a cache file is missing or `--recompute` is passed.
The highlighted models are listed in `config/highlight_models.json` (shared
with `02_token_shape_latency/`); after editing it, rerun with `--recompute`.

Two things worth knowing:

- **Figure 7 needs the trace on the first run.** Its query result
  (`iat_per_user_model_pair.parquet`, about 700 MB) does not ship. The first
  run writes it to `output/paper/cache/`, and later runs replot from there.
  Figure 5's result (`iat_monthly_per_user.parquet`) ships; recomputing it
  writes one file per batch of users under `iat_monthly_per_user_batches/`, so
  an interrupted run resumes where it stopped.
- **§6's two eviction figures are not drawn here.** They come from
  libCacheSim rather than from the trace, so `pipeline/caching/` runs that
  sweep and draws them (`make figures` redraws them from the committed results,
  `make caching` reruns the sweep first).
