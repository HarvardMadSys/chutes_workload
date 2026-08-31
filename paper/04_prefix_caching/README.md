# §5–§6 Prefix caching

Paper floats: **`fig:cache_behavior_overview`** and **`fig:iat_overview`**
(in §5 Workload Evolution), **`fig:pair_iat_cdf`** (§6 Prefix Caching).

```bash
python paper/04_prefix_caching/reproduce.py [--db PATH] [--recompute]
```

| # | Output (`figures/paper/04_prefix_caching/`) | Paper label | What it shows | Needs |
|---|---|---|---|---|
| 1 | `fig_request_cached_fraction_cdf_by_input.pdf` | `fig:request_cache_fraction_cdf` | Token hit ratio CDF: Global vs <1K / >8K input cohorts | cached ✓ |
| 2 | `fig_request_token_hit_rate_cdf_modelhl.pdf` | `fig:model_cache_fraction_cdf` | Token hit ratio CDF, Global + 3 highlighted models | cached ✓ (global part needs DB) |
| 3 | `fig_model_nreq_vs_avg_hitratio_hexbin.pdf` | `fig:model_cache_profiles` | Hexbin: per-model request count vs avg hit ratio | cached ✓ |
| 4 | `fig_user_nreq_vs_avg_hitratio_hexbin.pdf` | `fig:user_cache_profiles` | Same per user | cached ✓ |
| 5 | `fig_iat_monthly_per_user_band.pdf` | `fig:iat_per_user_monthly_trend` | Per-user inter-arrival time by month, median + bands | cached ✓ |
| 6 | `fig_per_user_iat_p50_by_model.pdf` | (paper uses `useriatmodel.png`) | Boxplot of per-(user, model) IAT p50 per highlighted model | cached ✓ |
| 7 | `fig_iat_pair_ttl_coverage.pdf` | `fig:pair_iat_cdf` | Request-weighted TTL coverage of (user, model) reuse gaps | **DB** |

**Data**: the DuckDB view over the trace, plus `data/paper/chute_models.csv`.

Two things worth knowing:

- **Figure 7 hard-fails without the trace.** Its cache
  (`_cache_iat_per_user_model_pair_quantiles_fresh.parquet`) is too large to
  ship, so it raises rather than plotting something wrong.
- **§6's two eviction figures are not drawn here.** They come from
  libCacheSim rather than from the trace, so `pipeline/caching/` runs that
  sweep and draws them (`make figures-caching`, or `make caching` to rerun the
  sweep first).
