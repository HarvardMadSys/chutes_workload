# §3 Production Trace Analysis — token shape and latency

Paper floats: **`fig:highlight_token_shape`** (per-request token-length
distributions for three representative models), **`fig:cache_latency_pair`**
(latency behavior), **`fig:todo_ttft_diagnostics_pair`** (median latency
binned by request shape).

```bash
python paper/02_token_shape_latency/reproduce.py [--db PATH] [--recompute]
```

| # | Output (`figures/paper/02_token_shape_latency/`) | Paper label | What it shows |
|---|---|---|---|
| 1 | `fig_token_length_input_cdf_modelhl.pdf` | `fig:highlight_input_cdf` | Input-token CDF per request |
| 2 | `fig_token_length_output_cdf_modelhl.pdf` | `fig:highlight_output_cdf` | Output-token CDF per request |
| 3 | `fig_output_input_ratio_cdf_modelhl.pdf` | `fig:highlight_ratio_cdf` | CDF of `ot / it` |
| 4 | `fig_latency_e2e_cdf_modelhl.pdf` | `fig:latency_cdf_global` | End-to-end latency CDF |
| 5 | `fig_prefill_fraction_cdf_modelhl.pdf` | `fig:todo_ttft_cache_bucket` | CDF of `ttft / duration` (prefill fraction) |
| 6 | `fig17c_duration_p50_heatmap.pdf` | `fig:todo_ttft_sequence_length` | P50 duration over input × output token buckets |
| 7 | `fig_ttft_p50_heatmap_input_output.pdf` | `fig:todo_ttft_request_rate` | P50 TTFT over the same grid |
| 8 | `fig16a_ttft_heatmap_input_cache.pdf` | `fig:todo_ttft_cache_effect` | Median TTFT over input tokens × token hit ratio (the paper embeds this as `chat_input_hit.png`) |

The script also draws two standalone model legends; the paper includes
neither, so `organize_figures.py` removes them.

**Data**: the DuckDB trace, plus `data/paper/chute_models.csv` (optional here —
the three highlighted models carry explicit `chute_id`s, so the CSV only
validates them). The trace is required — the script opens it at import for a
row-count check — but cached query results shorten a rerun; `--recompute`
forces the queries.

The highlighted-model set is hashed into the cache filename
(`models_ca5aaf063c`), so changing the highlight list produces a new cache
rather than silently reusing the old one.
