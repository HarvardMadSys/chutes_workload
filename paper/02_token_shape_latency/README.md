# §3 Production Trace Analysis — token shape and latency

Paper floats: **`fig:model_characteristics`** (token shape and latency of three
highlighted models against the whole workload) and
**`fig:todo_ttft_diagnostics_pair`** (median latency binned by request shape).

```bash
python paper/02_token_shape_latency/reproduce.py [--db PATH] [--recompute]
```

| # | Output (`figures/paper/02_token_shape_latency/`) | Paper label | What it shows |
|---|---|---|---|
| 1 | `fig_token_length_input_cdf_by_model.pdf` | `fig:highlight_input_cdf` | CDF of input tokens per request |
| 2 | `fig_token_length_output_cdf_by_model.pdf` | `fig:highlight_output_cdf` | CDF of output tokens per request |
| 3 | `fig_output_input_ratio_cdf_by_model.pdf` | `fig:highlight_ratio_cdf` | CDF of `ot / it` |
| 4 | `fig_latency_e2e_cdf_by_model.pdf` | `fig:latency_cdf_global` | CDF of end-to-end latency |
| 5 | `fig_prefill_fraction_cdf_by_model.pdf` | `fig:todo_ttft_cache_bucket` | CDF of `ttft / duration` (prefill fraction) |
| 6 | `fig_duration_p50_heatmap.pdf` | `fig:todo_ttft_sequence_length` | P50 duration over input × output token buckets |
| 7 | `fig_ttft_p50_heatmap_input_output.pdf` | `fig:todo_ttft_request_rate` | P50 TTFT over the same grid |
| 8 | `fig_ttft_heatmap_input_cached.pdf` | `fig:todo_ttft_cache_effect` | Median TTFT over input tokens × token hit ratio |

Figures 1–5 show all requests ("Global") and the three models in
`config/highlight_models.json` (figure label → `chute_id`).

**Data**: every query result ships in `data/paper/cache/`, so a replot does not
need the trace. The script opens the DuckDB trace only when a cache file is
missing or `--recompute` is passed. After editing
`config/highlight_models.json`, rerun with `--recompute`.
