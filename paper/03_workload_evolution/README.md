# §4–§5 Workload evolution

How the model mix and the request shape change over the year. Paper floats:
**`fig:model_mix_over_time`** (§4 Users and Models),
**`fig:token_shape_evolution`** and **`fig:app_user_release_cohort_shape`**
(both §5 One-Year Workload Evolution). All seven figures this section draws
are in the paper.

```bash
python paper/03_workload_evolution/reproduce.py [--db PATH] [--recompute]
```

| # | Output (`figures/paper/03_workload_evolution/`) | Paper label (§) | What it shows |
|---|---|---|---|
| 1 | `fig_top10_model_monthly_request_share.pdf` | `fig:top10_model_monthly_request_share` (§4) | Stacked monthly share of requests: the 10 most-requested models (X and X-TEE merged) and "Others" |
| 2 | `fig_top10_model_monthly_token_share.pdf` | `fig:top10_model_monthly_token_share` (§4) | The same models' share of input + output tokens |
| 3 | `fig_top10_model_monthly_share_legend.pdf` | legend of `fig:model_mix_over_time` (§4) | The 11-entry legend of figures 1 and 2 |
| 4 | `fig_monthly_input_token_intervals.pdf` | `fig:monthly_input_token_intervals` (§5) | Input tokens per request, by month: median, P25–P75 and P10–P90 |
| 5 | `fig_monthly_output_token_intervals.pdf` | `fig:monthly_output_token_intervals` (§5) | The same for output tokens |
| 6 | `fig_user_release_cohort_input_band.pdf` | `fig:app_user_release_cohort_input` (§5) | Each user's median input tokens by release cohort: median, P25–P75 and P10–P90 over the users |
| 7 | `fig_user_release_cohort_output_band.pdf` | `fig:app_user_release_cohort_output` (§5) | The same for output tokens |

The x axes count months of the trace, 1 to 13, from its first request; month
13 holds only the trace's last two days. A user's release cohort is the month
in which the person first appeared: the trace's `release_cohort` column (see
`scripts/build_trace_view.py`), because a `user_id` rotates every 3 months and
cannot tell it.

**Data**: the DuckDB trace, and `data/paper/chute_models.csv`, which maps
`chute_id` to the model name that figures 1–3 group and label by.

**Cache**: each query's result is saved in `output/paper/cache/`. The two
monthly tables ship in `data/paper/cache/`, so figures 1–5 redraw without the
trace. The cohort figures 6–7 need the trace on the first run, because their
per-user table (10 MB) does not ship; after that the script only replots.
`--recompute` reruns every query.

| Cache file | Contents | Figures |
|---|---|---|
| `monthly_requests_tokens_by_model.parquet` | requests and tokens per month and chute | 1–3 |
| `monthly_token_percentiles.parquet` | P10, P25, P50, P75 and P90 of input and output tokens per month | 4–5 |
| `user_cohort_user_rank.parquet` | requests per user, which orders the batches of the next query | 6–7 |
| `user_cohort_token_medians.parquet` | per user: release cohort, requests, median input and output tokens | 6–7 |

The per-user query is the slow one. A single `GROUP BY user_id` over the full
trace is too heavy, so it runs on the 10 heaviest users one at a time, then on
batches of 20,000 users; budget several minutes.
