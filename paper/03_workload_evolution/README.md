# §4–§5 Workload evolution

Paper floats: **`fig:model_mix_over_time`** (in §4 Users and Models),
**`fig:todo_token_percentiles_over_time`** and
**`fig:app_user_release_cohort_shape`** (both in §5 Workload Evolution).
All seven figures this section draws are in the paper.

```bash
python paper/03_workload_evolution/reproduce.py [--db PATH] [--recompute]
```

| # | Output (`figures/paper/03_workload_evolution/`) | Paper label (§) | What it shows |
|---|---|---|---|
| 1 | `fig_top10_model_monthly_request_share.pdf` | `fig:top10_model_monthly_request_share` (§4) | Stacked monthly share of requests, top-10 TEE-merged models + "Others" |
| 2 | `fig_top10_model_monthly_token_share.pdf` | `fig:top10_model_monthly_token_share` (§4) | Same, share of total tokens |
| 3 | `fig_top10_model_monthly_share_legend.pdf` | shared legend (§4) | Standalone 11-entry model legend |
| 4 | `fig_monthly_input_token_intervals.pdf` | `fig:monthly_input_token_intervals` (§5) | Monthly input-token P50 with P25–P75 and P10–P90 bands |
| 5 | `fig_monthly_output_token_intervals.pdf` | `fig:monthly_output_token_intervals` (§5) | Same for output tokens |
| 6 | `fig_user_release_cohort_input_band.pdf` | `fig:app_user_release_cohort_input` (§5) | Per-user median input tokens by first-active-month cohort |
| 7 | `fig_user_release_cohort_output_band.pdf` | `fig:app_user_release_cohort_output` (§5) | Same for output tokens |

**Data**: the DuckDB trace and `data/paper/chute_models.csv` (**required** —
it supplies `chute_id → model name` before the TEE-merge).

Figures 1–5 replot from the committed cache once it is present. Figures 6–7
always requery on a first run: their two caches are 10 MB and 13 MB, too large
for git. Those figures batch their
`GROUP BY user` (top 10 users individually, then chunks of 20,000) because a
single grouping over the full trace is too heavy — budget several minutes.
