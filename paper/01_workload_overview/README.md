# §2 Background — workload overview

Paper float: **`fig:workload_temporal_structure`** ("Daily request volume,
token volume, active models, and API mix evolve differently over the year").

```bash
python paper/01_workload_overview/reproduce.py [--db PATH] [--recompute]
```

| # | Output (`figures/paper/01_workload_overview/`) | Paper panel | What it shows |
|---|---|---|---|
| 1 | `fig_request_rate_day.pdf` | (a) `fig:request_rate_over_time` | Daily request count over the one-year trace |
| 2 | `fig_token_rate_day.pdf` | (b) `fig:todo_active_users_over_time` | Daily total tokens (`it + ot`) |
| 3 | `fig_daily_unique_models.pdf` | (c) `fig:active_models_over_time` | Distinct models active per day |
| 4 | `fig_function_name_daily_requests.pdf` | (d) `fig:api_call_over_time` | Daily requests split by API `function_name` |
| 5 | `fig_seasonality.pdf` | (e) `fig:hour_of_week_heatmap` | Hour-of-day × day-of-week request heatmap |

The script also draws `fig_daily_unique_users.pdf` and a standalone legend for
panel (d); the paper includes neither, so `organize_figures.py` removes them.

**Data**: the DuckDB trace only — no other inputs. All five query results
ship in `data/paper/cache/` (41 KB), so a rerun replots without re-querying;
`--recompute` forces the queries. The trace is still required — the script
opens it at import for a row-count check.

Note: panel (e) reuses the in-memory result of panel (a) for its
day-of-week normalization, so the first block runs even in cached mode.
