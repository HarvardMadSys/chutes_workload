# §2 Background: workload overview

Paper float **`fig:workload_temporal_structure`** ("Daily request volume, token
volume, active models, and API mix evolve differently over the year").

```bash
python paper/01_workload_overview/reproduce.py [--db PATH] [--recompute]
```

| Panel | Figure (`figures/paper/01_workload_overview/`) | LaTeX label | What it shows | Query cache (`output/paper/cache/`) |
|---|---|---|---|---|
| (a) | `fig_request_rate_day.pdf` | `fig:request_rate_over_time` | Requests per day | `daily_requests.parquet` |
| (b) | `fig_token_rate_day.pdf` | `fig:todo_active_users_over_time` | Input + output tokens (`it + ot`) per day, over requests with a `completed_at` | `daily_tokens.parquet` |
| (c) | `fig_daily_unique_models.pdf` | `fig:active_models_over_time` | Distinct models (`chute_id`) with at least one request that day | `daily_unique_users_models.parquet` |
| (d) | `fig_function_name_daily_requests.pdf` | `fig:api_call_over_time` | Requests per day by `function_name`: chat, chat_stream, completion, completion_stream, and Other for every other function | `function_name_daily_requests.parquet` |
| (e) | `fig_seasonality.pdf` | `fig:hour_of_week_heatmap` | Average requests in each hour of the week (weekday × hour of day), in millions | `hour_of_week_requests.parquet` |

Panels (a) to (d) plot against the trace day: whole days since the trace's
first request, which is day 0. Panel (e) divides each weekday's total by the
number of days that fall on that weekday, counted from panel (a)'s result.

**Data.** Each panel is one aggregate query over the trace view
(`all_metrics_user`), cached in `output/paper/cache/` under the name in the
table. A rerun reuses the caches and only replots; `--recompute` reruns all
five queries. The trace is opened only when a query runs, so once the five
caches exist the figures redraw without it.

**What ships.** All five query results ship in `data/paper/cache/` (about
40 KB; the first run copies them into `output/paper/cache/`), so all five
figures redraw without the trace.
