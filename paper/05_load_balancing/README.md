# §6–§7 Load balancing (production trace)

Paper floats: **`fig:case_lb_demand_replication`** (in §6 Prefix Caching),
**`fig:case_lb_latency`** and **`fig:case_cache_regimes`** (§7 Load Balancing).

```bash
python paper/05_load_balancing/reproduce.py --part a [--db PATH]
```

Eight figures ship, all from **Part A** — the production trace analysis. Window:
the last two months of the trace (ending at day 364); models
`deepseek-ai/DeepSeek-V3.2-TEE` and `deepseek-ai/DeepSeek-V3-0324-TEE`.

| Output (`figures/paper/05_load_balancing/`) | Paper label | What it shows |
|---|---|---|
| `fig_lb_request_rate.pdf` | `fig:case_lb_reqrate` | Hourly request totals |
| `fig_lb_active_instances.pdf` | `fig:case_lb_active` | Median active instances per hour |
| `fig_lb_maxmean_tokens.pdf` | `fig:case_lb_maxmean` | Per-minute max/mean per-instance token load, 1-h smoothed |
| `fig_lb_concurrency.pdf` | `fig:case_lb_concurrency` | Per-instance request heatmap over the densest 24-h window |
| `f7b_ttft_p90_deepseek-ai__DeepSeek-V3-0324-TEE.pdf` | `fig:case_lb_ttft` | Instance load × input tokens hexbin, coloured by P90 TTFT |
| `f7d_decode_p90_deepseek-ai__DeepSeek-V3.2-TEE.pdf` | `fig:case_lb_duration` | Instance load × output tokens, coloured by P90 decode time |
| `f1f_user_zoom_high_top_deepseek-ai__DeepSeek-V3.2-TEE.pdf` | `fig:case_cache_high_load` | One user's busiest hour: per-instance request share vs token hit ratio |
| `f1f_user_zoom_high_min_deepseek-ai__DeepSeek-V3.2-TEE.pdf` | `fig:case_cache_low_load` | The same, for the low-load pick |

Part A draws considerably more than this — the same four panels for the
second model under their raw `f3a_/f4_/f5a_/f1b_` names, the `f7b_/f7d_`
hexbins for the model the paper does not use, and the med/low user tiers.
`organize_figures.py` removes them; only the eight above are in the paper.

**Data**: the DuckDB trace and `data/paper/chute_models.csv`. Part A builds
four cache layers under `output/paper/cache/` — `cim/` (77 MB), `mim/`
(22 MB), `f7_req_bucket/` (1.1 GB) and `f1d/`. None are small enough for git,
so Part A needs the trace or the cache archive; the first scan takes minutes
per model.

## Part B

`--part b` replots the routing sweep from `data/routing/figure20_sweep.csv` and
needs no trace. Its output does not ship; the paper's §7 simulation figure comes
from `make figures-routing`.
