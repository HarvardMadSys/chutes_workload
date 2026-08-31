# Paper figure reproduction

One directory per reproduction unit, each regenerating a group of the paper's
figures from the anonymized one-year trace. This is the third leg of the
artifact: `pipeline/routing/` and `pipeline/caching/` reproduce the two
*simulation studies*, and `paper/` reproduces the *trace-analysis figures*
around them.

**The paper decides what ships.** `config/figures.json` is the draft's
`\includegraphics` list, and `organize_figures.py` prunes `figures/` to exactly
it. Several scripts draw extra variants — a second model, a panel the draft
dropped, standalone legends — and those are removed rather than shipped.

| Section dir | Paper section(s) | Ships |
|---|---|---|
| [`01_workload_overview/`](01_workload_overview/) | §2 Background | 5 of 7 drawn |
| [`02_token_shape_latency/`](02_token_shape_latency/) | §3 Production Trace Analysis | 8 of 10 |
| [`03_workload_evolution/`](03_workload_evolution/) | §4 model mix · §5 token shape | 7 of 7 |
| [`04_prefix_caching/`](04_prefix_caching/) | §5 cache behaviour · §6 IAT | 7 of 7 |
| [`05_load_balancing/`](05_load_balancing/) | §6 demand · §7 latency, regimes | 8 of ~30 (Part A) |
| [`06_users_models/`](06_users_models/) | §4 Users and Models | 10 of 10 |

All six need the trace (see below); the shipped cache only shortens a rerun.

The paper's §6 eviction figures and §7 simulation figure come from the
pipelines, not from here — see "Relationship to the simulation studies".

```bash
python paper/01_workload_overview/reproduce.py     # one section
make paper                                         # every section, then prune
python paper/organize_figures.py --check           # audit figures/ against the list
```

Common flags: `--db PATH` (default `$CHUTES_DB_PATH`, else
`config/workload.json`), `--recompute` (rerun the queries instead of reusing
`output/paper/cache/`).

## Auditing the figure set

```bash
python paper/organize_figures.py --check      # missing / extra, changes nothing
python paper/organize_figures.py --dry-run    # what pruning would remove
```

`--check` exits non-zero if a figure the paper includes is missing, or if
`figures/` holds one the paper does not.

## The trace is required

Every section opens the trace at import for a row-count check, so all six need
it. `data/paper/cache/` ships the small query results, which shortens a rerun
but does not replace the trace. Pass `--recompute`, or set
`$CHUTES_SEED_CACHE=0`, to compute everything from the trace instead.

## Figures drawn elsewhere

| Paper figure | Built by |
|---|---|
| the two §6 token-hit-ratio figures | `pipeline/caching/plot_paper_figures.py` |
| the §7 routing tradeoff figure | `pipeline/routing/plot_tradeoff.py` |

These scripts are the paper's own figure scripts with only the paths changed;
path resolution is centralized in [`_paths.py`](_paths.py).
