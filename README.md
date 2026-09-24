<h1 align="center">A Year in LLM Serving</h1>

<p align="center">
  <strong>Workload Evolution, Caching, and Load Balancing</strong>
  <br>
  <sub>One year · 6.12B requests · 9,174 models</sub>
</p>

<p align="center">
  <a href="#the-trace">Trace</a>
  ·
  <a href="#sessions">Sessions</a>
  ·
  <a href="#reproducing-the-paper">Reproduction</a>
  ·
  <a href="#citation">Citation</a>
</p>

---

This repository contains the dataset and reproducibility artifacts for
**A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing**.

We analyze one year of production traffic from
[Chutes](https://chutes.ai/), covering **6.12 billion requests** across
**9,174 models**. The trace captures request-level timing, model and
anonymized user identifiers, serving instances, token counts, latency,
time-to-first-token (TTFT), and prefix-cache reuse.

The repository includes:

- the released production trace;
- scripts for reproducing the paper's workload analysis;
- prefix-cache eviction simulations; and
- routing and load-balancing simulations.

## Dataset

The released trace contains one row per API invocation.

| Field | Description |
|---|---|
| `invocation_id` | Request identifier |
| `function_name` | API endpoint |
| `chute_id` | Model identifier (maps to a model name in data/paper/chute_models.csv) |
| `user_id` | Anonymized user identifier |
| `rehash_round` | Which 3-month rotation round the row belongs to |
| `instance_id` | Serving instance |
| `started_at` | Start time, elapsed from the trace's first request |
| `completed_at` | Completion time, same origin |
| `it` | Input tokens |
| `ot` | Output tokens |
| `ct` | Cached tokens |
| `ttft` | Time to first token |

User identifiers are re-anonymized every three months, so
`(user_id, rehash_round)` is the user key. Timestamps are normalized. 

The dataset is published at
`https://harvardsys-datasets.s3.us-east-1.amazonaws.com/2026_chutes_anonymized/chutes_trace.parquet`

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make fetch    # download chutes_trace.parquet (91 GB, 6,122,413,756 rows) and verify it
make trace    # expose it to DuckDB as the view all_metrics_user
```


## Sessions

We reconstruct sessions by chaining requests from the same
`(user, model)` pair. A request is considered to continue an earlier request when
its prompt contains the earlier request's context as a prefix. For each request, we search up to the previous 10 requests from the same
`(user, model)` pair for a matching parent. 

Running `make sessions` reconstructs
sessions for the two models used in our caching and routing studies:

| Model | Key | Rehash rounds | Requests | Sessions |
|---|---|---:|---:|---:|
| DeepSeek-V3.2 | `deepseek_v32` | 302–309 | 7,197,399 | 2,473,709 |
| MiniMax-M2.5 | `minimax_m25` | 346–353 | 4,223,356 | 1,079,756 |

The output, `output/sessions/<key>.parquet`, contains the corresponding trace rows
plus four session-reconstruction columns:

| Column | Meaning |
|---|---|
| `session_id` | `<user_id>#<n>`, a unique identifier for each reconstructed session |
| `turn_id` | 0 for the first request in a session, then 1, 2, … |
| `parent_invocation_id` | The request that this request continues (`null` for turn 0) |
| `prefix_tokens` | The parent's `it + ot`, representing the prefix already present in the current prompt (0 for turn 0) |

These reconstructed sessions are used as the workloads for the caching and routing
simulations.


## Reproducing the paper

```bash
make figures      # the 3 simulation figures, from data/caching/ and data/routing/
make paper        # the 45 trace-analysis figures (the first run queries the trace)
make sessions     # reconstruct the sessions the two studies replay
make caching      # eviction sweep, ~1 h including the libCacheSim build
make routing      # routing sweep, ~30 min
```

## Repository structure

```text
config/     experiment configurations
data/       small experiment committed results
figures/    paper's figures
paper/      paper's plotting script
pipeline/   Session reconstruction, caching, and routing 
scripts/    Misc utilities
src/        Reconstruction and simulator code
```

## License

The dataset and this artifact are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You may share and adapt the material for any purpose, including commercially,
provided you give appropriate credit — please cite the paper below.

## Citation

If you use the dataset or artifact in your research, please cite our paper:

```bibtex
@article{nixon2026year,
  title={A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing},
  author={Nixon, William and Durbin, Jon and Standhartinger, Florian and Gunawi, Haryadi S and Yang, Juncheng},
  journal={arXiv preprint arXiv:2608.13573},
  year={2026}
}
```
