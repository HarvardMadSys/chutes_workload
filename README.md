<h1 align="center">A Year in LLM Serving</h1>

<p align="center">
  <strong>Workload Evolution, Caching, and Load Balancing</strong>
  <br>
  <sub>One year · 6.12B requests · 9,174 models</sub>
</p>

<p align="center">
  <a href="#dataset">Dataset</a>
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
| `chute_id` | Model identifier |
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

## Installation

```bash
git clone https://github.com/williamnixon20/VLDB_Chutes_Workload.git
cd VLDB_Chutes_Workload

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Data

The trace is a single Parquet file, `chutes_trace.parquet` (91 GB,
6,122,413,756 rows). It is the only download — everything else it needs,
including the user cohort table, ships in this repository.

```bash
scripts/fetch_data.sh          # downloads the trace, resumes if interrupted
make trace                     # expose it as the DuckDB view all_metrics_user
```

It is published at

```text
https://harvardsys-datasets.s3.us-east-1.amazonaws.com/2026_chutes_anonymized/chutes_trace.parquet
```

If the file is already on the machine, point at it instead — `--link`
symlinks rather than copying:

```bash
scripts/fetch_data.sh --from /path/to/chutes_trace.parquet
```

Either way the fetch ends in a verification pass; `scripts/verify_data.py`
repeats it at any time. The caching and routing studies run on recovered
multi-turn sessions, rebuilt from the trace in about 3 minutes:

```bash
make dataset
```

## Reproducing the Paper

The paper includes 48 figures: 45 from the workload analysis, 2 from the
caching study, 1 from the routing study. Three of them redraw from committed
numbers with no data at all, which is the quickest check that the environment
works:

```bash
make figures            # the 3 simulation figures, seconds, no download
```

The rest need the trace, and the two simulation studies need `make dataset`
first:

```bash
make paper              # the 45 workload-analysis figures
make routing            # routing sweep, ~30 min
make caching            # eviction sweep, ~1 h including the libCacheSim build
```

Two checks:

```bash
make smoke              # replay one routing cell against the published numbers
make paper-check        # audit figures/ against the paper's figure list
```

## Repository Structure

```text
paper/       Workload analysis and paper figure scripts
pipeline/    Caching and routing experiments
src/         Simulator implementation
config/      Dataset, experiment, and figure-list specifications
data/        Data and results used by the figures
figures/     Reproduced paper figures
scripts/     Dataset and reproduction utilities
output/      Everything generated, including the downloaded trace (gitignored)
```

## License

The dataset and this artifact are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You may share and adapt the material for any purpose, including commercially,
provided you give appropriate credit — please cite the paper below.

## Citation

If you use the dataset or artifact in your research, please cite our paper:

```bibtex
@article{nixon2027year,
  title   = {A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing},
  author  = {William Nixon and Jon Durbin and Florian Standhartinger
             and Haryadi S. Gunawi and Juncheng Yang},
  journal = {Proceedings of the VLDB Endowment},
  year    = {2027}
}
```
