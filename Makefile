# chutes-load-simulator — reproducibility artifact
#
#   make trace      build the DuckDB view over the anonymized trace (do this first)
#   make figures    replot every simulation figure from committed data
#   make paper      reproduce the paper's trace-analysis figures, then prune
#   make paper-check     audit figures/ against config/figures.json
#   make smoke      quick end-to-end check (one routing cell + one cachesim run)
#   make routing    rerun the routing sweep, rebuild the CSV, replot
#   make caching    rebuild traces, build libCacheSim, rerun the sweep, replot
#   make dataset    rebuild sessions.parquet from the anonymized trace
#   make fetch      download + verify the trace (one parquet file)
#   make verify     check the environment and what data is present
#   make clean      remove generated figures (keeps output/ and fetched data)
#   make distclean  also remove output/ (datasets, sweeps, libCacheSim build)
#
# Env: PYTHON, NJOBS, MAX_PARALLEL, CHUTES_TRACE_DIR, CHUTES_DB_PATH,
#      CHUTES_DATA_ROOT, CHUTES_OUTPUT_ROOT

PYTHON      ?= python3
MAX_PARALLEL ?=
NJOBS       ?=

ROUTING     := pipeline/routing
CACHING     := pipeline/caching
PREP        := pipeline/preprocess
PAPER       := paper
PAPER_SECTIONS := 01_workload_overview 02_token_shape_latency 03_workload_evolution \
                  04_prefix_caching 05_load_balancing 06_users_models

.PHONY: all trace figures figures-routing figures-caching smoke routing caching \
        dataset fetch verify clean distclean help paper paper-organize \
        paper-check $(PAPER_SECTIONS)

all: figures

help:
	@sed -n '2,18p' Makefile

## ---------------------------------------------------------------- trace
# One-time: expose the anonymized trace as the table every analysis queries.
trace:
	$(PYTHON) scripts/build_trace_view.py

## ---------------------------------------------------------------- figures
figures: figures-routing figures-caching

figures-routing:
	$(PYTHON) $(ROUTING)/plot_tradeoff.py

figures-caching:
	$(PYTHON) $(CACHING)/plot_paper_figures.py data/caching/results.txt \
	    --out-dir figures/caching

## ---------------------------------------------------------------- paper
# Every section, then prune to what the paper includes. Sections query the
# trace on a first run; pass CHUTES_DB_PATH or --db if it is not at the
# configured location.
paper: $(PAPER_SECTIONS) paper-organize

$(PAPER_SECTIONS):
	$(PYTHON) $(PAPER)/$@/reproduce.py

# The scripts draw more than the paper uses; keep only what it includes.
paper-organize:
	$(PYTHON) $(PAPER)/organize_figures.py

# Audit the artifact's figures against the paper's frozen figure list.
paper-check:
	$(PYTHON) $(PAPER)/organize_figures.py --check

## ---------------------------------------------------------------- checks
verify:
	$(PYTHON) scripts/verify_env.py

smoke:
	$(PYTHON) scripts/smoke_test.py

## ---------------------------------------------------------------- rerun
fetch:
	bash scripts/fetch_data.sh

routing:
	$(PYTHON) $(ROUTING)/run_sweep.py $(if $(MAX_PARALLEL),--max-parallel $(MAX_PARALLEL),)
	$(PYTHON) $(ROUTING)/build_data.py
	$(MAKE) figures-routing

caching:
	NJOBS=$(NJOBS) PYTHON=$(PYTHON) bash $(CACHING)/run_cachesim.sh

## ---------------------------------------------------------------- dataset
dataset:
	$(PYTHON) $(PREP)/build_sessions.py

## ---------------------------------------------------------------- cleanup
clean:
	rm -rf figures/routing/*.pdf figures/routing/*.png \
	       figures/caching/*.pdf figures/caching/*.png \
	       figures/paper/*/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf output/
