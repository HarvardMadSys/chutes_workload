# A Year in LLM Serving — trace release and reproducibility artifact
#
#   make fetch        download and verify the trace (one 91 GB parquet)
#   make trace        build the DuckDB view all_metrics_user over it (do this first)
#   make sessions     reconstruct sessions for the two simulated models (~3 min)
#   make figures      redraw the 3 simulation figures from committed data (seconds)
#   make paper        redraw the 45 trace-analysis figures (needs the trace)
#   make paper-check  compare figures/ with the paper's figure list
#   make routing      rerun the routing sweep (~30 min), rebuild its table, redraw
#   make caching      build libCacheSim, rerun the eviction sweep (~1 h), redraw
#   make smoke        replay one routing cell and one cachesim run against the published numbers
#   make clean        remove PNG previews and __pycache__ (keeps the committed figures)
#   make distclean    also empty output/, except the downloaded trace
#
# Env: PYTHON, NJOBS, MAX_PARALLEL, CHUTES_TRACE_DIR, CHUTES_DB_PATH, CHUTES_OUTPUT_ROOT

PYTHON       ?= python3
MAX_PARALLEL ?=
NJOBS        ?=

PAPER_SECTIONS := 01_workload_overview 02_token_shape_latency 03_workload_evolution \
                  04_prefix_caching 05_load_balancing 06_users_models

.PHONY: all help fetch trace sessions figures paper paper-check \
        routing caching smoke clean distclean $(PAPER_SECTIONS)

all: figures

help:
	@sed -n '/^$$/q;p' Makefile

## ---------------------------------------------------------------- the trace
fetch:
	bash scripts/fetch_data.sh

trace:
	$(PYTHON) scripts/build_trace_view.py

## ---------------------------------------------------------------- sessions
sessions:
	$(PYTHON) pipeline/sessions/build_sessions.py

## ---------------------------------------------------------------- figures
figures:
	$(PYTHON) pipeline/routing/plot_tradeoff.py
	$(PYTHON) pipeline/caching/plot_hit_ratio.py data/caching/results.txt --out-dir figures/caching

paper: $(PAPER_SECTIONS)

$(PAPER_SECTIONS):
	$(PYTHON) paper/$@/reproduce.py

paper-check:
	$(PYTHON) paper/check_figures.py

## ---------------------------------------------------------------- simulation studies
routing:
	$(PYTHON) pipeline/routing/run_sweep.py $(if $(MAX_PARALLEL),--max-parallel $(MAX_PARALLEL),)
	$(PYTHON) pipeline/routing/build_data.py
	$(PYTHON) pipeline/routing/plot_tradeoff.py

caching:
	NJOBS=$(NJOBS) PYTHON=$(PYTHON) bash pipeline/caching/run_cachesim.sh

smoke:
	$(PYTHON) scripts/smoke_test.py

## ---------------------------------------------------------------- cleanup
clean:
	rm -f figures/*/*.png figures/paper/*/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	-find output/ -mindepth 1 -maxdepth 1 ! -name trace -exec rm -rf {} +
