"""Path resolution shared by the five paper-figure reproduction scripts.

Each script under ``paper/<section>/reproduce.py`` is a faithful copy of the
original figure script; the only changes are the paths, and they all come
from here:

  PDF_DIR    figures/paper/<section>/     where that section's figures land
                                          ($CHUTES_FIGURE_ROOT overrides)
  CACHE_DIR  output/paper/cache/          heavy query results (gitignored),
                                          seeded from data/paper/cache/ unless
                                          $CHUTES_SEED_CACHE=0
  DB_DEFAULT the DuckDB view              $CHUTES_DB_PATH or config/workload.json
  CHUTE_MODELS_CSV  data/paper/chute_models.csv

The cache is shared across sections on purpose: several intermediates (the
model-highlight table, the per-(user, model) IAT quantiles) are computed by
one section and reused by another.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chutes_sim import artifact  # noqa: E402

__all__ = [
    "REPO", "PDF_DIR", "CACHE_DIR", "DB_DEFAULT",
    "CHUTE_MODELS_CSV", "ROUTING_SWEEP_CSV", "ROUTING_SWEEP_DIR",
    "section_paths",
]

REPO = artifact.artifact_root()

#: Committed inputs.
CHUTE_MODELS_CSV = REPO / "data/paper/chute_models.csv"
ROUTING_SWEEP_CSV = REPO / "data/routing/figure20_sweep.csv"

#: Generated roots. Both are overridable so a side run can write somewhere
#: else without touching the committed figures or the cache behind them.
CACHE_DIR = artifact.output_root() / "paper" / "cache"
PDF_DIR = Path(os.environ["CHUTES_FIGURE_ROOT"]) if os.environ.get("CHUTES_FIGURE_ROOT") \
    else REPO / "figures" / "paper"
ROUTING_SWEEP_DIR = artifact.routing_sweep_dir()

#: Committed cache seeds copied into CACHE_DIR on first use.
_CACHE_SEED_DIR = REPO / "data/paper/cache"

DB_DEFAULT = artifact.db_path()


def _seed_cache(cache_dir: Path) -> None:
    """Copy committed cache seeds into the working cache, once.

    These are the small query results that ship with the artifact so the
    figures they back can be replotted without re-querying the trace.
    Existing files are never overwritten, so a --recompute result wins.
    Set $CHUTES_SEED_CACHE=0 to skip seeding entirely — a run against a
    different trace must not inherit results computed from another one.
    """
    if os.environ.get("CHUTES_SEED_CACHE", "1") == "0":
        return
    if not _CACHE_SEED_DIR.is_dir():
        return
    for src in _CACHE_SEED_DIR.iterdir():
        if src.is_file() and not (cache_dir / src.name).exists():
            shutil.copy2(src, cache_dir / src.name)


def section_paths(section: str) -> tuple[Path, Path]:
    """``(PDF_DIR, CACHE_DIR)`` for one section, both created and seeded."""
    pdf_dir = PDF_DIR / section
    pdf_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _seed_cache(CACHE_DIR)
    return pdf_dir, CACHE_DIR
