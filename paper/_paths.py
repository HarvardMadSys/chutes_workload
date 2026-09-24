"""Paths shared by the six paper/<section>/reproduce.py scripts.

  PDF_DIR           figures/paper/<section>/  where a section's figures land
                                              ($CHUTES_FIGURE_ROOT overrides)
  CACHE_DIR         output/paper/cache/       query results (gitignored), seeded
                                              from the small ones committed in
                                              data/paper/cache/
  DB_DEFAULT        the DuckDB trace view     ($CHUTES_DB_PATH overrides)
  CHUTE_MODELS_CSV  data/paper/chute_models.csv, chute_id -> model name

The cache is shared across sections: 02 and 04 both read the highlighted-model
table, for example.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chutes_sim import artifact  # noqa: E402

__all__ = ["REPO", "PDF_DIR", "CACHE_DIR", "DB_DEFAULT", "CHUTE_MODELS_CSV", "section_paths"]

REPO = artifact.artifact_root()
CHUTE_MODELS_CSV = REPO / "data/paper/chute_models.csv"
CACHE_DIR = artifact.output_root() / "paper" / "cache"
PDF_DIR = Path(os.environ.get("CHUTES_FIGURE_ROOT") or REPO / "figures" / "paper")
DB_DEFAULT = artifact.db_path()

_COMMITTED_CACHE = REPO / "data/paper/cache"


def section_paths(section: str) -> tuple[Path, Path]:
    """``(PDF_DIR/<section>, CACHE_DIR)``, both created.

    Copies the committed query results into CACHE_DIR the first time, so the
    figures they back replot without the trace. Files already in CACHE_DIR are
    kept, so a --recompute result wins.
    """
    pdf_dir = PDF_DIR / section
    pdf_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for src in _COMMITTED_CACHE.glob("*.parquet"):
        if not (CACHE_DIR / src.name).exists():
            shutil.copy2(src, CACHE_DIR / src.name)
    return pdf_dir, CACHE_DIR
