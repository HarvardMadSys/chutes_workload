#!/usr/bin/env python3
"""Compare the PDFs under figures/ with config/figures.json, the paper's figure list.

Prints every figure the paper includes that is missing, and every PDF under
figures/ the paper does not include; exits 1 if either list is non-empty.

Usage:
    python paper/check_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    spec = json.loads((REPO / "config/figures.json").read_text())
    drawn_by = {name: "paper/" for names in spec["sections"].values() for name in names}
    drawn_by |= {name: v["generator"] for name, v in spec["external"].items()}
    have = {p.name: p for p in (REPO / "figures").rglob("*.pdf")}

    missing = sorted(set(drawn_by) - set(have))
    extra = sorted(set(have) - set(drawn_by))
    for name in missing:
        print(f"  MISSING  {name}  (drawn by {drawn_by[name]})")
    for name in extra:
        print(f"  EXTRA    {have[name].relative_to(REPO)}")
    print(f"{len(drawn_by) - len(missing)}/{len(drawn_by)} figures the paper includes are present; "
          f"{len(extra)} extra file(s) under figures/")
    return 1 if missing or extra else 0


if __name__ == "__main__":
    raise SystemExit(main())
