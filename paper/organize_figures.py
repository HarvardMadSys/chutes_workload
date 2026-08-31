#!/usr/bin/env python3
"""Prune figures/ to exactly the set the paper includes, and audit it.

The section scripts draw every figure they know how to draw, and several draw
more than the paper uses: per-model variants, standalone legends, extra
panels. This removes those, so what ships is exactly the paper's figure set.

The keep-list is ``config/figures.json`` — the paper's ``\\includegraphics``
calls, frozen at release time, so this runs without the paper source.

Usage:
    python paper/organize_figures.py              # prune figures/paper/
    python paper/organize_figures.py --dry-run    # report the plan only
    python paper/organize_figures.py --check      # audit, change nothing
    python paper/organize_figures.py --keep-extras
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGURE_LIST = REPO / "config" / "figures.json"
PAPER_FIG_DIR = REPO / "figures" / "paper"
ALL_FIG_DIR = REPO / "figures"


def load_keep_list() -> tuple[set[str], dict[str, dict]]:
    """-> (figures drawn by paper/, figures drawn by the pipelines)"""
    spec = json.loads(FIGURE_LIST.read_text())
    ours = {name for names in spec["sections"].values() for name in names}
    return ours, spec["external"]


def check() -> int:
    """Report the artifact's figures against the keep-list."""
    ours, external = load_keep_list()
    want = ours | set(external)
    have = {p.name: p for p in ALL_FIG_DIR.rglob("*.pdf")}

    missing = sorted(want - set(have))
    extra = sorted(n for n in set(have) - want)
    for name in missing:
        src = external.get(name, {}).get("generator", "paper/")
        print(f"  MISSING  {name}  (drawn by {src})")
    for name in extra:
        print(f"  EXTRA    {have[name].relative_to(REPO)}")
    print(f"\n{len(want)} figures included by the paper: "
          f"{len(want) - len(missing)} present, {len(missing)} missing; "
          f"{len(extra)} extra file(s) under figures/")
    return 1 if (missing or extra) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="audit figures/ against the keep-list; change nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the pruning plan without touching any file")
    ap.add_argument("--figures", type=Path, default=None,
                    help="figure tree to prune (default: figures/paper/)")
    ap.add_argument("--keep-extras", action="store_true",
                    help="move extras into figures/paper/_not_in_paper/ instead of deleting")
    args = ap.parse_args()

    if args.check:
        return check()

    fig_dir = args.figures or PAPER_FIG_DIR
    if not fig_dir.is_dir():
        print(f"nothing to organize: {fig_dir} does not exist", file=sys.stderr)
        return 0

    keep, external = load_keep_list()
    keep = keep | set(external)

    extras_dir = fig_dir / "_not_in_paper"
    kept = removed = 0
    for pdf in sorted(fig_dir.rglob("*.pdf")):
        if extras_dir in pdf.parents:
            continue
        if pdf.name in keep:
            kept += 1
            continue
        removed += 1
        if args.dry_run:
            print(f"  would {'park' if args.keep_extras else 'remove'} "
                  f"{pdf.relative_to(REPO)}")
        elif args.keep_extras:
            extras_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pdf), str(extras_dir / pdf.name))
        else:
            pdf.unlink()

    print(f"{'would keep' if args.dry_run else 'kept'} {kept} figure(s) the paper "
          f"includes; {'would remove' if args.dry_run else 'removed'} {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
