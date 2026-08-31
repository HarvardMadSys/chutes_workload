#!/usr/bin/env python3
"""Figure 20 — routing-simulation tradeoff (DeepSeek V3.2 / MiniMax M2.5).

Reads the tidy sweep table produced by ``build_data.py`` and the
panel spec in ``config/figure20.json``, and renders:

  figures/routing/fig20_routing_tradeoff_mean.pdf / .png   the 2×4 composite

The "_mean" suffix names the panel-(d) ratio and is the filename the paper's
LaTeX includes (figures/repro/fig20_routing_tradeoff_mean.pdf).

Panels (all at load_metric = total_tokens):
  (a) token hit ratio          vs cache size   · N = 20
  (b) replication ratio        vs cache size   · N = 20
  (c) replication ratio        vs # instances  · cache = 25,000 tok/instance
  (d) load imbalance (%)       vs # instances  · cache = 5,000,000 tok/instance
      imbalance = (max/mean of active_tokens_per_sec_mean - 1) × 100, sticky excluded

Usage:
    python pipeline/routing/plot_tradeoff.py
    python pipeline/routing/plot_tradeoff.py --data path/to/figure20_sweep.csv
    python pipeline/routing/plot_tradeoff.py --print-values   # dump the plotted numbers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, NullLocator

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chutes_sim import artifact

REPO = artifact.artifact_root()
HERE = Path(__file__).resolve().parent

# ── Style (as the figure was originally drawn) ──
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_grid(ax) -> None:
    """Dashed major + minor grid on both axes (paper style)."""
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linestyle="--", linewidth=0.4)


def draw_panel(ax, sub: pd.DataFrame, panel: dict, style: dict) -> dict:
    """One policy line per panel; returns {policy: Line2D} for the legend."""
    x_col, y_col = panel["x"], panel["y"]
    handles: dict[str, plt.Line2D] = {}

    for policy in panel["policies"]:
        pdata = sub[sub["policy_label"] == policy].sort_values(x_col)
        if pdata.empty:
            continue
        (line,) = ax.plot(
            pdata[x_col],
            pdata[y_col],
            marker=style["policy_markers"][policy],
            markersize=6,
            color=style["policy_colors"][policy],
            label=style["policy_labels"][policy],
        )
        handles[policy] = line

    ax.set_xlabel(panel["xlabel"])
    ax.set_ylabel(panel["ylabel"])

    if panel["x_scale"] == "log":
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Auto-ylim as originally drawn: 6 % padding on the span.
    y = sub[sub["policy_label"].isin(panel["policies"])][y_col].dropna()
    if len(y):
        ymin, ymax = float(y.min()), float(y.max())
        span = ymax - ymin
        pad = 0.06 * span if span > 0 else max(abs(ymax) * 0.05, 1e-9)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.tick_params(axis="both", which="major", length=4.5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)
    style_grid(ax)
    return handles


def select(df: pd.DataFrame, model: str, load_metric: str, panel: dict) -> pd.DataFrame:
    sub = df[(df["model"] == model) & (df["load_metric"] == load_metric)]
    for col, val in panel["fixed"].items():
        sub = sub[sub[col] == val]
    sub = sub[sub["policy_label"].isin(panel["policies"])]
    if sub.empty:
        raise SystemExit(
            f"no rows for model={model} load_metric={load_metric} fixed={panel['fixed']}"
        )
    return sub


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "config/figure20.json")
    ap.add_argument("--data", type=Path, default=REPO / "data/routing/figure20_sweep.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/routing")
    ap.add_argument("--print-values", action="store_true", help="dump every plotted point")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    style = cfg["style"]
    slice_cfg = cfg["figure_slice"]
    load_metric = slice_cfg["load_metric"]
    panels = slice_cfg["panels"]
    panel_keys = ["a", "b", "c", "d"]
    rows = [("v32", cfg["caption_models"]["top_row"]), ("minimax", cfg["caption_models"]["bottom_row"])]

    df = pd.read_csv(args.data)
    # Panel (d) is the max/mean token-load ratio: the peak instance against the
    # fleet average. (An earlier revision also offered max/min; the artifact
    # now plots one imbalance definition so the figure has a single reading.)
    df["load_imbalance_pct"] = (df["max_mean_token_load"] - 1.0) * 100.0

    # ── Composite 2×4 ──
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.4))
    handles_seen: dict[str, plt.Line2D] = {}

    for r, (model, _label) in enumerate(rows):
        for c, key in enumerate(panel_keys):
            panel = panels[key]
            sub = select(df, model, load_metric, panel)
            handles_seen.update(draw_panel(axes[r, c], sub, panel, style))

            if args.print_values:
                print(f"\n[{key}] {panel['title']} · {model} · fixed={panel['fixed']}")
                print(
                    sub.pivot_table(index=panel["x"], columns="policy_label", values=panel["y"])
                    .round(4)
                    .to_string()
                )

    order = [p for p in style["policy_order"] if p in handles_seen]
    handles = [handles_seen[p] for p in order]
    labels = [style["policy_labels"][p] for p in order]
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels), frameon=False, columnspacing=1.6, handlelength=1.8,
    )
    fig.tight_layout(pad=0.6, rect=(0, 0.035, 1, 0.95))

    # Sub-captions (a)–(d) under each column; in the paper these are the
    # LaTeX subfigure captions, reproduced here so the PNG stands alone.
    for c, key in enumerate(panel_keys):
        x = 0.5 * (axes[1, c].get_position().x0 + axes[1, c].get_position().x1)
        fig.text(x, 0.006, f"({key}) {panels[key]['title']}", ha="center", va="bottom", fontsize=12)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = args.out_dir / f"fig20_routing_tradeoff_mean.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
