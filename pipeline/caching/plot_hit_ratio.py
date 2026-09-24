#!/usr/bin/env python3
"""Stage 4 (caching study) — the paper's two token-hit-ratio figures.

Reads the libCacheSim sweep output (``data/caching/results.txt``) and draws
token hit ratio (1 - byte miss ratio) against cache size in tokens, in the
paper's own style: same rcParams, colors, markers and axes as the figure
script the draft was built from.

Only the seven algorithms the paper plots are drawn (FIFO, LRU, ARC,
LRB-BMR, GDSF, Sieve, BeladySize); S3FIFO and LHD are swept and present in
results.txt, but the paper does not plot them.

Usage:
  python pipeline/caching/plot_hit_ratio.py data/caching/results.txt \
      --out-dir figures/caching
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    AutoMinorLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator,
)

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'lines.linewidth': 1.5,
})


def style_grid(ax):
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.35, linestyle='--', linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.12, linestyle='--', linewidth=0.4)


def fmt_tokens(x, _pos):
    if x <= 0:
        return ''
    if x >= 1e9:
        return f'{x/1e9:g}B'
    if x >= 1e6:
        return f'{x/1e6:g}M'
    if x >= 1e3:
        return f'{x/1e3:g}K'
    return f'{x:g}'


ALGO_ORDER = ['FIFO', 'LRU', 'ARC', 'LRB-BMR', 'GDSF', 'Sieve', 'BeladySize']
ALGO_LABELS = {
    'FIFO': 'FIFO', 'LRU': 'LRU', 'ARC': 'ARC', 'LRB-BMR': 'LRB',
    'GDSF': 'GDSF', 'Sieve': 'Sieve', 'BeladySize': 'Belady (size-aware)',
}
ALGO_COLORS = {
    'FIFO': '#1f77b4', 'LRU': '#ff7f0e', 'ARC': '#2ca02c',
    'LRB-BMR': '#d62728', 'GDSF': '#9467bd', 'Sieve': '#8c564b',
    'BeladySize': '#000000',
}
ALGO_MARKERS = {
    'FIFO': 'o', 'LRU': 's', 'ARC': '^', 'LRB-BMR': 'D',
    'GDSF': 'v', 'Sieve': 'P', 'BeladySize': '*',
}

LINE_RE = re.compile(
    r'(\w+)\.oracleGeneral\s+(\S+)\s+cache size\s+(\d+)KiB,\s+'
    r'(\d+)\s+req,\s+miss ratio\s+([\d.]+),\s+byte miss ratio\s+([\d.]+)'
)


def load(results_path: Path) -> pd.DataFrame:
    records = []
    for trace, algo, size_kib, n_req, mr, tmr in LINE_RE.findall(results_path.read_text()):
        records.append({
            'trace': trace,
            'algorithm': algo,
            'cache_size_tokens': int(size_kib) * 1024,
            'n_req': int(n_req),
            'token_hit_ratio': 1.0 - float(tmr),
        })
    df = pd.DataFrame.from_records(records)
    return df.sort_values(['trace', 'algorithm', 'cache_size_tokens']).reset_index(drop=True)


def draw(df, trace, x_lim_max, legend_kwargs, out_path):
    sub = df[df['trace'] == trace]
    if x_lim_max is not None:
        sub = sub[sub['cache_size_tokens'] <= x_lim_max]

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    for algo in ALGO_ORDER:
        s = sub[sub['algorithm'] == algo].sort_values('cache_size_tokens')
        if s.empty:
            continue
        is_oracle = (algo == 'BeladySize')
        ax.plot(
            s['cache_size_tokens'], s['token_hit_ratio'],
            marker=ALGO_MARKERS[algo], markersize=4,
            linestyle='--' if is_oracle else '-',
            color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], linewidth=1.4,
        )

    y_pad = 0.05
    ax.set_xscale('log')
    ax.set_xlabel('Cache size (tokens)')
    ax.set_ylabel('Token hit ratio')
    ax.set_ylim(min(sub['token_hit_ratio']) - y_pad,
                min(1.0, sub['token_hit_ratio'].max() + 3 * y_pad))
    if x_lim_max is not None:
        ax.set_xlim(right=x_lim_max)

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=4.5, width=0.8)
    ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)

    style_grid(ax)
    ax.legend(**legend_kwargs)
    fig.tight_layout(pad=0.4)
    for ext in ['pdf', 'png']:
        fig.savefig(out_path.with_suffix(f'.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out-dir', default='figures/caching')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.results))
    print(f"Parsed {len(df)} records from {args.results}")

    # figure names + per-figure knobs, exactly as the paper draws them
    draw(df, 'minimax_m25', 5e6,
         dict(loc='lower right', frameon=False, ncol=2,
              columnspacing=0.8, handlelength=1.4, fontsize=7),
         out_dir / 'fig_hit_ratio_minimax_m25')
    draw(df, 'deepseek_v32', None,
         dict(loc='upper left', frameon=False, ncol=2,
              columnspacing=0.5, handlelength=1, fontsize=6),
         out_dir / 'fig_hit_ratio_deepseek_v32')


if __name__ == '__main__':
    main()
