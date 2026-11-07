#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  plot_baseline_vs_warmstart_multi.py
#
#      python analysis/plot_baseline_vs_warmstart_multi.py
#
#  Slice 7: baseline (E1_scratch, random init) vs warm start (E2_chain,
#  from slice 6). For each subject, a two-panel figure of epochs to converge
#  and validation loss, read at the FIRST learning-rate reduction (drop 1).
#
#  With two or more subjects listed in SUBJECTS below, it also makes an
#  aggregate figure: mean epochs, and the mean validation-loss RATIO
#  (warm start / baseline). The ratio is used rather than raw loss because
#  loss scales differ between subjects (anatomy, noise), independent of
#  whether warm-starting helps.
#
#  Reads only files already on disk. No training, no raw k-space, no GPU.
# ===============================================================
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import find_stats, drop1_record, BASELINE_TAG, JOINT_TAG  # noqa: E402

# label -> path to that subject's .mat file. By default, the one subject in
# config.py. Add more entries to get the aggregate figure, e.g.
#     SUBJECTS['subject02'] = '/path/to/subject02.mat'
SUBJECTS = {config.SUBJECT: config.DATA_FILE}
SLICE = 7
WARMSTART_TAG = JOINT_TAG
OUT_DIR = os.path.join(config.ANALYSIS_DIR, 'baseline_vs_warmstart')

GREY = '#8A8A8A'
TEAL = '#00806A'


def collect():
    rows = []
    for subj, data_file in SUBJECTS.items():
        base_path = find_stats(data_file, BASELINE_TAG, SLICE)
        warm_path = find_stats(data_file, WARMSTART_TAG, SLICE)
        missing = []
        if not base_path: missing.append(f'{BASELINE_TAG} slice {SLICE}')
        if not warm_path: missing.append(f'{WARMSTART_TAG} slice {SLICE}')
        if missing:
            print(f'  {subj}: SKIPPED - missing {", ".join(missing)} '
                  f'(not finished yet, or run failed)')
            continue
        base = drop1_record(base_path)
        warm = drop1_record(warm_path)
        if base is None or warm is None:
            print(f'  {subj}: SKIPPED - found the stats file but no drop-1 '
                  f'checkpoint recorded yet (run may still be mid-training)')
            continue
        rows.append(dict(subject=subj, base_epoch=base['epoch'], base_val=base['val'],
                         warm_epoch=warm['epoch'], warm_val=warm['val']))
        print(f'  {subj}: baseline {base["epoch"]} ep / {base["val"]:.4f}   '
              f'warm-start {warm["epoch"]} ep / {warm["val"]:.4f}')
    return rows


def two_panel(ax1, ax2, labels, epochs, vals, colors, title_suffix=''):
    b1 = ax1.bar(labels, epochs, color=colors, edgecolor='#1A1A1A', linewidth=0.8)
    for r, v in zip(b1, epochs):
        ax1.text(r.get_x()+r.get_width()/2, v + max(epochs)*0.02, f'{v:.0f}',
                 ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('epochs to converge (drop 1)')
    ax1.set_title(f'Epochs{title_suffix}')
    ax1.set_ylim(0, max(epochs)*1.18)

    b2 = ax2.bar(labels, vals, color=colors, edgecolor='#1A1A1A', linewidth=0.8)
    for r, v in zip(b2, vals):
        ax2.text(r.get_x()+r.get_width()/2, v + (max(vals)-min(vals)+1e-9)*0.06,
                 f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('validation loss (drop 1)')
    ax2.set_title(f'Validation loss{title_suffix}')
    span = max(vals) - min(vals)
    ax2.set_ylim(min(vals) - span*0.3 - 1e-6, max(vals) + span*0.5 + 1e-6)

    for a in (ax1, ax2):
        for s in a.spines.values(): s.set_visible(False)
        a.grid(axis='y', alpha=0.3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Collecting slice-7 baseline vs warm-start, drop-1 checkpoint, per dataset:\n')
    rows = collect()

    if not rows:
        print('\nNothing found yet - no individual or aggregate figure made. '
              'Re-run this once at least one dataset has finished both arms.')
        return

    # ---------------- one figure per dataset ----------------
    for r in rows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.2))
        two_panel(ax1, ax2, ['baseline\n(from scratch)', 'warm start\n(from slice 6)'],
                 [r['base_epoch'], r['warm_epoch']], [r['base_val'], r['warm_val']],
                 [GREY, TEAL])
        fig.suptitle(f'{r["subject"]}, slice {SLICE}', fontsize=13)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'{r["subject"]}_slice{SLICE}_baseline_vs_warmstart.png')
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  saved -> {out}')

    if len(rows) < 2:
        print('\n  One subject only - per-subject figure made, aggregate skipped '
              '(add more entries to SUBJECTS for it).')
        return

    # ---------------- aggregate across available datasets ----------------
    # Epochs are comparable across subjects, so their mean is used as-is.
    # Validation loss is not (different subjects have different loss scales),
    # so the aggregate uses the scale-invariant ratio warm start / baseline.
    n = len(rows)
    base_ep = np.array([r['base_epoch'] for r in rows], dtype=float)
    warm_ep = np.array([r['warm_epoch'] for r in rows], dtype=float)
    ratios = np.array([r['warm_val'] / r['base_val'] for r in rows], dtype=float)
    rng = np.random.default_rng(0)

    # Labels sit above the highest element of each bar (error-bar cap OR the
    # highest individual point), on a white box, so points never cover them.
    LABEL_BBOX = dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.6))
    labels = ['baseline\n(from scratch)', 'warm start\n(from slice 6)']
    means_ep = [base_ep.mean(), warm_ep.mean()]
    b1 = ax1.bar(labels, means_ep, color=[GREY, TEAL], edgecolor='#1A1A1A', linewidth=0.8,
                yerr=[base_ep.std(), warm_ep.std()], capsize=6)
    tops_ep = [max(base_ep.mean()+base_ep.std(), base_ep.max()),
              max(warm_ep.mean()+warm_ep.std(), warm_ep.max())]
    margin_ep = max(tops_ep) * 0.08
    for r, v, top in zip(b1, means_ep, tops_ep):
        ax1.text(r.get_x()+r.get_width()/2, top + margin_ep, f'{v:.0f}',
                 ha='center', fontsize=10, fontweight='bold', bbox=LABEL_BBOX, zorder=6)
    for x, vals in zip([0, 1], [base_ep, warm_ep]):
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax1.scatter(x + jitter, vals, color='#333333', s=22, zorder=5, alpha=0.75)
    ax1.set_ylabel('epochs to converge (drop 1)')
    ax1.set_title(f'Mean epochs  (n={n} dataset{"s" if n!=1 else ""})')
    ax1.set_ylim(0, max(tops_ep) + margin_ep*3.2)

    bar_color = TEAL if ratios.mean() <= 1.0 else '#A63A0F'
    b2 = ax2.bar(['warm start /\nbaseline'], [ratios.mean()], color=bar_color,
                edgecolor='#1A1A1A', linewidth=0.8, yerr=[ratios.std()], capsize=6, width=0.5)
    top_ratio = max(ratios.mean()+ratios.std(), ratios.max(), 1.0)
    span = max(0.02, ratios.max() - ratios.min())
    margin_ratio = max(span*0.18, 0.004)
    ax2.text(0, top_ratio + margin_ratio, f'{ratios.mean():.4f}',
             ha='center', fontsize=11, fontweight='bold', bbox=LABEL_BBOX, zorder=6)
    jitter = rng.uniform(-0.06, 0.06, len(ratios))
    ax2.scatter(jitter, ratios, color='#333333', s=24, zorder=5, alpha=0.8)
    ax2.axhline(1.0, color='#1A1A1A', lw=1.2, ls='--', alpha=0.7)
    ax2.text(0.32, 1.0 + span*0.08, 'baseline quality', fontsize=8, color='#555555',
             style='italic', bbox=LABEL_BBOX)
    ax2.set_ylabel('validation-loss ratio (warm start / baseline)')
    ax2.set_title(f'Mean quality ratio  (n={n} dataset{"s" if n!=1 else ""})')
    ax2.set_ylim(min(ratios.min(), 1.0) - span*0.6, top_ratio + margin_ratio*3.5)
    ax2.set_xlim(-0.5, 0.5)

    for a in (ax1, ax2):
        for s in a.spines.values(): s.set_visible(False)
        a.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Slice {SLICE}, baseline vs warm start - across datasets', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'AGGREGATE_slice{SLICE}_baseline_vs_warmstart_n{n}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n  aggregate saved -> {out}')
    print(f'  mean quality ratio: {ratios.mean():.4f}  (1.0 = identical to baseline; '
          f'below 1.0 = warm-start even lower loss than baseline)')
    if n < len(SUBJECTS):
        print(f'\n  NOTE: aggregate is built from {n} of {len(SUBJECTS)} datasets - '
              f're-run this script once the rest finish to update it.')


if __name__ == '__main__':
    main()
