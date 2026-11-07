#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  make_temporal_cut_ap.py
#
#      python make_temporal_cut_ap.py            (slice from SLICE below)
#      python make_temporal_cut_ap.py 7          (or pass the slice)
#
#  Temporal cut along the anterior-posterior direction (one image column,
#  all frames) for baseline (E1_scratch) and warm start (E2_chain), plus
#  |difference| x DIFF_SCALE. Saves a combined overview figure and the
#  individual pieces (temporal cuts, axial profiles with the cut line,
#  colorbar) for assembling a figure.
#
#  Also runs in Jupyter: paste it into a cell, or %run it.
#
#  Reads only the saved Recon120 .mat files. No GPU, no retraining.
# ===============================================================
import os
import re
import sys
import glob

import numpy as np
from scipy.io import loadmat
import matplotlib

try:
    from IPython import get_ipython
    _NB = get_ipython() is not None
except Exception:
    _NB = False
if not _NB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


# ===============================================================
# SETTINGS
# ===============================================================
SUBJECT = config.SUBJECT
DATA_FILE = config.DATA_FILE
SLICE = 5
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    SLICE = int(sys.argv[1])
# 1.0 = original brightness; 1.3 = brighter; 0.8 = darker.
BRIGHTNESS = 1.3

OUT_DIR = config.ANALYSIS_DIR

BASELINE_TAG = 'E1_scratch'
WARMSTART_TAG = 'E2_chain'

START = 0
N_FRAMES = 800
STRIDE = 1

ROTATE_CCW = 1
CROP_SIZE = 90

# Vertical column within the cropped image.
AP_COLUMN = 45

TR_FRAME = 3 * 61e-3  # Seconds per frame

DIFF_SCALE = 5.0    # x5, as in the abstract figures
DIFF_CMAP = 'jet'

# Display exploratory previews before the combined figure (Jupyter only).
SHOW_PREVIEW = _NB
PREVIEW_COLUMNS = [45, 46, 43]

# Display the combined overview (Jupyter only; it is saved either way).
SHOW_COMBINED = _NB

# Save the combined overview and six individual pieces.
# Individual pieces are saved without being displayed.
SAVE_FILES = True

# Shared grayscale window determined from baseline.
LO_PCT = 1.0
HI_PCT = 99.5

# Individual output sizes.
TEMPORAL_FIGSIZE = (5, 1.2)
AXIAL_FIGSIZE = (4, 2)
AXIAL_AXES_POSITION = [0, 0, 0.3, 1]


# ===============================================================
# HELPERS
# ===============================================================
def find_recon(tag, slice_no):
    """Final reconstruction for one run (tag) and slice, exact suffix match."""
    data_dir = os.path.dirname(os.path.abspath(DATA_FILE))
    stem = os.path.splitext(os.path.basename(DATA_FILE))[0]
    pattern = os.path.join(
        data_dir,
        f'{stem}_*',
        '**',
        f'*{tag}*{slice_no}thSlice*',
        'Recon_*.mat'
    )

    hits = [
        path
        for path in glob.glob(pattern, recursive=True)
        if re.search(
            rf'{re.escape(tag)}_{slice_no}thSlice(?![0-9])',
            path
        )
    ]

    return sorted(hits)[0] if hits else None


def load_complex(path):
    """Load a reconstruction shaped (frames, height, width, 2)."""
    mat = loadmat(path)

    if 'Recon120' in mat:
        key = 'Recon120'
    else:
        candidates = [
            key
            for key, value in mat.items()
            if not key.startswith('__')
            and isinstance(value, np.ndarray)
            and value.ndim == 4
            and value.shape[-1] == 2
        ]

        if not candidates:
            raise ValueError(
                'No reconstruction shaped (frames, H, W, 2) found in:\n'
                f'{path}'
            )

        key = candidates[0]

    array = np.asarray(mat[key])

    if array.ndim != 4 or array.shape[-1] != 2:
        raise ValueError(
            f'{key} has shape {array.shape}; '
            'expected (frames, height, width, 2).'
        )

    array = array.astype(np.float32)
    volume = array[..., 0] + 1j * array[..., 1]

    if ROTATE_CCW % 4:
        volume = np.rot90(
            volume,
            k=ROTATE_CCW,
            axes=(1, 2)
        ).copy()

    return volume


def center_crop(volume, size):
    """Center crop to size x size."""
    height, width = volume.shape[1:3]

    y0 = max((height - size) // 2, 0)
    x0 = max((width - size) // 2, 0)

    return volume[:, y0:y0 + size, x0:x0 + size]


def finish_figure(
    fig,
    filename,
    dpi=500,
    pad_inches=0.05,
    show=False
):
    """Optionally save, optionally display, then close."""
    if SAVE_FILES:
        path = os.path.join(OUT_DIR, filename)

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=pad_inches,
            facecolor='white'
        )

        print(f'Saved -> {path}')

    if show:
        plt.show()

    plt.close(fig)


# ===============================================================
# MAIN
# ===============================================================
def main():
    if not np.isfinite(BRIGHTNESS) or BRIGHTNESS <= 0:
        raise ValueError(
            'BRIGHTNESS must be a finite number greater than zero.'
        )

    if START < 0 or N_FRAMES <= 0 or STRIDE <= 0:
        raise ValueError(
            'START must be nonnegative; '
            'N_FRAMES and STRIDE must be positive.'
        )

    if SAVE_FILES:
        os.makedirs(OUT_DIR, exist_ok=True)

    baseline_path = find_recon(BASELINE_TAG, SLICE)
    warmstart_path = find_recon(WARMSTART_TAG, SLICE)

    if not baseline_path or not warmstart_path:
        missing = []

        if not baseline_path:
            missing.append(BASELINE_TAG)

        if not warmstart_path:
            missing.append(WARMSTART_TAG)

        raise FileNotFoundError(
            f'Slice {SLICE}: missing {", ".join(missing)}. '
            'Run run_pipeline.py first, and check SLICE and DATA_FILE in config.py.'
        )

    baseline = load_complex(baseline_path)
    warmstart = load_complex(warmstart_path)

    n_available = min(len(baseline), len(warmstart))

    if START >= n_available:
        raise ValueError(
            f'START={START}, but only {n_available} '
            'common frames are available.'
        )

    frame_selection = slice(
        START,
        min(START + N_FRAMES, n_available),
        STRIDE
    )

    baseline = center_crop(
        baseline[frame_selection],
        CROP_SIZE
    )
    warmstart = center_crop(
        warmstart[frame_selection],
        CROP_SIZE
    )

    if baseline.shape != warmstart.shape:
        raise ValueError(
            'Shapes do not match after cropping: '
            f'{baseline.shape} versus {warmstart.shape}.'
        )

    baseline_mag = np.abs(baseline)
    warmstart_mag = np.abs(warmstart)

    nf, height, width = baseline_mag.shape

    column = AP_COLUMN if AP_COLUMN is not None else width // 2

    if not 0 <= column < width:
        raise ValueError(
            f'AP_COLUMN={column} is out of range '
            f'for a {width}-pixel-wide crop.'
        )

    # Vertical cut: (frames, height) -> (height, frames).
    baseline_cut = baseline_mag[:, :, column].T
    warmstart_cut = warmstart_mag[:, :, column].T

    # Difference is calculated from the original magnitudes.
    # BRIGHTNESS does not change these data.
    difference_cut = (
        np.abs(baseline_cut - warmstart_cut) * DIFF_SCALE
    )

    # -----------------------------------------------------------
    # Shared grayscale display window and brightness adjustment.
    # Used for ALL displayed and saved grayscale images.
    # -----------------------------------------------------------
    lo = float(np.percentile(baseline_mag, LO_PCT))
    hi = float(np.percentile(baseline_mag, HI_PCT))

    if hi <= lo:
        hi = lo + np.finfo(np.float32).eps

    # Colorbar ceiling for the difference panels, anchored to the IMAGE
    # brightness (before the BRIGHTNESS adjustment). If it were a percentile
    # of the scaled difference itself, it would grow with DIFF_SCALE and
    # cancel it out, so DIFF_SCALE would have no visible effect.
    difference_vmax = max(hi, np.finfo(np.float32).eps)

    hi = lo + (hi - lo) / BRIGHTNESS

    mid_frame_baseline = baseline_mag[nf // 2]
    mid_frame_warmstart = warmstart_mag[nf // 2]

    extent = [
        0,
        nf * TR_FRAME * STRIDE,
        0,
        height
    ]

    difference_label = (
        f'|Absolute Difference| × {DIFF_SCALE:g}'
    )

    print(
        f'{SUBJECT}, slice {SLICE}: {nf} frames, '
        f'crop {height}×{width}, '
        f'vertical A-P cut at column {column} (0-indexed)'
    )
    print(f'Brightness factor: {BRIGHTNESS:g}')
    print(
        f'Adjusted grayscale display window: '
        f'[{lo:.4f}, {hi:.4f}]'
    )
    print(
        f'Difference scale (×{DIFF_SCALE:g}): '
        f'0 to {difference_vmax:.4f}'
    )

    # ===========================================================
    # EXPLORATORY PREVIEWS
    # ===========================================================
    if SHOW_PREVIEW:
        for preview_column in PREVIEW_COLUMNS:
            if not 0 <= preview_column < width:
                print(
                    f'Preview column {preview_column} '
                    'is out of range; skipped.'
                )
                continue

            preview_baseline = (
                baseline_mag[:, :, preview_column].T
            )
            preview_warmstart = (
                warmstart_mag[:, :, preview_column].T
            )
            preview_difference = (
                np.abs(preview_baseline - preview_warmstart)
                * DIFF_SCALE
            )

            preview_vmax = difference_vmax

            preview_panels = [
                (
                    'Baseline',
                    mid_frame_baseline,
                    preview_baseline,
                    'gray',
                    lo,
                    hi
                ),
                (
                    'Warm Start',
                    mid_frame_warmstart,
                    preview_warmstart,
                    'gray',
                    lo,
                    hi
                ),
                (
                    difference_label,
                    mid_frame_baseline,
                    preview_difference,
                    DIFF_CMAP,
                    0,
                    preview_vmax
                ),
            ]

            for (
                title,
                thumbnail,
                data,
                cmap_name,
                vmin,
                vmax
            ) in preview_panels:
                fig = plt.figure(figsize=(8, 2.5))

                ax_thumbnail = fig.add_axes(
                    [0, 0, 0.3, 1]
                )
                ax_thumbnail.imshow(
                    thumbnail,
                    cmap='gray',
                    vmin=lo,
                    vmax=hi
                )
                ax_thumbnail.axvline(
                    preview_column,
                    color='lime',
                    linewidth=2
                )
                ax_thumbnail.set_title(
                    f'Cut at column {preview_column}',
                    fontsize=9
                )
                ax_thumbnail.axis('off')

                ax_cut = fig.add_axes(
                    [0.35, 0.3, 0.65, 0.4]
                )
                ax_cut.imshow(
                    data,
                    aspect='auto',
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    origin='upper'
                )
                ax_cut.set_title(title, fontsize=9)
                ax_cut.axis('off')

                plt.show()
                plt.close(fig)

    # ===========================================================
    # COMBINED OVERVIEW
    # ===========================================================
    if SHOW_COMBINED or SAVE_FILES:
        fig = plt.figure(figsize=(13.0, 7.5))

        grid = fig.add_gridspec(
            3,
            4,
            width_ratios=[0.35, 0.65, 7.0, 0.18],
            height_ratios=[1, 1, 1],
            wspace=0.015,
            hspace=0.12
        )

        row_labels = [
            'Baseline',
            'Warm Start',
            difference_label
        ]

        for row, label in enumerate(row_labels):
            ax_label = fig.add_subplot(grid[row, 0])
            ax_label.text(
                0.5,
                0.5,
                label,
                rotation=90,
                ha='center',
                va='center',
                fontsize=11,
                transform=ax_label.transAxes
            )
            ax_label.axis('off')

        thumbnails = [
            mid_frame_baseline,
            mid_frame_warmstart
        ]

        for row, thumbnail in enumerate(thumbnails):
            ax_thumbnail = fig.add_subplot(grid[row, 1])

            ax_thumbnail.imshow(
                thumbnail,
                cmap='gray',
                vmin=lo,
                vmax=hi
            )
            ax_thumbnail.axvline(
                column,
                color='#00FF66',
                linewidth=1.6
            )
            ax_thumbnail.set_title(
                'Axial Profile',
                fontsize=10
            )
            ax_thumbnail.set_xticks([])
            ax_thumbnail.set_yticks([])

            for spine in ax_thumbnail.spines.values():
                spine.set_visible(False)

        panels = [
            (baseline_cut, 'gray', lo, hi),
            (warmstart_cut, 'gray', lo, hi),
            (
                difference_cut,
                DIFF_CMAP,
                0,
                difference_vmax
            ),
        ]

        plotted_images = []

        for row, (
            data,
            cmap_name,
            vmin,
            vmax
        ) in enumerate(panels):
            ax = fig.add_subplot(grid[row, 2])

            image = ax.imshow(
                data,
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
                extent=extent,
                origin='upper'
            )

            ax.set_xticks([])
            ax.set_yticks([])
            plotted_images.append(image)

        colorbar_axis = fig.add_subplot(grid[2, 3])
        fig.colorbar(
            plotted_images[2],
            cax=colorbar_axis
        )

        fig.suptitle(
            'Mid-temporal cut along anterior-posterior direction\n'
            f'{SUBJECT}, Slice {SLICE} ({nf} frames)',
            fontsize=13
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.97,
            top=0.90,
            bottom=0.08
        )

        finish_figure(
            fig,
            f'{SUBJECT}_slice{SLICE}_AP_temporal_cut.png',
            dpi=200,
            pad_inches=0.1,
            show=SHOW_COMBINED
        )

    # Individual figures are saved without being displayed.
    if not SAVE_FILES:
        return

    # ===========================================================
    # INDIVIDUAL TEMPORAL CUTS
    # Matches reference: figsize=(5, 1.2), tight_layout().
    # ===========================================================
    temporal_panels = [
        ('Baseline', baseline_cut, 'gray', lo, hi),
        ('WarmStart', warmstart_cut, 'gray', lo, hi),
        (
            'AbsoluteError',
            difference_cut,
            DIFF_CMAP,
            0,
            difference_vmax
        ),
    ]

    for (
        name,
        data,
        cmap_name,
        vmin,
        vmax
    ) in temporal_panels:
        fig, ax = plt.subplots(
            figsize=TEMPORAL_FIGSIZE
        )

        ax.imshow(
            data,
            aspect='auto',
            cmap=cmap_name,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            origin='upper'
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        fig.tight_layout()

        finish_figure(
            fig,
            f'{SUBJECT}_slice{SLICE}_temporal_cut_{name}.png',
            show=False
        )

    # ===========================================================
    # INDIVIDUAL AXIAL PROFILES
    # Matches reference:
    # figsize=(4, 2), add_axes([0, 0, 0.3, 1]).
    # ===========================================================
    axial_panels = [
        ('Baseline', mid_frame_baseline),
        ('WarmStart', mid_frame_warmstart),
    ]

    for name, thumbnail in axial_panels:
        fig = plt.figure(
            figsize=AXIAL_FIGSIZE
        )

        ax = fig.add_axes(
            AXIAL_AXES_POSITION
        )

        ax.imshow(
            thumbnail,
            cmap='gray',
            vmin=lo,
            vmax=hi
        )
        ax.axvline(
            column,
            color='lime',
            linewidth=2
        )
        ax.axis('off')

        finish_figure(
            fig,
            f'{SUBJECT}_slice{SLICE}_axial_profile_{name}.png',
            show=False
        )

    # ===========================================================
    # INDIVIDUAL COLORBAR
    # ===========================================================
    fig = plt.figure(figsize=(0.4, 1.2))
    colorbar_axis = fig.add_axes(
        [0.1, 0.05, 0.3, 0.9]
    )

    scalar_mappable = plt.cm.ScalarMappable(
        cmap=DIFF_CMAP,
        norm=plt.Normalize(
            vmin=0,
            vmax=difference_vmax
        )
    )

    fig.colorbar(
        scalar_mappable,
        cax=colorbar_axis
    )

    finish_figure(
        fig,
        f'{SUBJECT}_slice{SLICE}_colorbar.png',
        show=False
    )


if __name__ == '__main__':
    main()