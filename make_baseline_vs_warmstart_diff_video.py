#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  make_baseline_vs_warmstart_diff_video.py
#
#      python make_baseline_vs_warmstart_diff_video.py
#
#  Video (MP4 + GIF) of all 11 slices, three rows per slice:
#     row 1  BASELINE   (E1_scratch, the reference)      - grayscale
#     row 2  WARM START (E2_chain)                       - grayscale, same window
#     row 3  |DIFFERENCE| x DIFF_SCALE                    - jet, one shared colorbar
#  with the per-slice SER (dB) of warm start against baseline above row 3.
#
#  SER: optimal complex least-squares scale fit of warm start to baseline,
#  then 10*log10(signal energy / error energy), averaged over frames. The
#  data have no ground truth, so the from-scratch baseline is the reference:
#  this measures how different warm start is from baseline, not whether
#  either is "correct".
#
#  Reads only the saved Recon120 .mat files. No GPU, no retraining.
# ===============================================================
import os
import re
import sys
import glob

import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

# ---------------------------------------------------------------
#  SETTINGS
# ---------------------------------------------------------------
SUBJECT = config.SUBJECT
DATA_FILE = config.DATA_FILE

BASELINE_TAG = 'E1_scratch'
WARMSTART_TAG = 'E2_chain'
SLICES = list(range(1, 12))

START, N_FRAMES, STRIDE = 0, 800, 1
ROTATE_CCW = 1
SCALE = 1.4
SPEED = 5.0
LO_PCT, HI_PCT = 1.0, 99.5
BRIGHTNESS_REF_SLICE = None         # None -> each slice windowed to its own
                                    #   baseline percentiles (every slice looks
                                    #   similarly exposed).
                                    # a slice number (e.g. 6) -> one window from
                                    #   that slice's baseline, used for all slices
                                    #   (true relative intensity between slices).
DIFF_SCALE = 5.0                   # |difference| is multiplied by this before display (x5, as in the abstract)
GAMMA = 0.9
CROP_SIZE = 90                     # center crop (pixels)
DIFF_CMAP = 'jet'

OUT_DIR = config.ANALYSIS_DIR
TR_FRAME = 3 * 61e-3

GOLD = (224, 164, 0)


# ---------------------------------------------------------------
def find_recon(tag, slice_no):
    """Final reconstruction for one run (tag) and slice.

    Strict suffix match, so slice 1 never matches '11thSlice' and the
    '_drop1' / '_drop2' folders are not picked up.
    """
    data_dir = os.path.dirname(os.path.abspath(DATA_FILE))
    stem = os.path.splitext(os.path.basename(DATA_FILE))[0]
    pat = os.path.join(data_dir, f'{stem}_*', '**', f'*{tag}*{slice_no}thSlice*', 'Recon_*.mat')
    hits = [h for h in glob.glob(pat, recursive=True)
            if re.search(rf'{re.escape(tag)}_{slice_no}thSlice(?![0-9])', h)]
    return sorted(hits)[0] if hits else None


def load_complex(path):
    m = loadmat(path)
    key = 'Recon120' if 'Recon120' in m else \
          next(k for k in m if not k.startswith('__') and m[k].ndim == 4)
    a = np.asarray(m[key], dtype=np.float32)
    cplx = a[..., 0] + 1j * a[..., 1]
    if ROTATE_CCW % 4:
        cplx = np.array([np.rot90(f, k=ROTATE_CCW) for f in cplx])
    return cplx


def center_crop(vol, size):
    """vol: (frames, H, W, ...). Crops H and W to size x size around the center."""
    h, w = vol.shape[1], vol.shape[2]
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return vol[:, y0:y0 + size, x0:x0 + size]


def ser_ref(ideal, recon):
    """Per-frame SER in dB, averaged over frames: optimal complex LS scale
    fit, then signal/error energy ratio."""
    sers = []
    for t in range(len(ideal)):
        i, r = ideal[t].ravel(), recon[t].ravel()
        denom = np.vdot(r, r).real
        alpha = (np.vdot(r, i) / denom) if denom > 1e-12 else 1.0
        err = i - alpha * r
        num = np.sum(np.abs(i) ** 2)
        den = np.sum(np.abs(err) ** 2)
        if den > 1e-12 and num > 0:
            sers.append(10 * np.log10(num / den))
    return float(np.mean(sers)) if sers else float('nan')


def build_lut(cmap_name, n=256):
    """256-entry RGB lookup table from a matplotlib colormap, built once -
    turning a normalized [0,1] value into a colored pixel is then a single
    fast numpy index operation per frame, not a per-pixel matplotlib call.
    """
    try:                                   # matplotlib >= 3.6
        cmap = matplotlib.colormaps[cmap_name].resampled(n)
    except AttributeError:                 # older matplotlib
        cmap = cm.get_cmap(cmap_name, n)
    return (np.array([cmap(i)[:3] for i in range(n)]) * 255).astype(np.uint8)


def apply_lut(vol01, lut):
    """vol01: array in [0,1], any shape. Returns the same shape + 3 (RGB)."""
    idx = np.clip((vol01 * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
    return lut[idx]


def to_uint8(vol, lo, hi):
    v = np.clip((vol - lo) / (hi - lo + 1e-9), 0, 1)
    if abs(GAMMA - 1.0) > 1e-6:
        v = v ** GAMMA
    return (v * 255).astype(np.uint8)


def _load_font(size):
    for cand in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ROW_LABELS = ['Baseline', 'Warm Start',
                 f'|Absolute Error| \u00d7 {DIFF_SCALE:.0f}']
    nrow = 3
    rows = [dict() for _ in range(nrow)]
    sers = {}
    used = []
    diff_raw = {}
    windows_hi = []

    # Optional single brightness window from BRIGHTNESS_REF_SLICE's baseline;
    # falls back (with a warning) to per-slice windows if that slice is missing.
    shared_lo = shared_hi = None
    if BRIGHTNESS_REF_SLICE is None:
        print('  brightness: each slice independently auto-windowed to its own '
              'percentiles (BRIGHTNESS_REF_SLICE = None)\n')
    else:
        ref_pb = find_recon(BASELINE_TAG, BRIGHTNESS_REF_SLICE)
        if ref_pb:
            ref_base = load_complex(ref_pb)[slice(START, START + N_FRAMES, STRIDE)]
            ref_base = center_crop(ref_base, CROP_SIZE)
            ref_mag = np.abs(ref_base)
            shared_lo = np.percentile(ref_mag, LO_PCT)
            shared_hi = np.percentile(ref_mag, HI_PCT)
            print(f'  shared brightness window, from slice {BRIGHTNESS_REF_SLICE} only: '
                  f'[{shared_lo:.4f}, {shared_hi:.4f}] - applied to every slice below\n')
        else:
            print(f'  WARNING: could not find slice {BRIGHTNESS_REF_SLICE} (the brightness '
                  f'reference) - falling back to each slice\'s own window instead.\n')

    for s in SLICES:
        pb = find_recon(BASELINE_TAG, s)
        pw = find_recon(WARMSTART_TAG, s)
        if not (pb and pw):
            missing = (BASELINE_TAG if not pb else '') + (' ' + WARMSTART_TAG if not pw else '')
            print(f'  slice {s:>2}: missing {missing.strip()} - skipped')
            continue

        base = load_complex(pb)
        warm = load_complex(pw)
        n = min(len(base), len(warm))
        sl = slice(START, min(START + N_FRAMES, n), STRIDE)
        base, warm = base[sl], warm[sl]
        base = center_crop(base, CROP_SIZE)
        warm = center_crop(warm, CROP_SIZE)

        base_mag = np.abs(base)
        warm_mag = np.abs(warm)
        diff_mag = np.abs(base - warm) * DIFF_SCALE

        if shared_lo is not None:
            lo, hi = shared_lo, shared_hi
        else:
            lo = np.percentile(base_mag, LO_PCT)        # fallback only
            hi = np.percentile(base_mag, HI_PCT)
        rows[0][s] = to_uint8(base_mag, lo, hi)
        rows[1][s] = to_uint8(warm_mag, lo, hi)
        diff_raw[s] = diff_mag                       # colorized once the shared vmax is known
        windows_hi.append(hi)

        sers[s] = ser_ref(base, warm)
        used.append(s)
        ser_str = f'{sers[s]:6.2f} dB' if sers[s] == sers[s] else '   NaN (!)'
        print(f'  slice {s:>2}: {base_mag.shape[0]} frames, SER = {ser_str}, '
              f'window [{lo:.4f}, {hi:.4f}]', flush=True)
        if sers[s] != sers[s]:   # NaN check
            print(f'    ^^ NaN means baseline and warm-start came back numerically '
                  f'IDENTICAL for this slice (zero error after alpha-fit). On real,')
            print(f'       independently-trained reconstructions this should essentially')
            print(f'       never happen - if you see this, it is worth checking whether')
            print(f'       find_recon() matched the same file twice, the way an earlier')
            print(f'       bug in this project did - see debug_path_check.py\'s approach.')

    if not used:
        print('\nNo complete baseline/warm-start pairs found. Run run_pipeline.py first, and check DATA_FILE in config.py.')
        return

    # ---- shared colorbar scale, set only once every slice has been seen ----
    # The colorbar ceiling is anchored to the IMAGE brightness (mean of the
    # grayscale window ceilings), not to the difference itself. If it were a
    # percentile of the scaled difference, it would grow with DIFF_SCALE and
    # cancel it out, so DIFF_SCALE would have no visible effect.
    DIFF_VMAX = float(np.mean(windows_hi))
    lut = build_lut(DIFF_CMAP)
    for s in used:
        rows[2][s] = apply_lut(np.clip(diff_raw[s] / (DIFF_VMAX + 1e-12), 0, 1), lut)
    print(f'\n  shared colorbar scale (|difference| x{DIFF_SCALE:.0f}): 0 to {DIFF_VMAX:.4f}  '
          f'(mean grayscale window ceiling)')

    # Columns are shown in reversed file order (label = min + max - slice) for
    # anatomical top-to-bottom orientation. Labels come from the full SLICES
    # range, so a missing slice cannot shift the other labels.
    order = sorted(used, reverse=True)
    label_of = lambda s: min(SLICES) + max(SLICES) - s
    nf = min(min(rows[r][s].shape[0] for s in used) for r in range(nrow))
    h, w = rows[0][order[0]].shape[1:]
    tw, th = int(w * SCALE), int(h * SCALE)
    ncol = len(order)
    s8 = max(1.0, SCALE * 0.8)

    # Layout: rotated row labels on the left, slice numbers above row 1, a gap
    # between rows 1 and 2, SER above row 3, thin border around every tile.
    LEFT_MARGIN = int(26 * s8)
    HDR1 = int(22 * s8)
    GAP = int(8 * s8)
    HDR2 = int(22 * s8)
    BORDER = max(1, int(1 * s8))

    font = _load_font(int(13 * s8))
    cb_font = _load_font(int(11 * s8))        # smaller, dedicated colorbar-tick font

    # Size the colorbar column from the measured tick-label width.
    _meas = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    def _text_w(text, fnt):
        try:
            return _meas.textlength(text, font=fnt)
        except AttributeError:
            return _meas.textsize(text, font=fnt)[0]
    cb_tick_values = [DIFF_VMAX, DIFF_VMAX / 2, 0.0]
    cb_text_w = int(max(_text_w(f'{v:.3f}', cb_font) for v in cb_tick_values))

    cb_pad_left = int(10 * s8)                 # gap between last column and the bar
    cb_bar_w = int(16 * s8)
    cb_text_gap = int(4 * s8)                  # gap between the bar and its numbers
    cb_right_margin = LEFT_MARGIN               # match the left side, as requested
    cbw = cb_pad_left + cb_bar_w + cb_text_gap + cb_text_w + cb_right_margin

    # Bottom margin so the lowest colorbar tick label is not cut off.
    BOTTOM_MARGIN = int(16 * s8)

    row_y0 = [HDR1, HDR1 + th + GAP, HDR1 + 2*th + GAP + HDR2]
    W = LEFT_MARGIN + ncol * tw + cbw
    H = row_y0[2] + th + BOTTOM_MARGIN

    canvas0 = Image.new('RGB', (W, H), (0, 0, 0))
    dr = ImageDraw.Draw(canvas0)

    # White background for the colorbar column (nothing else is drawn there).
    cb_col_x0 = LEFT_MARGIN + ncol * tw
    dr.rectangle([cb_col_x0, 0, W - 1, H - 1], fill=(255, 255, 255))

    def centered_text(x_left, x_right, y, text, fill):
        try:
            tw_px = dr.textlength(text, font=font)
        except AttributeError:
            tw_px = dr.textsize(text, font=font)[0]
        dr.text((x_left + (x_right - x_left - tw_px) / 2, y), text, fill=fill, font=font)

    # slice-number header, above row 1 - using the relabeled (reversed) number
    for i, s in enumerate(order):
        x0 = LEFT_MARGIN + i * tw
        centered_text(x0, x0 + tw, 4, f'Slice {label_of(s)}', (255, 255, 255))

    # SER header, above row 3
    for i, s in enumerate(order):
        x0 = LEFT_MARGIN + i * tw
        centered_text(x0, x0 + tw, row_y0[2] - HDR2 + 4, f'SER = {sers[s]:.1f} dB', (255, 255, 255))

    # Row labels, rotated 90 deg, left of each row.
    for r, text in enumerate(ROW_LABELS):
        # Shrink the font until the rotated label fits the row height
        # (7 px floor; hard-clipped below as a last resort).
        fsize = int(14 * s8)
        while fsize > 7:
            lf = _load_font(fsize)
            text_w = int(dr.textlength(text, font=lf)) + 8
            if text_w <= th:
                break
            fsize -= 1
        else:
            lf = _load_font(7)
            text_w = int(dr.textlength(text, font=lf)) + 8

        tmp = Image.new('RGB', (text_w, int(fsize * 1.4)), (0, 0, 0))
        tdr = ImageDraw.Draw(tmp)
        tdr.text((4, 0), text, fill=(210, 210, 210), font=lf)
        rotated = tmp.rotate(90, expand=True)
        if rotated.height > th:                     # hard safety clip, last resort
            excess = rotated.height - th
            rotated = rotated.crop((0, excess // 2, rotated.width, excess // 2 + th))
        ry = row_y0[r] + max(0, (th - rotated.height) // 2)
        canvas0.paste(rotated, (2, ry))

    # Colorbar - static, drawn once, spans the difference row.
    cb_y0 = row_y0[2]
    cb_x0 = LEFT_MARGIN + ncol * tw + cb_pad_left
    gradient = (lut[::-1][:, None, :] * np.ones((1, max(cb_bar_w, 1), 1), dtype=np.uint8))
    gradient = np.array(Image.fromarray(gradient).resize((cb_bar_w, th), Image.NEAREST))
    canvas0.paste(Image.fromarray(gradient), (cb_x0, cb_y0))
    dr.rectangle([cb_x0, cb_y0, cb_x0 + cb_bar_w - 1, cb_y0 + th - 1], outline=(0, 0, 0), width=1)
    for frac, val in [(0.0, DIFF_VMAX), (0.5, DIFF_VMAX / 2), (1.0, 0.0)]:
        yy = cb_y0 + int(frac * th)
        dr.text((cb_x0 + cb_bar_w + cb_text_gap, yy - 6), f'{val:.3f}',
                fill=(0, 0, 0), font=cb_font)

    frames = []
    for k in range(nf):
        im = canvas0.copy()
        for i, s in enumerate(order):
            x0 = LEFT_MARGIN + i * tw
            for r in range(nrow):
                tile_arr = rows[r][s][k]
                if tile_arr.ndim == 2:            # grayscale (rows 1-2) -> RGB
                    tile_arr = np.stack([tile_arr] * 3, -1)
                tile = Image.fromarray(tile_arr)
                if SCALE != 1.0:
                    tile = tile.resize((tw, th), Image.LANCZOS)
                im.paste(tile, (x0, row_y0[r]))
        # Tile borders are drawn after pasting, or the tiles would cover them.
        fdr = ImageDraw.Draw(im)
        for i in range(ncol):
            x0 = LEFT_MARGIN + i * tw
            for r in range(nrow):
                y0 = row_y0[r]
                fdr.rectangle([x0, y0, x0 + tw - 1, y0 + th - 1],
                             outline=(120, 120, 120), width=BORDER)
        frames.append(np.array(im))

    fps = SPEED / (TR_FRAME * STRIDE)

    def pad_even(a):
        ph, pw = a.shape[0] % 2, a.shape[1] % 2
        if not (ph or pw):
            return a
        pad_width = ((0, ph), (0, pw), (0, 0)) if a.ndim == 3 else ((0, ph), (0, pw))
        return np.pad(a, pad_width, mode='constant')
    frames = [pad_even(f) for f in frames]

    tag = f'{SUBJECT}_baseline_vs_warmstart_diff_{START}-{START+N_FRAMES}_stride{STRIDE}'
    base_out = os.path.join(OUT_DIR, f'compare_{tag}')

    print(f'\n  rows      : {nrow} ({", ".join(ROW_LABELS)})   columns: {ncol}')
    print(f'  frames    : {nf}  ({nf*TR_FRAME*STRIDE:.1f} s real time)')
    print(f'  mean SER across slices: {np.nanmean(list(sers.values())):.2f} dB')

    try:
        mp4 = base_out + '.mp4'
        imageio.mimwrite(mp4, frames,          # already RGB, no stacking needed
                         fps=fps, codec='libx264', macro_block_size=1,
                         pixelformat='yuv420p',
                         output_params=['-crf', '18', '-preset', 'slow'])
        print(f'  mp4       : {mp4}   [{os.path.getsize(mp4)/1e6:.1f} MB]')
    except Exception as e:
        print(f'  mp4 skipped ({e})')

    gif_path = base_out + '.gif'
    gframes = frames[::4]
    imageio.mimwrite(gif_path, gframes, fps=fps/4, loop=0, palettesize=256, subrectangles=True)
    print(f'  gif       : {gif_path}   [{os.path.getsize(gif_path)/1e6:.1f} MB]')

    print(f'\n  Rows 1-2 share ONE grayscale display window, taken from BASELINE, so')
    print(f'  any brightness difference between them is in the reconstruction, not the')
    print(f'  contrast. Row 3 is a {DIFF_CMAP} heatmap of |difference| x{DIFF_SCALE:.0f}, using ONE')
    print(f'  colorbar scale shared across all {ncol} slices (shown once, at the right edge)')
    print(f'  so slices stay comparable to each other - a slice with a visibly "hotter"')
    print(f'  row 3 than its neighbours genuinely has a larger difference, not just a')
    print(f'  different display window.\n')


if __name__ == '__main__':
    main()