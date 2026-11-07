#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  gen_STORM_transfer.py  --  training driver
#
#  Trains the self-supervised generative manifold model (a CNN generator
#  G driven by a per-frame latent z(t)) slice by slice, and supports
#  warm-starting each slice from a neighbouring slice.
#
#  MODE = 'baseline'  every slice trained from scratch (two-stage recipe:
#                     WARMUP_EPOCHS generator-only, then joint training).
#  MODE = 'transfer'  SOURCE_SLICE trained from scratch; every other slice
#                     is warm-started from a parent slice's weights + latent.
#                     TRANSFER_MODE='chain' walks outward from the source,
#                     each slice inheriting from its immediate neighbour.
#  MODE = 'ablate'    compare layer-freeze layouts on ONE target slice.
#  MODE = 'isolate'   on ONE target slice, separate the contribution of the
#                     transferred weights from that of the transferred latent.
#
#  Training length is not fixed: every run stops when the reduce-on-plateau
#  scheduler has cut the learning rate `stop_after_lr_reductions` times.
#  The *_EPOCHS settings are ceilings, not targets.
#
#  Do not edit settings here for normal use - the run_*.py scripts in the
#  repository root call configure() / run() with the published protocol.
#
#  Earlier freeze-layout ablation (slice 7 from slice 6, 40 epochs each):
#      layout   train%   time    final val
#      all      100.0%   2:53    0.6597     <- warm start, nothing frozen
#      ends3     70.7%   2:48    0.7002
#      ends2     69.7%   2:42    0.7590
#      ends1     69.2%   2:23    0.8803
#      first4    69.2%   2:11    0.9585
#      first3    37.9%   2:02    0.9549
#  Freezing does not buy speed (backprop still traverses frozen layers),
#  so the method uses warm-starting with every layer trainable.
# ===============================================================
import sys, os, importlib, time, gc
import numpy as np
import torch

import matplotlib
try:                              # inline figures in Jupyter, Agg when run as a script
    from IPython import get_ipython
    _NB = get_ipython() is not None
except Exception:
    _NB = False
if not _NB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
from scipy.io import savemat
from matplotlib.transforms import Bbox
import imageio.v2 as imageio

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CODE_DIR = os.path.dirname(os.path.abspath(__file__))   # repository root
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import dataOpNewKbnufft
from dataOpNewKbnufft import dataAndOperators
import generator_320
from generator_320 import generatorNew
import latentVariable
from latentVariable import latentVariableNew
import optimize_gen_sub
from optimize_gen_sub import optimize_generator
from runlog import RunLog, hms

# ===============================================================
#  EXPERIMENT SETTINGS  -- the only block you normally edit
# ===============================================================
MODE            = 'baseline'      # 'baseline' | 'transfer' | 'ablate' | 'isolate'
                                  # 'ablate' = train the source slice once, then try
                                  #   several freeze layouts on ONE target slice and
                                  #   print a comparison. Use this to pick a layout
                                  #   before committing to all 11 slices.
SLICES          = [7]             # <- the control run you still need.
                                  #    Use list(range(1, 12)) for a full pass.
SOURCE_SLICE    = 6               # trained from scratch in transfer mode

EXTERNAL_SOURCE_CKPT = None       # CROSS-SUBJECT TRANSFER.
                                  # None  -> the source slice is trained from scratch,
                                  #          as in every run so far.
                                  # path  -> the source slice is instead WARM-STARTED
                                  #          from a checkpoint trained on ANOTHER
                                  #          SUBJECT. Only the generator weights are
                                  #          taken; the latent always starts fresh,
                                  #          because z(t) encodes that subject's own
                                  #          breathing and means nothing here.
                                  #          After the source slice, the chain
                                  #          proceeds within this subject as usual.
                                  # The file must be {'G': state_dict, 'z': tensor},
                                  # which is what every run already saves.

CHAIN_HANDOFF   = 'final'         # WHICH WEIGHTS EACH SLICE PASSES TO THE NEXT.
                                  # 'final' -> the restored best-validation model at
                                  #            the END of training (after the last LR
                                  #            reduction).
                                  # 'drop1' -> the model as it stood at the FIRST
                                  #            learning-rate reduction (the setting
                                  #            used by run_experiment.py). If you use
                                  #            the drop-1 reconstructions, this is the
                                  #            setting that makes them a genuine
                                  #            one-drop chain - with 'final', a slice's
                                  #            drop-1 recon was trained from a parent
                                  #            that had already been through two drops.
                                  #            A slice that never reaches a drop falls
                                  #            back to 'final' (and says so).
                                  # Training still runs to STOP_AFTER_LR_REDUCTIONS
                                  # either way, so both drops are saved for every
                                  # slice; only what is handed DOWN the chain differs.

TRANSFER_MODE   = 'hub'           # 'hub' or 'chain'
                                  #   hub   : every slice inherits from SOURCE_SLICE.
                                  #           Each slice is an independent test of the
                                  #           same starting point, and all targets can
                                  #           run in parallel once the source is done.
                                  #   chain : each slice inherits from its NEIGHBOUR,
                                  #           walking outward from the source in both
                                  #           directions (6->7->8->... and 6->5->4->...).
                                  #           Shorter jumps, but strictly serial, and a
                                  #           failure far from the source could be either
                                  #           distance OR accumulated drift.

FREEZE_SPEC     = 'all'           # which conv layers stay TRAINABLE.
                                  # 'all' = warm start with no freezing, which the
                                  # ablation showed is both best and fastest.
                                  # Accepts a preset name, an int (first N convs),
                                  # or an explicit list like [0,1,2,3,9,10].
                                  #
                                  #   preset      trainable %   convs
                                  #   all             100.0%    everything (control)
                                  #   first3           37.9%    [0,1,2]
                                  #   first4           69.2%    [0,1,2,3]
                                  #   ends1            69.2%    [0,1,2,3,10]
                                  #   ends2            69.7%    [0,1,2,3,9,10]
                                  #   ends3            70.7%    [0,1,2,3,8,9,10]
                                  #   ends_wide        88.3%    [0,1,2,3,4,7,8,9,10]
                                  #   out_only          0.5%    [9,10]
                                  #
                                  # WHY 'ends2' RATHER THAN 'first3':
                                  # convs 0-2 work at 1x1..5x5 only, so with 'first3' a
                                  # transferred slice can express itself solely as a
                                  # 5x5 code decoded by slice-6-specific weights, and
                                  # the frozen 320x320 convs stamp slice 6's fine-scale
                                  # pattern onto it. 'ends2' costs almost the same
                                  # parameters but lets the output convs re-adapt.

ABLATE_SLICE    = 7               # target slice used by MODE='ablate'
ISOLATE_SLICE   = 7               # target slice used by MODE='isolate'
ABLATE_SPECS    = ['all', 'ends2'] # already swept the full set; these two suffice
                                  # to get motion numbers for the record
REUSE_SOURCE    = True            # cache the trained source slice to disk and reuse it,
                                  # so an ablation sweep doesn't retrain it every time

Z_INIT_TARGET   = 'source'        # 'source' (reuse source slice's z(t)) or 'ones'
TRANSFER_EPOCHS = 40              # fine-tune budget for a transferred slice
WARMUP_EPOCHS   = 10              # stage-1 epochs (scratch training only)
FINAL_EPOCHS    = 150             # stage-2 epochs (scratch training only)

MAKE_GIF        = False           # the preview panel now shows quality; the GIF
                                  # costs minutes per slice. Turn on for final runs.
RUN_NAME        = (f'{MODE}_{FREEZE_SPEC}_src{SOURCE_SLICE}'
                   + ('_chain' if MODE == 'transfer' and TRANSFER_MODE == 'chain' else ''))
# -> the run_*.py scripts override this

# GPU to use. Override with the RECON_DEVICE environment variable, e.g.
# RECON_DEVICE=cuda:1. A CUDA GPU is required (the NUFFT operators run on it).
gpu = torch.device(os.environ.get('RECON_DEVICE', 'cuda:0'))

params_template = {
    'name': 'parameters',
    'directory': '',
    'device': gpu,
    'filename': '',               # set by the run_*.py scripts from config.py
    'dtype': torch.float,
    'fastMode': True,
    'verbose': True,
    'im_size': (320, 320),
    'nintlPerFrame': 3,
    'nintlvsToDelete': 0,
    'nFramesDesired': 800,        # MUST satisfy nFramesDesired * nintlPerFrame <= arms in file.
                                  # e.g. a file with 2400 arms
                                  # -> 2400 / 3 = 800 frames maximum.
    'factor': 1,
    'nBatch': 6,
    'gen_base_size': 60,
    'gen_reg': 0.000001,
    'virtual_coils': 8,
    'mask_size': 0.90,
    'coilEst': 'espirit',
    'siz_l': 30,
    'Undersample': False,
    'Usp_arms_PerFrame': 3,
    'splitRatio': 0.05,           # 5% of each readout held out for validation
    'ssTrainMode': True,
    'splitDist': 'rightSkewed',
    'CoilMapPrecomputed': False,
    'coil_map_path': 'ESPIRIT_def.npy',  # only read when CoilMapPrecomputed=True
    # ---- new knobs ----
    'plot_every': 0,              # 0 = off. KEEP 0 FOR TIMING RUNS: the ablation ran
                                  # with plotting off, so the baseline must too or the
                                  # speedup ratio is measured against a handicapped arm.
                                  # Set 9 to watch training live in Jupyter.
    'plot_mode': 'auto',          # 'auto' = inline in Jupyter, PNG when run as a script.
                                  # force with 'inline' / 'file' / 'both' / 'off'.
    'detect_anomaly': False,      # set_detect_anomaly(True) is roughly 2x slower.
                                  # Keep False for any timing run.
    'stop_training': 0,           # separate patience counter; 0 = disabled.
    'stop_min_delta': 1e-3,       # minimum relative improvement that resets it.
    'lr_scheduler_threshold': 0.001,  # how much better a new epoch must be to
                                  # count as real progress (fraction of the
                                  # current best). LARGER = the patience counter
                                  # fills up sooner = the scheduler cuts sooner.
                                  # 0.001 is every run so far; see optimize_gen_sub.py.
    'stop_after_lr_reductions': 1,# STOP when the reduce-on-plateau scheduler cuts
                                  # the learning rate this many times. The scheduler
                                  # already uses the paper's settings (patience 20,
                                  # threshold 1e-3, factor 0.1), so it IS the plateau
                                  # detector - the algorithm chooses the epoch rather
                                  # than us. Measured on the real curve it fires once,
                                  # at epoch 85, against our arbitrary 40 which was
                                  # 6.5% short. 0 disables it.
    'restore_best': True,         # reconstruct from the MINIMUM-VALIDATION epoch,
                                  # as the paper states, not from the final epoch.
    'bestckpt_start_epoch': 5,    # start tracking the best-val checkpoint at epoch 5.
                                  # Low on purpose: a transfer run that converges
                                  # around epoch 10 must still have its minimum saved.
                                  # (Replaces the original `if(epoch > 150)` guard, which
                                  #  never fired because the run was exactly 150 epochs.)
}

ALPHA = [0.1, 0.15, 0.2, 0.25, 0.3, 0.1, 0.9, 0.2, 0.1, 0.5] * 3


def central_crop(array, crop_size=(120, 120)):
    h, w = array.shape[:2]
    sh = max(0, (h - crop_size[0]) // 2)
    sw = max(0, (w - crop_size[1]) // 2)
    return array[sh:sh+crop_size[0], sw:sw+crop_size[1]]


def result_dir(params, SLICE, extra=''):
    base = params['filename'].replace(
        '.mat', f'_RECONS_{params["gen_base_size"]}d/results_{params["coilEst"]}')
    d = (f'{base}_{RUN_NAME}{extra}_{SLICE}thSlice_{params["nintlPerFrame"]}arms_'
         f'{params["splitRatio"]}SplitRatio_{params["siz_l"]}latVec'
         f'{params["nFramesDesired"]}frms')
    os.makedirs(d, exist_ok=True)
    return d


def load_data(params, SLICE):
    """dataAndOperators, but with a readable error when the file has too few arms.

    dataOpNewKbnufft prints 'Too few interleaves' and then carries on to a reshape
    that fails with an unhelpful message. This turns that into a clear one.
    """
    try:
        return dataAndOperators(params)
    except ValueError as e:
        if 'reshape' not in str(e):
            raise
        need = params['nintlPerFrame'] * params['nFramesDesired']
        msg = str(e)
        have = None
        try:  # 'cannot reshape array of size 12825600 into shape (16,1703400,1)'
            size = int(msg.split('size')[1].split('into')[0])
            tgt = msg.split('shape (')[1].rstrip(')')
            nch = int(tgt.split(',')[0])
            nkpts = need and int(tgt.split(',')[1]) // need
            have = size // (nch * nkpts) if nkpts else None
        except Exception:
            pass

        print('\n' + '!'*78)
        print(f'  SLICE {SLICE}: not enough arms in the data file.')
        print('!'*78)
        print(f'  nFramesDesired   = {params["nFramesDesired"]}')
        print(f'  nintlPerFrame    = {params["nintlPerFrame"]}')
        print(f'  arms needed      = {need}')
        if have:
            print(f'  arms available   = {have}')
            print(f'\n  --> set nFramesDesired to at most {have // params["nintlPerFrame"]}')
        print(f'  file             = {params["filename"]}')
        print('!'*78 + '\n')
        raise


def build_transfer_plan(slices, source, mode):
    """Return (order, parent): the order to process slices and who each inherits from.

    hub   -> parent[s] = source for every s
    chain -> parent[s] = the neighbour one step closer to the source, walking
             outward in both directions so every hand-off is a single slice.
    """
    if source not in slices:
        raise ValueError(f'SOURCE_SLICE {source} is not in SLICES {slices}')

    if mode == 'hub':
        order = [source] + [s for s in slices if s != source]
        parent = {s: source for s in slices if s != source}
        return order, parent

    if mode != 'chain':
        raise ValueError("TRANSFER_MODE must be 'hub' or 'chain'")

    order, parent = [source], {}
    for direction in (sorted(s for s in slices if s > source),
                      sorted((s for s in slices if s < source), reverse=True)):
        prev = source
        for s in direction:
            parent[s] = prev
            order.append(s)
            prev = s
    return order, parent


# ---------------------------------------------------------------
#  train one slice from scratch  (the original two-stage recipe)
# ---------------------------------------------------------------
def train_from_scratch(dop, params, SLICE):
    alpha = torch.FloatTensor(ALPHA).to(gpu)

    G = generatorNew(params).to(torch.float32).cuda(gpu)
    z = latentVariableNew(params, init='ones', alpha=alpha, klreg=1)

    print(f'\n>> SCRATCH stage 1 (generator only, {WARMUP_EPOCHS} epochs) - slice {SLICE}')
    params['lr_g'], params['lr_z'] = 2e-4, 0.0
    G, z, th1, _, s1 = optimize_generator(dop, G, z, params,
                                        train_epoch=WARMUP_EPOCHS,
                                        save_weight=False, gtExists=False,
                                        tag=f'{RUN_NAME}_stage1')

    print(f'\n>> SCRATCH stage 2 (joint, {FINAL_EPOCHS} epochs) - slice {SLICE}')
    params['lr_g'], params['lr_z'] = 2e-5, 4e-4
    G.set_trainable_convs('all')                      # everything trainable
    G, z, th2, ep, s2 = optimize_generator(dop, G, z, params,
                                         train_epoch=FINAL_EPOCHS,
                                         save_weight=False, gtExists=False,
                                         tag=f'{RUN_NAME}_stage2')

    stats = dict(s2)
    stats['train_seconds'] = s1['train_seconds'] + s2['train_seconds']
    stats['epochs_run'] = s1['epochs_run'] + s2['epochs_run']
    stats['sec_per_epoch'] = stats['train_seconds']/max(1, stats['epochs_run'])
    return G, z, stats, th2


# ---------------------------------------------------------------
#  train one slice by transferring from the source slice
# ---------------------------------------------------------------
def train_by_transfer(dop, params, SLICE, src_G_state, src_z, spec=None, tag_extra='',
                      use_weights=True, z_mode=None):
    alpha = torch.FloatTensor(ALPHA).to(gpu)

    G = generatorNew(params).to(torch.float32).cuda(gpu)
    if use_weights:
        G.load_state_dict(src_G_state)                # <-- inherit the source slice
    else:
        print('  generator weights: RANDOM (source weights NOT used)', flush=True)

    spec = FREEZE_SPEC if spec is None else spec
    G.set_trainable_convs(spec)                       # <-- freeze layout
    print(f'\n>> TRANSFER slice {SLICE} from slice {SOURCE_SLICE}  [freeze: {spec}]')
    print(G.trainable_report())

    zm = Z_INIT_TARGET if z_mode is None else z_mode
    if zm == 'source':
        z = latentVariableNew(params, init='source', z_source=src_z,
                              alpha=alpha, klreg=1)
    else:
        z = latentVariableNew(params, init='ones', alpha=alpha, klreg=1)
    print(f'  latent init: {zm}', flush=True)

    # no stage-1 warm-up: the generator already knows the anatomy.
    # Same learning rates as the baseline's joint-training stage (2e-5 / 4e-4).
    params['lr_g'], params['lr_z'] = params.get('lr_g_transfer', 2e-5), 4e-4
    G, z, th, ep, stats = optimize_generator(dop, G, z, params,
                                             train_epoch=TRANSFER_EPOCHS,
                                             save_weight=False, gtExists=False,
                                              tag=f'{RUN_NAME}_transfer{tag_extra}')
    return G, z, stats, th


# ---------------------------------------------------------------
#  generate + save frames
# ---------------------------------------------------------------
def generate_and_save(G, z, params, SLICE, dirname, name_extra=''):
    """Generate all frames and save the 120x120 complex reconstruction (+ optional GIF).

    The uncropped 320x320 output is no longer saved. Frames are cropped as they are
    produced, so the full-resolution series is never held in memory: at 1700 frames
    that is ~196 MB instead of ~1.4 GB.
    """
    n_frames = params["nFramesDesired"]
    im_crop = (120, 120)

    t0 = time.time()
    images_cropped = np.zeros((n_frames, *im_crop, 2), dtype=np.float32)

    G.eval()
    with torch.no_grad():
        for i in range(n_frames):
            out = G(z.z_[i:i+1, :, :, :, 0]).squeeze(0).detach().cpu()
            if out.is_complex():
                r, im = out.real.squeeze().numpy(), out.imag.squeeze().numpy()
            elif out.ndim == 3 and out.shape[0] == 2:
                r, im = out[0].squeeze().numpy(), out[1].squeeze().numpy()
            else:
                raise ValueError(f'Unexpected generator output: {out.shape}')
            images_cropped[i, ..., 0] = central_crop(r, im_crop)
            images_cropped[i, ..., 1] = central_crop(im, im_crop)
    recon_s = time.time() - t0

    t0 = time.time()
    name120 = f'Recon_{SLICE}thSlice_{n_frames}frms{name_extra}.mat'
    savemat(os.path.join(dirname, name120), {
        "Recon120": images_cropped,
        "label": "Complex reconstruction [frames x 120 x 120 x {real, imag}]"})
    print(f'  saved {name120}  shape {images_cropped.shape}', flush=True)

    if MAKE_GIF:
        # TR_SLICE is how often THIS slice is revisited: all 11 slices are
        # interleaved, so 61 ms. One frame = nintlPerFrame arms of that slice.
        #   frame duration = 3 x 61 ms = 183 ms  ->  5.46 fps
        # The original code used TR = 61e-3/11 with nsl hardcoded to 1, which lost
        # the factor of 11 and played every GIF 11x too fast.
        TR_SLICE = 61e-3
        frame_dur = params['nintlPerFrame'] * TR_SLICE
        fps = 1.0 / frame_dur
        print(f'  gif: {frame_dur*1000:.0f} ms per frame -> {fps:.2f} fps', flush=True)
        mag = np.sqrt(images_cropped[..., 0]**2 + images_cropped[..., 1]**2)
        lo, hi = np.percentile(mag, 1.0), np.percentile(mag, 100.0)
        norm = np.clip((np.clip(mag, lo, hi) - lo) / (hi - lo + 1e-10) * 1.5, 0, 1)
        frames8 = (norm*255).astype(np.uint8)
        imageio.mimsave(os.path.join(dirname, name120.replace('.mat', '.gif')),
                        list(frames8), fps=fps, loop=0)
    save_s = time.time() - t0

    del images_cropped
    return recon_s, save_s



# ---------------------------------------------------------------
#  quick preview image (no GIF) so quality can be eyeballed fast
# ---------------------------------------------------------------
def save_preview(G, z, params, path, title='', n_probe=120, chunk=8):
    """Preview + motion check.

    Top row    : three example frames.
    Bottom row : temporal mean, temporal standard deviation (the motion map), and the
                 time-intensity curve of the most dynamic region.

    Returns peak_ratio = mean temporal std over the most dynamic 1% of pixels,
    divided by mean intensity inside the body mask.
    A near-static reconstruction gives a low value and a flat, featureless std map;
    real airway motion shows up as a bright band in the pharyngeal region. This
    matters most for transferred slices, where z(t) was copied from another slice and
    could in principle collapse to something static.
    """
    n = params['nFramesDesired']
    idx = np.unique(np.linspace(0, n - 1, min(n, n_probe)).astype(int))

    G.eval()
    mags = []
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            sel = idx[s:s + chunk]
            out = G(z.z_[sel, :, :, :, 0]).squeeze(1).detach().cpu()
            for k in range(out.shape[0]):
                o = out[k]
                if o.is_complex():
                    r, im = o.real.squeeze().numpy(), o.imag.squeeze().numpy()
                else:
                    r, im = o[0].squeeze().numpy(), o[1].squeeze().numpy()
                mags.append(np.sqrt(central_crop(r)**2 + central_crop(im)**2))
    mags = np.stack(mags)                      # (K, 120, 120)

    mean_img = mags.mean(0)
    std_img = mags.std(0)

    body = mean_img > 0.10 * mean_img.max()
    motion_ratio = float(std_img[body].mean() / (mean_img[body].mean() + 1e-9)) if body.any() else 0.0

    thr = np.percentile(std_img, 99.0)         # most dynamic 1% of pixels
    roi = std_img >= thr
    curve = mags[:, roi].mean(1)

    # Localized measure: airway motion is confined to a small region, so the global
    # average above is diluted by the static head. This is the number to compare.
    peak_ratio = float(std_img[roi].mean() / (mean_img[body].mean() + 1e-9)) if body.any() else 0.0

    fig, ax = plt.subplots(2, 3, figsize=(13, 8.6))
    picks = [0, len(mags)//2, len(mags)-1]
    for a, k in zip(ax[0], picks):
        a.imshow(mags[k], cmap='gray', vmin=0, vmax=0.9*np.percentile(mags[k], 99))
        a.set_title(f'frame {idx[k]}')
        a.axis('off')

    ax[1, 0].imshow(mean_img, cmap='gray', vmin=0, vmax=0.9*np.percentile(mean_img, 99))
    ax[1, 0].set_title('temporal mean')
    ax[1, 0].axis('off')

    im1 = ax[1, 1].imshow(std_img, cmap='inferno')
    ax[1, 1].set_title(f'temporal std (motion)   peak {peak_ratio:.3f} / global {motion_ratio:.3f}')
    ax[1, 1].axis('off')
    fig.colorbar(im1, ax=ax[1, 1], fraction=0.046)

    ax[1, 2].plot(idx, curve, lw=1)
    ax[1, 2].set_title('intensity in most dynamic 1% of pixels')
    ax[1, 2].set_xlabel('frame')
    ax[1, 2].grid(alpha=0.3)

    fig.suptitle(f'{title}   |   peak motion {peak_ratio:.3f}   global {motion_ratio:.3f}')
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches='tight')
    if _NB:
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            pass
    plt.close(fig)

    print(f'  motion: peak {peak_ratio:.4f}   global {motion_ratio:.4f}   '
          f'(flat map / low peak => near-static recon)', flush=True)
    return peak_ratio



# ---------------------------------------------------------------
#  source-weight cache: find it, rebuild it from an existing run, or train it
# ---------------------------------------------------------------
def handoff_state(G, z, stats, who=''):
    """The (weights, latent) a slice passes on, per CHAIN_HANDOFF."""
    if CHAIN_HANDOFF not in ('final', 'drop1'):
        raise ValueError(f"CHAIN_HANDOFF must be 'final' or 'drop1', got {CHAIN_HANDOFF!r}")
    if CHAIN_HANDOFF == 'drop1':
        snaps = [s for s in stats.get('drop_snapshots', []) if s.get('G_state') is not None]
        if snaps:
            s = snaps[0]
            print(f'  >> hand-off{who}: FIRST-DROP model (epoch {s["drop_epoch"]}, '
                  f'best epoch {s["best_epoch"]}, val {s["best_val"]:.4f}) - not the final one',
                  flush=True)
            return ({k: v.detach().cpu().clone() for k, v in s['G_state'].items()},
                    s['z_data'].detach().cpu().clone())
        print(f'  >> hand-off{who}: CHAIN_HANDOFF="drop1" but this slice never reached '
              f'a learning-rate drop - passing on the FINAL model instead', flush=True)
    return ({k: v.detach().cpu().clone() for k, v in G.state_dict().items()},
            z.z_.detach().cpu().clone())


def source_ckpt_path():
    return params_template['filename'].replace(
        '.mat', f'_SOURCEWEIGHTS_slice{SOURCE_SLICE}.pth')


def rebuild_source_cache_from_disk():
    """Recover the source weights from a finished run instead of retraining.

    Every optimize_generator() call writes ForPlottingLossStats_*.pth containing
    model_state and latVec_state. A previous 'transfer' or 'baseline' run trained
    the source slice from scratch, so its stage-2 checkpoint is already on disk.
    Returns True if the cache was written.
    """
    import glob
    p = params_template
    base = p['filename'].replace('.mat', f'_{p["gen_base_size"]}d')
    zero_based = SOURCE_SLICE - 1
    pat = os.path.join(base, '*', 'ForPlottingLossStats_*.pth')
    hits = glob.glob(pat)

    # keep checkpoints that belong to the source slice; prefer stage2 / scratch runs
    want = f'_{zero_based}_{p["nintlPerFrame"]}arms_'
    hits = [h for h in hits if want in os.path.basename(os.path.dirname(h))]
    if not hits:
        return False
    stage2 = [h for h in hits if 'stage2' in h]
    pick = sorted(stage2 or hits, key=os.path.getmtime)[-1]

    ck = torch.load(pick, map_location='cpu', weights_only=False)
    if 'model_state' not in ck or 'latVec_state' not in ck:
        return False
    torch.save({'G': ck['model_state'],
                'z': ck['latVec_state'].detach().cpu()}, source_ckpt_path())
    print(f'  rebuilt source cache from an existing run:\n    {pick}', flush=True)
    if 'stage2' not in pick:
        print('  NOTE: this was not a stage2 checkpoint - check it is the fully '
              'trained source, not a warm-up.', flush=True)
    return True


def get_source(train_if_missing=True):
    """Return (src_G_state, src_z), creating the cache if needed."""
    sp = source_ckpt_path()
    if not os.path.exists(sp):
        if not rebuild_source_cache_from_disk():
            if not train_if_missing:
                raise FileNotFoundError(
                    f'No source weights at {sp} and none recoverable from disk.')
            print(f'  no cached source found - training slice {SOURCE_SLICE} '
                  f'from scratch (about 13 min)', flush=True)
            p_src = params_template.copy()
            p_src['slice'] = SOURCE_SLICE - 1
            dop_src = load_data(p_src, SOURCE_SLICE)
            G, z, stats, _ = train_from_scratch(dop_src, p_src, SOURCE_SLICE)
            torch.save({'G': {k: v.detach().cpu() for k, v in G.state_dict().items()},
                        'z': z.z_.detach().cpu()}, sp)
            del dop_src, G, z
            torch.cuda.empty_cache(); gc.collect()
    ck = torch.load(sp, map_location=gpu, weights_only=False)
    print(f'  source weights: {sp}', flush=True)
    return ck['G'], ck['z'].to(gpu)


# ---------------------------------------------------------------
#  MODE = 'ablate' : compare freeze layouts on ONE target slice
# ---------------------------------------------------------------
def run_ablation():
    src_ckpt = params_template['filename'].replace(
        '.mat', f'_SOURCEWEIGHTS_slice{SOURCE_SLICE}.pth')

    # ---- source slice: train once, then cache ----
    if REUSE_SOURCE and os.path.exists(src_ckpt):
        print(f'  loading cached source weights: {src_ckpt}', flush=True)
        ck = torch.load(src_ckpt, map_location=gpu, weights_only=False)
        src_G_state, src_z = ck['G'], ck['z'].to(gpu)

        # Preview the source even from cache - we need its motion value as the
        # reference every transferred slice is compared against. No data load needed:
        # save_preview only runs the generator.
        p_src = params_template.copy()
        p_src['slice'] = SOURCE_SLICE - 1
        Gs = generatorNew(p_src).to(torch.float32).cuda(gpu)
        Gs.load_state_dict(src_G_state)
        zs = latentVariableNew(p_src, init='source', z_source=src_z,
                               alpha=torch.FloatTensor(ALPHA).to(gpu), klreg=1)
        src_motion = save_preview(Gs, zs, p_src,
                                  os.path.join(result_dir(p_src, SOURCE_SLICE),
                                               'preview_SOURCE.png'),
                                  f'slice {SOURCE_SLICE} - SOURCE (from cache)')
        print(f'  >> SOURCE peak motion = {src_motion:.4f}  '
              f'(reference for every transferred slice)', flush=True)
        del Gs, zs
        torch.cuda.empty_cache(); gc.collect()
    else:
        p_src = params_template.copy()
        p_src['slice'] = SOURCE_SLICE - 1
        dop_src = load_data(p_src, SOURCE_SLICE)
        G, z, stats, _ = train_from_scratch(dop_src, p_src, SOURCE_SLICE)
        src_G_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
        src_z = z.z_.detach().clone()
        torch.save({'G': src_G_state, 'z': src_z.cpu()}, src_ckpt)
        print(f'  cached source weights -> {src_ckpt}', flush=True)
        outdir = result_dir(p_src, SOURCE_SLICE)
        src_motion = save_preview(G, z, p_src, os.path.join(outdir, 'preview_SOURCE.png'),
                                  f'slice {SOURCE_SLICE} - source, trained from scratch')
        print(f'  >> SOURCE peak motion = {src_motion:.4f}  '
              f'(reference for every transferred slice)', flush=True)
        del dop_src, G, z
        torch.cuda.empty_cache(); gc.collect()

    # ---- target slice: load data ONCE, reuse across every layout ----
    p_tgt = params_template.copy()
    p_tgt['slice'] = ABLATE_SLICE - 1
    p_tgt['plot_every'] = 0                    # keep the sweep quiet
    dop = load_data(p_tgt, ABLATE_SLICE)

    outdir = result_dir(p_tgt, ABLATE_SLICE)
    rows = []
    for spec in ABLATE_SPECS:
        print('\n' + '='*78)
        print(f'  ABLATION  slice {ABLATE_SLICE}   freeze layout: {spec}')
        print('='*78, flush=True)
        G, z, stats, _ = train_by_transfer(dop, p_tgt, ABLATE_SLICE, src_G_state, src_z,
                                           spec=spec, tag_extra=f'_{spec}')
        n_tr = sum(q.numel() for q in G.parameters() if q.requires_grad)
        n_to = sum(q.numel() for q in G.parameters())
        mr = save_preview(G, z, p_tgt, os.path.join(outdir, f'preview_{spec}.png'),
                          f'slice {ABLATE_SLICE} - freeze layout "{spec}" - '
                          f'val {stats["final_val_loss"]:.4f}')
        rows.append((spec, 100*n_tr/n_to, stats['epochs_run'], stats['train_seconds'],
                     stats['final_val_loss'], stats['best_val_loss'], mr))
        del G, z
        torch.cuda.empty_cache(); gc.collect()

    print('\n' + '='*88)
    print(f'  ABLATION RESULTS - slice {ABLATE_SLICE} transferred from slice {SOURCE_SLICE}')
    print('='*94)
    print(f'{"layout":>12} {"train %":>9} {"epochs":>7} {"time":>10} '
          f'{"final val":>12} {"best val":>12} {"peak mot":>9}   vs all')
    print('-'*94)
    base = next((r[4] for r in rows if r[0] == 'all'), None)
    for spec, pct, ep, t, fv, bv, mr in rows:
        rel = f'{100*(fv-base)/abs(base):+.1f}%' if base else '-'
        print(f'{spec:>12} {pct:>8.1f}% {ep:>7} {hms(t):>10} {fv:>12.4f} {bv:>12.4f} '
              f'{mr:>9.3f}   {rel}')
    print('-'*94)
    print(f'  SOURCE slice {SOURCE_SLICE} peak motion = {src_motion:.3f}   <- the reference')
    print('  peak mot = temporal std over the most dynamic 1% of pixels, relative to')
    print('             mean body intensity. Compare against the source slice: a value')
    print('             far below it means the latent has stopped driving motion.')
    print(f'  previews written to: {outdir}')
    print('='*94 + '\n')
    return rows



# ---------------------------------------------------------------
#  MODE = 'isolate' : does the speed-up come from the weights or the latent?
# ---------------------------------------------------------------
#  Three conditions on ONE slice, each trained to the SAME stopping rule as
#  every other experiment (reduce-on-plateau, stop_after_lr_reductions) rather
#  than a fixed epoch count - TRANSFER_EPOCHS is a ceiling here, same as
#  everywhere else, not a target. The "neither" control (random weights, random
#  latent) that used to sit alongside these is dropped: that comparison only
#  meant something against a FIXED, possibly-too-short epoch cap, where it
#  answered "is the cap simply too short to need a warm start at all". Once
#  every condition runs to its own genuine stopping point, "neither" is just a
#  slower version of a from-scratch run you already have on record elsewhere -
#  it stops answering a live question.
#    A  weights from source + latent from source   (what the real chain does)
#    B  weights from source + latent fresh         -> isolates the WEIGHTS
#    C  weights fresh       + latent from source   -> isolates the LATENT
# ---------------------------------------------------------------
ISOLATE_CONDITIONS = [
    ('A  weights+latent', True,  'source'),
    ('B  weights only',   True,  'ones'),
    ('C  latent only',    False, 'source'),
]


def run_isolation():
    src_G_state, src_z = get_source()

    p = params_template.copy()
    p['slice'] = ISOLATE_SLICE - 1
    p['plot_every'] = 0
    dop = load_data(p, ISOLATE_SLICE)
    outdir = result_dir(p, ISOLATE_SLICE)

    rows = []
    for label, use_w, zm in ISOLATE_CONDITIONS:
        print('\n' + '='*78)
        print(f'  ISOLATION  slice {ISOLATE_SLICE}   condition: {label}')
        print('='*78, flush=True)
        G, z, stats, _ = train_by_transfer(dop, p, ISOLATE_SLICE, src_G_state, src_z,
                                           spec='all', tag_extra=f'_iso{label[0]}',
                                           use_weights=use_w, z_mode=zm)
        mot = save_preview(G, z, p, os.path.join(outdir, f'preview_iso_{label[0]}.png'),
                           f'slice {ISOLATE_SLICE} - {label} - val {stats["final_val_loss"]:.4f}')
        rows.append((label, use_w, zm, stats['train_seconds'],
                     stats['final_val_loss'], stats['best_val_loss'], mot))
        del G, z
        torch.cuda.empty_cache(); gc.collect()

    print('\n' + '='*92)
    print(f'  ISOLATION - slice {ISOLATE_SLICE}, each condition stopped by its own '
          f'validation-loss plateau (cap {TRANSFER_EPOCHS} epochs), source = slice {SOURCE_SLICE}')
    print('='*92)
    print(f'{"condition":>18} {"weights":>9} {"latent":>8} {"time":>9} '
          f'{"final val":>12} {"best val":>11} {"peak mot":>9}')
    print('-'*92)
    for lab, uw, zm, t, fv, bv, mo in rows:
        print(f'{lab:>18} {("source" if uw else "random"):>9} '
              f'{("source" if zm=="source" else "fresh"):>8} {hms(t):>9} '
              f'{fv:>12.4f} {bv:>11.4f} {mo:>9.3f}')
    print('-'*92)
    print(f'  Compare against the from-scratch baseline for slice {ISOLATE_SLICE} '
          f'(its own separate E1_scratch run).')
    print('='*92 + '\n')
    return rows


# ===============================================================
#  CONFIGURE / RUN  -- lets the job scripts set options without editing this file
# ===============================================================
def configure(params=None, **kw):
    """Override any top-level setting, then rebuild RUN_NAME.

    configure(MODE='baseline', SLICES=[1,2,3], params={'plot_every': 0})
    Unknown names raise, so a typo fails loudly instead of being silently ignored.
    """
    g = globals()
    for k, v in kw.items():
        if k not in g:
            raise KeyError(f'Unknown setting {k!r}. Valid: '
                           f'{sorted(n for n in g if n.isupper() and not n.startswith("_"))}')
        g[k] = v
    if params:
        for k in params:
            if k not in params_template:
                raise KeyError(f'Unknown params key {k!r}')
        params_template.update(params)
    g['RUN_NAME'] = (f"{g['MODE']}_{g['FREEZE_SPEC']}_src{g['SOURCE_SLICE']}"
                     + ('_chain' if g['MODE'] == 'transfer'
                        and g.get('TRANSFER_MODE') == 'chain' else ''))
    return g['RUN_NAME']


def run():
    """Print the resolved configuration, then execute."""
    print('#'*78)
    print(f'#  RUN_NAME      : {RUN_NAME}')
    print(f'#  MODE          : {MODE}')
    print(f'#  slices        : {SLICES}')
    if MODE != 'baseline':
        print(f'#  source slice  : {SOURCE_SLICE}   plan: {TRANSFER_MODE}')
        if EXTERNAL_SOURCE_CKPT:
            print(f'#  CROSS-SUBJECT : source warm-started from another subject')
            print(f'#                  {EXTERNAL_SOURCE_CKPT}')
        print(f'#  freeze layout : {FREEZE_SPEC}')
        print(f'#  z init        : {Z_INIT_TARGET}')
        print(f'#  epochs        : {TRANSFER_EPOCHS}')
    else:
        print(f'#  epochs        : {WARMUP_EPOCHS} + {FINAL_EPOCHS}')
    print(f'#  frames        : {params_template["nFramesDesired"]}')
    print(f'#  plot_every    : {params_template["plot_every"]}   '
          f'detect_anomaly: {params_template["detect_anomaly"]}')
    print(f'#  make gif      : {MAKE_GIF}')
    print('#'*78, flush=True)
    return main()


# ===============================================================
#  MAIN
# ===============================================================
def main():
    if MODE == 'ablate':
        return run_ablation()
    if MODE == 'isolate':
        return run_isolation()

    log_dir = os.path.dirname(params_template['filename'].replace('.mat', '_LOGS'))
    log = RunLog(os.path.join(log_dir, 'timing_logs'), run_name=RUN_NAME)

    print('='*78)
    print(f'  MODE          : {MODE}')
    print(f'  slices        : {SLICES}')
    if MODE == 'transfer':
        print(f'  source slice  : {SOURCE_SLICE}')
        print(f'  freeze layout : {FREEZE_SPEC}')
        print(f'  z init        : {Z_INIT_TARGET}')
        print(f'  epochs        : {TRANSFER_EPOCHS} (vs {WARMUP_EPOCHS}+{FINAL_EPOCHS} from scratch)')
    print('='*78)

    states = {}                      # slice -> (G_state on CPU, z on CPU)
    last_train_hist = None
    order, parent = SLICES, {}
    if MODE == 'transfer':
        order, parent = build_transfer_plan(SLICES, SOURCE_SLICE, TRANSFER_MODE)
        print(f'  transfer plan ({TRANSFER_MODE}):')
        for s in order:
            print(f'    slice {s:>2}  <- ' +
                  ('trained from scratch' if s == SOURCE_SLICE
                   else f'warm start from slice {parent[s]}'))

    for SLICE in order:
        slice_t0 = time.time()
        print('\n' + '='*78)
        print(f'  PROCESSING SLICE {SLICE}')
        print('='*78)

        params = params_template.copy()
        params['slice'] = SLICE - 1

        t0 = time.time()
        dop = load_data(params, SLICE)
        load_s = time.time() - t0
        print(f'  data loaded in {hms(load_s)}')

        is_source = (MODE == 'transfer' and SLICE == SOURCE_SLICE)
        par = parent.get(SLICE)
        if is_source and EXTERNAL_SOURCE_CKPT:
            # Cross-subject: start this subject's source slice from another
            # subject's weights, with a fresh latent.
            ck = torch.load(EXTERNAL_SOURCE_CKPT, map_location=gpu,
                            weights_only=False)
            ext = {k: v.to(gpu) for k, v in ck['G'].items()}
            print(f'  >> CROSS-SUBJECT source: weights from\n     '
                  f'{EXTERNAL_SOURCE_CKPT}', flush=True)
            print('     latent starts fresh - z(t) is subject specific', flush=True)
            G, z, stats, th = train_by_transfer(dop, params, SLICE, ext, None,
                                                tag_extra='_xsubj', z_mode='ones')
            last_train_hist = th
            mode_tag = 'xsubj'
            n_convs_tag = FREEZE_SPEC
            src_tag = 'ext'
            states[SLICE] = handoff_state(G, z, stats, ' (cross-subject source slice)')
        elif MODE == 'baseline' or is_source:
            G, z, stats, th = train_from_scratch(dop, params, SLICE)
            last_train_hist = th
            mode_tag = 'scratch'
            n_convs_tag = 'all'
            src_tag = '-'
            if is_source:
                states[SLICE] = handoff_state(G, z, stats, ' (source slice)')
                torch.save({'G': states[SLICE][0], 'z': states[SLICE][1]},
                           source_ckpt_path())
                print(f'  >> stored slice {SLICE} as the transfer source')
                print(f'  >> cached to {source_ckpt_path()}')
        else:
            if par not in states:
                raise RuntimeError(f'slice {SLICE} needs slice {par} first; '
                                   f'have {sorted(states)}')
            pG, pz = states[par]
            pG = {k: v.to(gpu) for k, v in pG.items()}
            G, z, stats, th = train_by_transfer(dop, params, SLICE, pG, pz.to(gpu))
            last_train_hist = th
            mode_tag = f'transfer-{TRANSFER_MODE}'
            n_convs_tag = FREEZE_SPEC
            src_tag = par
            if TRANSFER_MODE == 'chain':      # this slice becomes the next parent
                states[SLICE] = handoff_state(G, z, stats)
            del pG

        dirname = result_dir(params, SLICE)

        # One reconstruction per learning-rate drop, plus the final model. The
        # snapshots are earlier points on the SAME trajectory, so this costs only
        # the frame generation - no extra training.
        drop_rows = []
        for snap in stats.get('drop_snapshots', []):
            if snap['G_state'] is None:
                continue
            cur_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
            cur_z = z.z_.detach().clone()
            G.load_state_dict({k: v.to(gpu) for k, v in snap['G_state'].items()})
            z.z_.data = snap['z_data'].to(z.z_.device)
            d_dir = result_dir(params, SLICE, extra=f'_drop{snap["n_drop"]}')
            save_preview(G, z, params,
                         os.path.join(d_dir, f'preview_slice{SLICE}.png'),
                         f'slice {SLICE} - stopped at LR drop {snap["n_drop"]}')
            generate_and_save(G, z, params, SLICE, d_dir)
            print(f'  drop {snap["n_drop"]}: epoch {snap["drop_epoch"]}, '
                  f'best epoch {snap["best_epoch"]}, val {snap["best_val"]:.4f}, '
                  f'{snap["elapsed_s"]/60:.1f} min of training', flush=True)
            drop_rows.append(snap)
            G.load_state_dict(cur_G)               # restore the final model
            z.z_.data = cur_z

        motion = save_preview(G, z, params,
                              os.path.join(dirname, f'preview_slice{SLICE}.png'),
                              f'slice {SLICE} - {mode_tag}')
        print(f'  slice {SLICE}: peak motion {motion:.4f}, '
              f'final val {stats["final_val_loss"]:.4f}', flush=True)
        recon_s, save_s = generate_and_save(G, z, params, SLICE, dirname)
        total_s = time.time() - slice_t0

        print(f'  slice {SLICE}: {stats["epochs_run"]} epochs, '
              f'best epoch {stats["best_epoch"]}, '
              f'LR reductions {stats.get("lr_reductions", 0)} '
              f'{stats.get("lr_reduction_epochs", [])}, '
              f'stop: {stats["stop_reason"]}', flush=True)

        for snap in drop_rows:
            log.add(slice=SLICE, mode=f'{mode_tag}-drop{snap["n_drop"]}',
                    source_slice=src_tag, n_train_convs=n_convs_tag,
                    epochs_run=snap['drop_epoch'], load_s=load_s,
                    train_s=snap['elapsed_s'], recon_s=0.0, save_s=0.0,
                    total_s=load_s + snap['elapsed_s'],
                    sec_per_epoch=snap['elapsed_s']/max(1, snap['drop_epoch']),
                    final_val_loss=snap['best_val'], best_val_loss=snap['best_val'],
                    best_epoch=snap['best_epoch'], stop_reason=f'LR drop {snap["n_drop"]}')

        log.add(slice=SLICE, mode=mode_tag, source_slice=src_tag,
                n_train_convs=n_convs_tag, epochs_run=stats['epochs_run'],
                load_s=load_s, train_s=stats['train_seconds'],
                recon_s=recon_s, save_s=save_s, total_s=total_s,
                sec_per_epoch=stats['sec_per_epoch'],
                final_val_loss=stats['final_val_loss'],
                best_val_loss=stats['best_val_loss'],
                best_epoch=stats['best_epoch'],
                stop_reason=stats['stop_reason'])

        del dop, G, z
        torch.cuda.empty_cache()
        gc.collect()

    log.summary()
    return last_train_hist


# When imported by the run_*.py scripts nothing happens until run() is called,
# so those scripts control the settings.
if __name__ == '__main__':
    train_hist_last = run()
