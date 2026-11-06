# -*- coding: utf-8 -*-
# ===============================================================
#  stats_utils.py  --  shared helpers for the analysis scripts
#
#  Every training run writes ForPlottingLossStats_*.pth into a folder named
#      <stem>_60d/weights_..._<zero-based slice>_..._<run tag>
#  next to the data file. These helpers find that file for a given run and
#  slice, and read the numbers the figures need.
# ===============================================================
import os
import re
import glob

# Folder tags written by the run_*.py scripts
BASELINE_TAG = 'E1_scratch'   # run_experiment.py 1
JOINT_TAG = 'E2_chain'        # run_experiment.py 2  (weights + latent)
# run_isolation.py writes '..._isolate_all_src6_transfer_isoB' (weights only)
# and '..._isolate_all_src6_transfer_isoC' (latent only).


def find_stats(data_file, tag, slice_no):
    """Training-stats file for one run (tag) and 1-based slice, or None.

    Only folders that start with this data file's stem are searched, and the
    folder name must END in the tag (optionally followed by a stage suffix),
    so slice 1 can never match slice 11. For from-scratch runs the stage-2
    (joint training) file is preferred over the 10-epoch stage-1 warm-up.
    """
    zb = slice_no - 1
    data_dir = os.path.dirname(os.path.abspath(data_file))
    stem = os.path.splitext(os.path.basename(data_file))[0]
    pat = os.path.join(data_dir, f'{stem}_*', '**', f'weights_*_{zb}_*{tag}*',
                       'ForPlottingLossStats_*.pth')
    hits = [h for h in glob.glob(pat, recursive=True)
            if re.search(rf'_{re.escape(tag)}(_stage\d|_transfer|_xsubj)*$',
                         os.path.basename(os.path.dirname(h)))]
    stage2 = [h for h in hits if 'stage2' in h]
    chosen = sorted(stage2 or hits)
    return chosen[0] if chosen else None


def load_stats(stats_path):
    import torch
    return torch.load(stats_path, map_location='cpu', weights_only=False)


def drop1_record(stats_path):
    """Convergence point = the FIRST learning-rate reduction.

    Returns dict(epoch, val, stats) - the epoch at which the reduce-on-plateau
    scheduler first cut the learning rate (20 epochs without a >0.1%
    validation improvement), and the best validation loss up to then.
    None if the run never reached a reduction.
    """
    st = load_stats(stats_path)
    meta = {m['n_drop']: m for m in st.get('drop_meta', [])}
    d1 = meta.get(1)
    if d1 is None:
        return None
    return dict(epoch=d1['drop_epoch'], val=d1['best_val'], stats=st)
