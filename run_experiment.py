#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  run_experiment.py  --  run ONE experiment on the subject in config.py
#
#      python run_experiment.py 1     # EXP 1: baseline, all 11 slices from scratch
#      python run_experiment.py 2     # EXP 2: warm-start chain from slice 6
#
#  EXP 1  Every slice is trained independently from random initialization.
#  EXP 2  Slice 6 is trained from scratch, then the chain walks outward in
#         both directions (6->7->...->11 and 6->5->...->1). Each slice is
#         warm-started from its neighbour's weights AND latent, as they
#         stood at that neighbour's first learning-rate reduction
#         (CHAIN_HANDOFF='drop1'). Nothing is frozen.
#
#  Both experiments use the same stopping protocol: train until the
#  reduce-on-plateau scheduler (threshold 0.001) has cut the learning rate
#  twice, with a 700-epoch ceiling. The model at each reduction is saved,
#  so both drop-1 and drop-2 reconstructions exist for every slice.
#
#  To run both back to back, use run_pipeline.py.
# ===============================================================
import os
import sys

import config

if len(sys.argv) != 2 or sys.argv[1] not in ('1', '2'):
    raise SystemExit('Usage: python run_experiment.py <EXP>\n'
                     '  EXP 1 = baseline, all 11 slices from scratch\n'
                     '  EXP 2 = warm-start chain from slice 6')
EXP = int(sys.argv[1])

# ---------------------------------------------------------------
#  Protocol
# ---------------------------------------------------------------
SLICES = list(range(1, 12))       # all 11 slices
SOURCE_SLICE = 6                  # chain starts here (EXP 2)
CAP = 700                         # epoch ceiling - training normally stops earlier
CHAIN_HANDOFF = 'drop1'
LR_SCHEDULER_THRESHOLD = 0.001

DATA_FILE = config.resolve_data_file()

if config.RECON_DIR not in sys.path:
    sys.path.insert(0, config.RECON_DIR)
import gen_STORM_transfer as gst  # noqa: E402

COMMON = dict(
    filename=DATA_FILE,
    nFramesDesired=800,
    plot_every=0,                 # keep 0 for timing comparisons
    detect_anomaly=False,
    stop_after_lr_reductions=2,
    stop_training=0,
    stop_min_delta=1e-3,
    restore_best=True,
    bestckpt_start_epoch=5,
    lr_scheduler_threshold=LR_SCHEDULER_THRESHOLD,
)
CAPS = dict(WARMUP_EPOCHS=10, FINAL_EPOCHS=CAP, TRANSFER_EPOCHS=CAP)

if EXP == 1:
    gst.configure(MODE='baseline', SLICES=SLICES,
                  SOURCE_SLICE=SOURCE_SLICE, FREEZE_SPEC='all', MAKE_GIF=False,
                  EXTERNAL_SOURCE_CKPT=None,
                  params=COMMON, **CAPS)
    gst.RUN_NAME = 'E1_scratch'   # folder tag the analysis scripts look for
else:
    gst.configure(MODE='transfer', TRANSFER_MODE='chain',
                  SLICES=SLICES, SOURCE_SLICE=SOURCE_SLICE,
                  FREEZE_SPEC='all', Z_INIT_TARGET='source',
                  EXTERNAL_SOURCE_CKPT=None, CHAIN_HANDOFF=CHAIN_HANDOFF,
                  MAKE_GIF=False, params=COMMON, **CAPS)
    gst.RUN_NAME = 'E2_chain'

print(f'SUBJECT = {config.SUBJECT}   file = {DATA_FILE}', flush=True)
print(f'EXP = {EXP}   RUN_NAME = {gst.RUN_NAME}   '
      f'lr_scheduler_threshold = {LR_SCHEDULER_THRESHOLD}   '
      + (f'CHAIN_HANDOFF = {CHAIN_HANDOFF}' if EXP == 2 else ''), flush=True)
gst.run()
