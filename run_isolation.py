#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  run_isolation.py  --  does the speed-up come from the weights or the latent?
#
#      python run_isolation.py          (run AFTER run_pipeline.py)
#
#  On slice 7, warm-started from slice 6, trains two extra conditions under
#  the same stopping protocol as run_experiment.py:
#      B  weights from slice 6 + fresh latent    -> isolates the WEIGHTS
#      C  fresh weights + latent from slice 6    -> isolates the LATENT
#
#  The other two arms of the comparison already exist after run_pipeline.py:
#      baseline        = EXP 1 (E1_scratch), slice 7
#      A weights+latent = EXP 2 (E2_chain),  slice 7
#  so they are not re-run here.
#
#  The slice-6 source model is not retrained either: gen_STORM_transfer's
#  get_source() finds the slice-6 checkpoint that EXP 2 cached next to the
#  data file and reuses it.
# ===============================================================
import sys

import config

ISOLATE_SLICE = 7
SOURCE_SLICE = 6
CAP = 700
LR_SCHEDULER_THRESHOLD = 0.001   # same protocol as run_experiment.py

DATA_FILE = config.resolve_data_file(slice_for_cache=ISOLATE_SLICE)

if config.RECON_DIR not in sys.path:
    sys.path.insert(0, config.RECON_DIR)
import gen_STORM_transfer as gst  # noqa: E402

COMMON = dict(
    filename=DATA_FILE,
    nFramesDesired=800,
    plot_every=0,
    detect_anomaly=False,
    stop_after_lr_reductions=2,
    stop_training=0,
    stop_min_delta=1e-3,
    restore_best=True,
    bestckpt_start_epoch=5,
    lr_scheduler_threshold=LR_SCHEDULER_THRESHOLD,
)

# (label, use source weights?, latent init)
ISOLATE_CONDITIONS = [
    ('B  weights only', True,  'ones'),
    ('C  latent only',  False, 'source'),
]

gst.configure(MODE='isolate', ISOLATE_SLICE=ISOLATE_SLICE, SOURCE_SLICE=SOURCE_SLICE,
              ISOLATE_CONDITIONS=ISOLATE_CONDITIONS, TRANSFER_EPOCHS=CAP,
              WARMUP_EPOCHS=10, FINAL_EPOCHS=CAP,   # only used if slice 6 must be retrained
              MAKE_GIF=False, params=COMMON)

print(f'SUBJECT = {config.SUBJECT}   file = {DATA_FILE}', flush=True)
print(f'ISOLATE_SLICE = {ISOLATE_SLICE}   SOURCE_SLICE = {SOURCE_SLICE}   '
      f'lr_scheduler_threshold = {LR_SCHEDULER_THRESHOLD}', flush=True)
print(f'conditions: {[c[0] for c in ISOLATE_CONDITIONS]} '
      f'(A = the existing EXP 2 chain run)', flush=True)
gst.run()
