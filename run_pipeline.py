#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  run_pipeline.py  --  EXP 1 (baseline) then EXP 2 (warm-start chain)
#
#      python run_pipeline.py
#
#  Runs run_experiment.py 1 and run_experiment.py 2 back to back on one GPU
#  for the subject in config.py, then prints a timing summary.
#
#  Each experiment runs in its own fresh Python process. gen_STORM_transfer's
#  configure() only overwrites the settings it is given and keeps every other
#  global as it was, so a fresh process guarantees EXP 2 cannot inherit a
#  stale setting from EXP 1.
#
#  Everything the experiments print streams straight through, unbuffered, so
#  redirecting this script's output to a file captures the complete log:
#      python -u run_pipeline.py | tee pipeline.log
# ===============================================================
import os
import sys
import subprocess
import time
from datetime import datetime, timedelta

import config

SCRIPT = os.path.join(config.REPO_DIR, 'run_experiment.py')


def fmt(seconds):
    return str(timedelta(seconds=round(seconds)))


def run_stage(exp, label):
    print('=' * 78)
    print(f'  {config.SUBJECT}  EXP {exp}  ({label})  -  starting at '
          f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 78, flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, '-u', SCRIPT, str(exp)], env=env)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print('-' * 78)
    print(f'  {config.SUBJECT}  EXP {exp}  ({label})  -  '
          f'{"FINISHED" if ok else "FAILED (returncode " + str(result.returncode) + ")"}'
          f'  in {fmt(elapsed)}')
    print('-' * 78 + '\n', flush=True)
    return ok, elapsed


def main():
    config.resolve_data_file()            # fail fast if the path is not set
    print(f'\nSUBJECT = {config.SUBJECT}   running EXP 1 then EXP 2, back to back.\n')
    t_start = time.time()

    ok1, t1 = run_stage(1, 'baseline, all 11 slices from scratch')
    if not ok1:
        print(f'EXP 1 failed - stopping before EXP 2. Total elapsed: {fmt(time.time() - t_start)}')
        raise SystemExit(1)

    ok2, t2 = run_stage(2, 'warm-start chain from slice 6')
    total = time.time() - t_start

    print('=' * 78)
    print(f'  {config.SUBJECT}  SUMMARY')
    print('=' * 78)
    print(f'  EXP 1 (baseline, all 11 slices) : {fmt(t1)}')
    print(f'  EXP 2 (chain, all 11 slices)    : {fmt(t2)}')
    print(f'  TOTAL (both experiments)        : {fmt(total)}')
    print('=' * 78)
    if not ok2:
        print('  NOTE: EXP 2 reported a failure (returncode nonzero) - check its log above.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
