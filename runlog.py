#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# runlog.py  --  per-slice timing, printed to stdout and written to CSV

import os
import csv
import time
from datetime import timedelta


def hms(seconds):
    """1234.5 -> '0:20:34'"""
    return str(timedelta(seconds=int(round(seconds))))


class Stopwatch:
    """with Stopwatch() as sw: ...   then sw.elapsed"""
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
        return False


class RunLog:
    """Collects one record per slice and prints a summary table at the end."""

    FIELDS = ['slice', 'mode', 'source_slice', 'n_train_convs', 'epochs_run',
              'load_s', 'train_s', 'recon_s', 'save_s', 'total_s',
              'sec_per_epoch', 'final_val_loss', 'best_val_loss', 'best_epoch',
              'stop_reason']

    def __init__(self, out_dir, run_name='run'):
        self.rows = []
        self.t_start = time.time()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, f'{run_name}_timing.csv')

    def add(self, **kw):
        row = {k: kw.get(k, '') for k in self.FIELDS}
        self.rows.append(row)
        self._flush()
        self._print_slice(row)

    def _flush(self):
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    def _print_slice(self, r):
        print('\n' + '-'*78)
        print(f'  TIMING - slice {r["slice"]}  [{r["mode"]}]')
        print('-'*78)
        print(f'    data load        : {hms(r["load_s"]):>10}')
        print(f'    training         : {hms(r["train_s"]):>10}   '
              f'({r["epochs_run"]} epochs, {float(r["sec_per_epoch"]):.1f} s/epoch)')
        print(f'    frame generation : {hms(r["recon_s"]):>10}')
        print(f'    saving + gif     : {hms(r["save_s"]):>10}')
        print(f'    TOTAL THIS SLICE : {hms(r["total_s"]):>10}')
        print('-'*78 + '\n')

    def summary(self):
        wall = time.time() - self.t_start
        print('\n' + '='*100)
        print('  RUN SUMMARY - TIME PER SLICE')
        print('='*100)
        hdr = (f'{"slice":>6} {"mode":>10} {"src":>5} {"convs":>6} {"epochs":>7} '
               f'{"load":>9} {"train":>10} {"recon":>9} {"save":>9} {"TOTAL":>10} '
               f'{"s/epoch":>9} {"val loss":>12}')
        print(hdr)
        print('-'*100)
        tot_load = tot_train = tot_recon = tot_save = tot_total = 0.0
        for r in self.rows:
            tot_load += float(r['load_s'] or 0)
            tot_train += float(r['train_s'] or 0)
            tot_recon += float(r['recon_s'] or 0)
            tot_save += float(r['save_s'] or 0)
            tot_total += float(r['total_s'] or 0)
            vl = r['final_val_loss']
            vl = f'{float(vl):.4f}' if vl not in ('', None) else '-'
            print(f'{str(r["slice"]):>6} {str(r["mode"]):>10} {str(r["source_slice"]):>5} '
                  f'{str(r["n_train_convs"]):>6} {str(r["epochs_run"]):>7} '
                  f'{hms(float(r["load_s"] or 0)):>9} {hms(float(r["train_s"] or 0)):>10} '
                  f'{hms(float(r["recon_s"] or 0)):>9} {hms(float(r["save_s"] or 0)):>9} '
                  f'{hms(float(r["total_s"] or 0)):>10} '
                  f'{float(r["sec_per_epoch"] or 0):>9.1f} {vl:>12}')
        print('-'*100)
        n = max(1, len(self.rows))
        print(f'{"TOTALS":>6} {"":>10} {"":>5} {"":>6} {"":>7} '
              f'{hms(tot_load):>9} {hms(tot_train):>10} {hms(tot_recon):>9} '
              f'{hms(tot_save):>9} {hms(tot_total):>10}')
        print(f'\n  slices processed     : {len(self.rows)}')
        print(f'  mean time per slice  : {hms(tot_total/n)}')
        print(f'  sum of slice times   : {hms(tot_total)}')
        print(f'  wall clock (script)  : {hms(wall)}')
        print(f'  timing csv           : {self.csv_path}')
        print('='*100 + '\n')
        return {'total_s': tot_total, 'wall_s': wall, 'n_slices': len(self.rows)}
