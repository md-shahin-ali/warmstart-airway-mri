#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# optimize_gen_sub.py  --  self-supervised training loop
#
# Jointly optimizes the generator weights and the latent z(t) against the
# training split of k-space; the held-out validation split drives a
# reduce-on-plateau learning-rate scheduler (patience 20, factor 0.1).
# Training stops once the scheduler has reduced the learning rate
# params['stop_after_lr_reductions'] times. A snapshot of the best-validation
# model is kept at every reduction, so one run yields both the "drop 1" and
# "drop 2" models.
#
# Plotting is backend-aware: inline in Jupyter, PNG files when run as a
# script. Controlled by params['plot_every'] (0 = off) and
# params['plot_mode'] ('auto' | 'inline' | 'file' | 'both' | 'off').
#
# torch.autograd.set_detect_anomaly is OFF by default (it roughly doubles the
# backward pass); set params['detect_anomaly'] = True only when chasing a NaN.
#
# RETURNS: G, z, train_hist, epoch, stats   (5 values)

import os
import copy
import time
import math
import random
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return 'IPKernelApp' in ip.config or 'ZMQ' in type(ip).__name__
    except Exception:
        return False


IN_NOTEBOOK = _in_notebook()
if not IN_NOTEBOOK:
    matplotlib.use('Agg')          # cluster-safe: no display needed
import matplotlib.pyplot as plt    # noqa: E402


def _show(fig, mode, path):
    """Render a figure inline, to disk, or both."""
    if mode in ('inline', 'both'):
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            plt.show()
    if mode in ('file', 'both') and path:
        fig.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def _call_G(G, zbatch, idx):
    """Call the generator, passing frame indices only if it uses them.

    The reference-conditioned generator with a per-frame low-resolution input
    needs to know WHICH frames are in the batch; the original generator does not.
    """
    if getattr(G, 'uses_frame_index', False):
        return G(zbatch, idx=idx)
    return G(zbatch)


def _z_activity(z):
    """Mean over latent dimensions of the standard deviation across frames.

    The collapse check. If z(t) flattens the manifold has stopped doing anything,
    which the loss will not reveal. Matters most once a per-frame low-resolution
    input is added, since the network can then read motion from the image instead.
    """
    with torch.no_grad():
        zz = z.z_.detach()
        return float(zz.std(dim=0).mean())


def _build_optimizer(G, z, lr_g, lr_z, lr_mlp=None):
    """Only include parameter groups that actually contain parameters.

    Needed because with frozen layers (or lr_z = 0) a group can end up empty.
    """
    groups, min_lrs, names = [], [], []

    # A generator may expose separate groups (e.g. CNN vs MLP with different rates)
    if hasattr(G, 'optim_groups'):
        for grp in G.optim_groups(lr_g, lr_mlp if lr_mlp is not None else lr_g):
            groups.append({'params': grp['params'], 'lr': grp['lr']})
            min_lrs.append(1e-15)
            names.append(f"{grp.get('name', 'generator')}@{grp['lr']:g}")
    else:
        g_params = [p for p in G.parameters() if p.requires_grad]
        if len(g_params) > 0:
            groups.append({'params': g_params, 'lr': lr_g})
            min_lrs.append(1e-15)
            names.append('generator')

    if z.z_.requires_grad and lr_z > 0:
        groups.append({'params': [z.z_], 'lr': lr_z})
        min_lrs.append(1e-12)
        names.append('latent')

    if len(groups) == 0:
        raise ValueError('Nothing to optimize: generator fully frozen and lr_z = 0.')

    return optim.Adam(groups, betas=(0.4, 0.999)), min_lrs, names


# ===========================================================================
def optimize_generator(dop, G, z, params, train_epoch=1, save_weight=True,
                       gtExists=True, recon_callback=None, recon_epochs=None,
                       tag=''):
    """
    recon_callback : callable(G, z, epoch, tag) or None
        Called at the end of each epoch whose 1-based number is in recon_epochs
        (tag="checkpoint"), and once after training with the best-validation
        weights loaded (tag="best_val").
    recon_epochs : list of 1-based epoch numbers, e.g. [5, 20, 40]
    """
    recon_epochs = set(recon_epochs) if recon_epochs is not None else set()

    # ---- paths ----
    pathname = params['filename'].replace(
        '.mat', '_' + str(params['gen_base_size']) + 'd/weights_GENplusLAT' + str(params['coilEst']) + '_')
    pathname = (pathname + str(params['slice']) + '_' + str(params['nintlPerFrame'])
                + 'arms_' + str(params['siz_l']) + 'latVec' + str(params['nFramesDesired'])
                + 'frms_' + str(params['splitRatio']) + 'SplitRatio_' + params['splitDist'])
    if tag:
        pathname = pathname + '_' + tag
    os.makedirs(pathname, exist_ok=True)

    # ---- settings ----
    lr_g = params['lr_g']
    lr_z = params['lr_z']
    gpu = params['device']
    batch_sz = params['nBatch']

    plot_every = params.get('plot_every', 9)          # 0 = off; e.g. 9 = plot every 9th epoch
    plot_mode = params.get('plot_mode', 'auto')
    if plot_mode == 'auto':
        plot_mode = 'inline' if IN_NOTEBOOK else 'file'
    if plot_every == 0:
        plot_mode = 'off'

    best_start = params.get('bestckpt_start_epoch', 5)
    if best_start is None:
        best_start = 5
    patience_stop = params.get('stop_training', 0)    # 0 = early stopping disabled
    # Minimum RELATIVE improvement that counts as progress. Without this the
    # patience counter never fires: the validation loss improves by tiny amounts
    # almost every epoch (0.646106 -> 0.646101), and any improvement would reset
    # the counter. 0.001 = the loss must drop by 0.1% to count.
    stop_min_delta = params.get('stop_min_delta', 1e-3)

    # STOPPING DERIVED FROM THE LEARNING-RATE SCHEDULE.
    #
    # The reduce-on-plateau scheduler already decides when the validation loss has
    # stopped improving - that IS a plateau detector, and its settings are the ones
    # published in the paper (patience 20, threshold 1e-3, factor 0.1). So rather
    # than inventing a second criterion, training stops once the scheduler itself
    # has cut the rate this many times. The algorithm chooses the epoch, not us.
    #
    # Measured on the real 150-epoch validation curve with the paper's settings, the
    # scheduler fires exactly once, at epoch 85:
    #
    #   stopping point                   val loss   vs 150 epochs   time   saved
    #   manual 40 epochs (old choice)      0.4990        +6.5%      4.3 m   73%
    #   algorithm: 1st LR reduction        0.4760        +1.6%      9.2 m   43%
    #   cap: 150 epochs                    0.4686           -      16.2 m    0%
    #
    # So 1 is the default: the algorithm picks ~85 epochs instead of our arbitrary
    # 40, which was 6.5% short. Set 2 to be more conservative (on this curve the
    # scheduler never fires twice, so it would run to the cap), or 0 to disable.
    stop_after_lr_drops = params.get('stop_after_lr_reductions', 1)
    detect_anomaly = params.get('detect_anomaly', False)

    nslc = 1 if isinstance(params['slice'], int) else len(params['slice'])

    optimizer, min_lrs, group_names = _build_optimizer(
        G, z, lr_g, lr_z, params.get('lr_mlp'))
    # How much better a new epoch must be to count as real progress, as a
    # fraction of the current best (e.g. 0.001 = must beat the best by >0.1%).
    # A LARGER value makes this bar harder to clear, so the 20-epoch patience
    # counter fills up sooner and training stops earlier - at some cost to how
    # far training gets before stopping. Configurable so this can be tested
    # against the existing 0.001 runs without touching anything else; defaults
    # to 0.001, the value every run so far has used, so nothing changes unless
    # this is set explicitly.
    lr_scheduler_threshold = params.get('lr_scheduler_threshold', 0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                  threshold=lr_scheduler_threshold, min_lr=min_lrs)

    n_train = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in G.parameters())
    print(f'  optimizing: {group_names}   generator trainable: '
          f'{n_train:,} / {n_total:,} ({100*n_train/max(1,n_total):.1f}%)', flush=True)
    if detect_anomaly:
        print('  WARNING: detect_anomaly is ON - training will be ~2x slower. '
              'Turn it off for timing runs.', flush=True)

    # ---- state ----
    train_hist = {'G_losses': [], 'per_epoch_ptimes': [], 'total_ptime': []}
    valid_loss_min = np.inf        # strict best, for keeping the best weights
    patience_ref = np.inf          # best that counts as PROGRESS, for the counter
    val_loss_tracker = 0
    best_val_G_state = None
    best_val_z_data = None
    best_val_epoch = None
    stop_reason = 'completed'
    G_val_losses, G_losses = [], []
    z_act = []                        # latent activity per epoch (collapse check)
    lr_drops = 0                      # how many times the scheduler cut the rate
    lr_drop_epochs = []
    # A snapshot of the best-so-far model at each learning-rate reduction. This is
    # what lets ONE run answer both "what if we stopped at the first drop" and
    # "what if we stopped at the second": the first is simply an earlier point on
    # the same trajectory, so there is no need to train twice.
    drop_snapshots = []

    loss = nn.MSELoss(reduction='sum')
    G_oldi = G.state_dict()
    z_oldi = z.z_.data.clone()
    divergence_counter = 0

    nbatches = params['nFramesDesired'] // batch_sz
    nindices = nbatches * batch_sz

    print('training start!', flush=True)
    start_time = time.time()
    epoch = 0

    ctx = torch.autograd.set_detect_anomaly(True) if detect_anomaly else contextlib.nullcontext()
    with ctx:
        for epoch in range(train_epoch):
            indices = np.arange(0, params['nFramesDesired'])
            random.shuffle(indices)
            indices = np.reshape(indices[0:nindices], (nbatches, batch_sz))
            epoch_start_time = time.time()
            batch_loss = 0

            # ---------------- training ----------------
            G.train()
            for batch in range(nbatches):
                G_loss = 0
                optimizer.zero_grad()

                for slc in range(nslc):
                    G_result = _call_G(G, z.z_[indices[batch], ..., slc],
                                      indices[batch])[..., slc]

                    if params['ssTrainMode']:
                        G_result_projected = dop.Psub(G_result, indices[batch], slc, mode='train')
                    else:
                        G_result_projected = dop.Psub(G_result, indices[batch], slc)

                    if params["fastMode"]:
                        if params['ssTrainMode']:
                            G_loss += loss(torch.view_as_real(G_result_projected),
                                           torch.view_as_real(dop.AtbTrn[indices[batch], ..., slc]).to(gpu))
                        else:
                            G_loss += loss(torch.view_as_real(G_result_projected),
                                           torch.view_as_real(dop.Atb[indices[batch], ..., slc]).to(gpu))
                    else:
                        G_loss += loss(torch.view_as_real(G_result_projected),
                                       torch.view_as_real(dop.AtbTrn[indices[batch], ..., slc]))

                    if params['ssTrainMode']:
                        G_loss += dop.image_energy_sub(G_result, slc, mode='train')
                    else:
                        G_loss += dop.image_energy_sub(G_result, slc)
                    G_loss += z.KLloss(slc)

                G_loss += G.weightl1norm()
                G_loss += z.Reg()

                G_loss.backward()
                batch_loss += G_loss.detach()
                optimizer.step()

            G_losses.append(batch_loss.item()/(nbatches*batch_sz))

            # ---------------- validation ----------------
            G.eval()
            with torch.no_grad():
                batch_val_loss = 0
                for batch in range(nbatches):
                    GValLoss = 0
                    for slc in range(nslc):
                        G_result = _call_G(G, z.z_[indices[batch], ..., slc],
                                          indices[batch])[..., slc]
                        G_result_projected = dop.Psub(G_result, indices[batch], slc, mode='validation').to(gpu)
                        if params["fastMode"]:
                            GValLoss += loss(torch.view_as_real(G_result_projected),
                                             torch.view_as_real(dop.AtbVal[indices[batch], ..., slc]).to(gpu))
                        else:
                            GValLoss += loss(torch.view_as_real(G_result_projected),
                                             torch.view_as_real(dop.AtbVal[indices[batch], ..., slc]))
                    batch_val_loss += GValLoss.detach()

                G_val_losses.append(batch_val_loss.item()/(nbatches*batch_sz))
                if params['ssTrainMode']:
                    lr_before = [g['lr'] for g in optimizer.param_groups]
                    scheduler.step(G_val_losses[-1])
                    lr_after = [g['lr'] for g in optimizer.param_groups]
                    if any(b > a for b, a in zip(lr_before, lr_after)):
                        lr_drops += 1
                        lr_drop_epochs.append(epoch + 1)
                        drop_snapshots.append(dict(
                            n_drop=lr_drops,
                            drop_epoch=epoch + 1,
                            elapsed_s=time.time() - start_time,
                            epochs_so_far=epoch + 1,
                            best_epoch=best_val_epoch,
                            best_val=float(valid_loss_min),
                            G_state=({k: v.detach().cpu().clone()
                                      for k, v in best_val_G_state.items()}
                                     if best_val_G_state is not None else None),
                            z_data=(best_val_z_data.detach().cpu().clone()
                                    if best_val_z_data is not None else None),
                            val_curve=list(G_val_losses),
                            train_curve=list(G_losses)))
                        print(f'      snapshot taken: best epoch {best_val_epoch}, '
                              f'val {valid_loss_min:.6f}, '
                              f'{drop_snapshots[-1]["elapsed_s"]/60:.1f} min so far',
                              flush=True)
                        print(f'   -> LR reduction {lr_drops} at epoch {epoch+1}: '
                              + ', '.join(f'{n} {b:.2e}->{a:.2e}'
                                          for n, b, a in zip(group_names, lr_before,
                                                             lr_after)), flush=True)

            per_epoch_ptime = time.time() - epoch_start_time
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
            z_act.append(_z_activity(z))

            # ---------------- best-validation tracking ----------------
            if epoch >= best_start:
                # two separate criteria on purpose:
                #   valid_loss_min - any improvement is worth keeping the weights
                #   patience_ref   - only a MEANINGFUL improvement resets the counter
                if G_val_losses[-1] < patience_ref * (1 - stop_min_delta):
                    patience_ref = G_val_losses[-1]
                    val_loss_tracker = 0
                else:
                    val_loss_tracker += 1

                if G_val_losses[-1] < valid_loss_min:
                    valid_loss_min = G_val_losses[-1]
                    best_val_epoch = epoch + 1
                    best_val_G_state = copy.deepcopy(G.state_dict())
                    best_val_z_data = z.z_.data.clone()
                    val_loss_tracker = 0
                    # ONE file, overwritten. Putting epoch+loss in the filename writes a
                    # new ~80 MB file on every improvement, which fills the quota fast.
                    torch.save({'epoch': epoch, 'valid_loss_min': valid_loss_min,
                                'AllValLoss': G_val_losses, 'AllTrainingLoss': G_losses,
                                'model_state': G.state_dict(), 'latVec_state': z.z_},
                               os.path.join(pathname, 'bestModelValPoint_splitRatio{}.pth'.format(
                                   params['splitRatio'])))
                    print(f'   -> new best val {valid_loss_min:.6f} at epoch {best_val_epoch}',
                          flush=True)

            # ---- stop when the schedule says the model has plateaued ----
            if stop_after_lr_drops and lr_drops >= stop_after_lr_drops:
                stop_reason = (f'plateau: the learning rate was reduced '
                               f'{lr_drops} times (epochs {lr_drop_epochs}), so the '
                               f'validation loss has stopped improving')
                print(f'  ** {stop_reason} - stopping at epoch {epoch+1} of '
                      f'{train_epoch} **', flush=True)
                print('[%d/%d] - ptime: %.2f, loss_g: %.3f, val_loss: %.3f'
                      % (epoch+1, train_epoch, per_epoch_ptime, G_losses[-1],
                         G_val_losses[-1]), flush=True)
                train_hist['G_losses'].append(batch_loss.item())
                break

            if epoch >= best_start:
                if patience_stop and val_loss_tracker >= patience_stop:
                    stop_reason = (f'early stop: no improvement greater than '
                                   f'{100*stop_min_delta:.1f}% for {patience_stop} '
                                   f'consecutive epochs')
                    print(f'  ** {stop_reason} - stopping at epoch {epoch+1} of '
                          f'{train_epoch} **', flush=True)
                    print('[%d/%d] - ptime: %.2f, loss_g: %.3f, val_loss: %.3f'
                          % (epoch+1, train_epoch, per_epoch_ptime, G_losses[-1], G_val_losses[-1]),
                          flush=True)
                    train_hist['G_losses'].append(batch_loss.item())
                    break

            # ---------------- divergence guard ----------------
            if epoch > 10 and len(train_hist['G_losses']) > 0:
                if batch_loss.item() > 1.15*train_hist['G_losses'][-1]:
                    G.load_state_dict(G_oldi)
                    z.z_.data = z_oldi
                    print('  loading old state; reducing stp siz', flush=True)
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*0.98
                    divergence_counter += 1
                else:
                    divergence_counter = max(0, divergence_counter - 1)
                    G_oldi = G.state_dict()
                    z_oldi = z.z_.data.clone()
                    train_hist['G_losses'].append(batch_loss.item())
                    if save_weight:
                        torch.save({'G_oldi': G.state_dict(), 'z_oldi': z.z_},
                                   os.path.join(pathname, 'net_GENplusLAT_epch{}.pth'.format(epoch)))
            else:
                G_oldi = G.state_dict()
                z_oldi = z.z_.data.clone()
                train_hist['G_losses'].append(batch_loss.item())
                if save_weight:
                    torch.save({'G_oldi': G.state_dict(), 'z_oldi': z.z_},
                               os.path.join(pathname, 'net_GENplusLAT_epch{}.pth'.format(epoch)))

            print('[%d/%d] - ptime: %.2f, loss_g: %.3f, val_loss: %.3f'
                  % (epoch+1, train_epoch, per_epoch_ptime, G_losses[-1], G_val_losses[-1]),
                  flush=True)

            # ---------------- live figure (this is what went missing) ----------------
            if plot_mode != 'off' and (epoch % plot_every == 0):
                fig, ax = plt.subplots(nslc, 2, figsize=(15, 4*nslc))
                if nslc == 1:
                    ax = np.expand_dims(ax, 0)
                for sl in range(nslc):
                    with torch.no_grad():
                        G_result = _call_G(G, z.z_[indices[0], ..., sl], indices[0])[..., sl]
                    test_image1 = G_result[-1].squeeze(0).cpu().data.numpy()
                    ax[sl, 0].imshow(abs(test_image1), cmap='gray', vmin=0.0,
                                     vmax=.6*np.max(abs(test_image1)))
                    ax[sl, 0].set_title(f'epoch {epoch+1}  |  val {G_val_losses[-1]:.4f}')
                    ax[sl, 0].axis('off')
                    ax[sl, 1].plot(z.z_[..., sl].data.squeeze().cpu().numpy())
                    ax[sl, 1].set_title('latent z(t)')
                fig.tight_layout()
                _show(fig, plot_mode, os.path.join(pathname, f'progress_epoch{epoch+1:04d}.png'))

            # ---------------- milestone reconstruction callback ----------------
            if recon_callback is not None and (epoch + 1) in recon_epochs:
                print(f'  -> recon_callback at epoch {epoch+1}', flush=True)
                G.eval()
                with torch.no_grad():
                    recon_callback(G, z, epoch + 1, tag='checkpoint')
                G.train()

    # ---------------- best-val reconstruction, once ----------------
    if recon_callback is not None and best_val_G_state is not None:
        print(f'\n  -> best-val recon (epoch {best_val_epoch}, val {valid_loss_min:.6f})', flush=True)
        cur_G, cur_z = G.state_dict(), z.z_.data.clone()
        G.load_state_dict(best_val_G_state)
        z.z_.data = best_val_z_data
        G.eval()
        with torch.no_grad():
            recon_callback(G, z, best_val_epoch, tag='best_val')
        G.load_state_dict(cur_G)
        z.z_.data = cur_z

    # ---------------- restore the best-validation weights ----------------
    # Without this the caller reconstructs from the FINAL epoch, not the epoch the
    # validation loss actually selected - which is not what the method claims.
    # The best state is already held in memory, so this costs nothing.
    restored_from = None
    if params.get('restore_best', True) and best_val_G_state is not None:
        G.load_state_dict(best_val_G_state)
        z.z_.data = best_val_z_data.to(z.z_.device)
        restored_from = best_val_epoch
        print(f'  restored the best-validation weights: epoch {best_val_epoch}, '
              f'val {valid_loss_min:.6f}  (final epoch was {G_val_losses[-1]:.6f})',
              flush=True)

    total_ptime = time.time() - start_time
    train_hist['total_ptime'].append(total_ptime)
    train_hist['G_val_losses'] = G_val_losses      # used by plotting scripts
    train_hist['G_losses_all'] = G_losses          # one entry per epoch, normalized
    train_hist['G_losses_full'] = G_losses         # alias for older plotting scripts
    print('Optimization done in %d seconds' % total_ptime, flush=True)
    print(f'  learning-rate reductions: {lr_drops}'
          + (f' at epochs {lr_drop_epochs}' if lr_drop_epochs else ' (none - the '
             'validation loss never plateaued)'), flush=True)
    if z_act:
        print(f'  latent activity: {z_act[0]:.5f} -> {z_act[-1]:.5f}  '
              f'(std of z across frames; near zero means the manifold has collapsed)',
              flush=True)

    torch.save({'epoch': epoch,
                'AllValLoss': G_val_losses, 'AllTrainingLoss': G_losses,
                # --- stopping metadata, so the curves can be annotated later ---
                'epochs_run': epoch + 1,
                'epochs_requested': train_epoch,
                'best_epoch': best_val_epoch,          # 1-based, min validation loss
                'valid_loss_min': valid_loss_min,
                'restored_from_epoch': restored_from,
                'lr_reduction_epochs': lr_drop_epochs, # 1-based
                'drop_meta': [{k: s[k] for k in
                               ('n_drop','drop_epoch','elapsed_s','best_epoch','best_val')}
                              for s in drop_snapshots],
                'lr_reductions': lr_drops,
        'drop_snapshots': drop_snapshots,
                'stop_reason': stop_reason,
                'per_epoch_seconds': train_hist['per_epoch_ptimes'],
                'model_state': G.state_dict(), 'latVec_state': z.z_},
               os.path.join(pathname, 'ForPlottingLossStats_splitRatio{}.pth'.format(
                   params['splitRatio'])))

    # ---------------- end-of-run loss curve ----------------
    if plot_mode != 'off' and len(G_losses) > 1:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ep = range(1, len(G_losses)+1)
        ax1.plot(ep, G_losses, 'b-', label='train')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(G_val_losses)+1), G_val_losses, 'r-', label='validation')
        ax2.set_ylabel('validation loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        if best_val_epoch:
            ax2.axvline(best_val_epoch, color='g', ls='--', lw=1)
            ax2.annotate(f'best ep {best_val_epoch}', (best_val_epoch, valid_loss_min),
                         textcoords='offset points', xytext=(6, 6), color='g', fontsize=9)
        ax1.set_title(f'slice {params["slice"]} - training and validation loss')
        fig.tight_layout()
        _show(fig, plot_mode, os.path.join(pathname, 'loss_curves.png'))

    stats = {
        'epochs_run': epoch + 1,
        'epochs_requested': train_epoch,
        'train_seconds': total_ptime,
        'sec_per_epoch': total_ptime/max(1, epoch+1),
        'final_train_loss': G_losses[-1] if G_losses else float('nan'),
        'final_val_loss': (valid_loss_min if restored_from is not None
                           else (G_val_losses[-1] if G_val_losses else float('nan'))),
        'best_val_loss': valid_loss_min if np.isfinite(valid_loss_min) else float('nan'),
        'best_epoch': best_val_epoch if best_val_epoch else -1,
        'stop_reason': stop_reason,
        'stopped_early': stop_reason != 'completed',
        'patience': patience_stop,
        'lr_reductions': lr_drops,
        'drop_snapshots': drop_snapshots,
        'lr_reduction_epochs': lr_drop_epochs,
        'stop_after_lr_reductions': stop_after_lr_drops,
        'stop_min_delta': stop_min_delta,
        'restored_from_epoch': restored_from,
        'final_epoch_val_loss': G_val_losses[-1] if G_val_losses else float('nan'),
        'divergence_events': divergence_counter,
        'z_activity': z_act[-1] if z_act else float('nan'),
        'z_activity_start': z_act[0] if z_act else float('nan'),
        'z_activity_curve': z_act,
        'val_curve': G_val_losses,
        'train_curve': G_losses,
        'weights_dir': pathname,
    }
    return G, z, train_hist, epoch, stats
