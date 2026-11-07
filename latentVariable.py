#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# latentVariable.py  --  per-frame latent trajectory z(t)
#
# Holds one latent vector per frame, with temporal-smoothness and KL
# regularizers. Initialization options:
#   'ones'    : constant per dimension (used for training from scratch)
#   'random'  : small Gaussian
#   'zeros'
#   'source'  : start from a latent trajectory fitted on another slice
#               (z_source). All slices are interleaved within the same TR and
#               see the same breathing, so a neighbour's z(t) is a physically
#               motivated starting point.

import torch
import numpy as np
import math
from torch.autograd import Variable


class latentVariableNew():
    def __init__(self, params, z_in=1, init='interpolate', dataPath=None,
                 alpha=[0, 0], klreg=0, z_source=None):

        self.gpu = torch.device(params['device'])
        self.klreg = klreg
        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha = torch.FloatTensor(np.asarray(alpha, dtype=np.float32)).to(self.gpu)
        self.alpha = alpha

        if isinstance(params['slice'], int):
            self.nsl = 1
        else:
            self.nsl = len(params['slice'])

        device = params['device']
        nFrames = params['nFramesDesired']
        siz_l = params['siz_l']

        if init == 'ones':
            zpnew = 0.5*np.ones((nFrames, siz_l, 1, 1, self.nsl))
            for i in range(siz_l):
                zpnew[:, i] *= np.random.rand()

        elif init == 'random':
            zpnew = 0.15*np.random.randn(nFrames, siz_l, 1, 1, self.nsl)

        elif init == 'zeros':
            zpnew = np.zeros((nFrames, siz_l, 1, 1, self.nsl))

        elif init == 'source':
            # ---- NEW: start from a latent trajectory fitted on another slice ----
            if z_source is None:
                raise ValueError("init='source' requires z_source=<tensor or array>")
            if torch.is_tensor(z_source):
                zsrc = z_source.detach().cpu().numpy()
            else:
                zsrc = np.asarray(z_source)
            zsrc = np.reshape(zsrc, (zsrc.shape[0], zsrc.shape[1], 1, 1, -1))

            zpnew = np.zeros((nFrames, siz_l, 1, 1, self.nsl))
            nf = min(nFrames, zsrc.shape[0])
            nl = min(siz_l, zsrc.shape[1])
            for s in range(self.nsl):
                ssrc = min(s, zsrc.shape[-1]-1)
                zpnew[:nf, :nl, 0, 0, s] = zsrc[:nf, :nl, 0, 0, ssrc]
            if nFrames > nf:   # pad by repeating the last frame
                zpnew[nf:] = zpnew[nf-1]

        else:
            zinput = z_in.z_.data.cpu().numpy()
            nin = np.size(zinput, 0)
            if (nFrames == nin):
                zpnew = zinput
            else:
                x = np.arange(0, nFrames)
                nf = np.size(zinput, 0)
                xp = np.arange(0, nf)*nFrames/nf
                zpnew = np.zeros((nFrames, np.size(zinput, 1), 1, 1, self.nsl))
                for i in range(np.size(zinput, 1)):
                    zpnew[:, i, :] = np.interp(x, xp, zinput[:, i, :])

        z_in_t = torch.tensor(zpnew, dtype=torch.float32)
        z_in_t = z_in_t.cuda(device)
        self.z_ = torch.zeros((nFrames, siz_l, 1, 1, self.nsl), dtype=torch.float32)
        self.z_ = Variable(self.z_.cuda(device), requires_grad=True)
        self.z_.data = z_in_t

    # ------------------------------------------------------------------
    def load_(self, tensor):
        """Overwrite the latent in place, keeping requires_grad=True.

        Safer than `z.z_ = ckpt['z']`, which silently replaces the Variable and
        can drop requires_grad so the optimizer never updates it.
        """
        t = tensor.detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
        self.z_.data = t.to(self.z_.device).to(self.z_.dtype).reshape(self.z_.shape)
        self.z_.requires_grad_(True)
        return self

    # ------------------------------------------------------------------
    def Reg(self):
        zsmoothness = self.z_[1:, ...] - self.z_[:-1, ...]
        zsmoothness = torch.sum(zsmoothness*zsmoothness, axis=(0, -1)).squeeze()
        zsmoothness = torch.sum(self.alpha*zsmoothness, axis=0)
        return (zsmoothness)

    def KLloss(self, slc):
        Nsamples = self.z_.shape[0]
        mn = torch.mean(self.z_[:, :, 0, 0, slc], 0)
        meansub = self.z_[:, :, 0, 0, slc] - mn
        Sigma = meansub.T@meansub/Nsamples

        tr = torch.trace(Sigma)
        if (tr > 0.001):
            loss = 0.5*(mn@mn.T + tr - self.z_.shape[1] - torch.logdet(Sigma))
            if (math.isnan(loss)):
                loss = 0.5*(mn@mn.T)
        else:
            loss = 0.5*(mn@mn.T)
        return (self.klreg*loss)
