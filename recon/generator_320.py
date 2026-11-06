# -*- coding: utf-8 -*-
#
# generator_320.py  --  CNN generator G(z) -> 320x320 complex image
#
# A 1x1 latent code is decoded through 11 conv layers with nearest-neighbour
# upsampling to a 320x320 complex image (real/imag as two channels).
#
# Also provides layer-freeze utilities used by the transfer experiments:
#   conv_index_map()      : positions of the Conv2d layers inside conv_blocks
#   set_trainable_convs() : choose which conv layers train; the rest are frozen
#   trainable_report()    : summary of what is frozen / trainable
#   weightl1norm()        : L1 weight penalty over TRAINABLE weights only

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad


def initialization(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=3e-7)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, std=3e-7)
        elif isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.02)
            print(f'this layer is a : {layer.__class__.__name__}')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            print(f'this layer is a : {layer.__class__.__name__}')
            nn.init.normal_(layer.weight, mean=1.0, std=3e-7)
            nn.init.constant_(layer.bias, 0)


class generatorNew(nn.Module):
    # initializers
    def __init__(self, params, out_channel=2):
        super().__init__()
        factor = params['factor']
        siz_latent = params['siz_l']
        d = params['gen_base_size']
        self.gen_reg = params['gen_reg']

        if isinstance(params['slice'], int):
            self.nsl = 1
        else:
            self.nsl = len(params['slice'])

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(siz_latent, 100, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 1 --> 1x1
            nn.Conv2d(100, d*8, 3, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 2 --> 3x3
            nn.Conv2d(d*8, d*8, 3, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 3 --> 5x5
            nn.Upsample(scale_factor=2, mode='nearest'),    # NN --> 10x10
            nn.Conv2d(d*8, d*8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 4 --> 10x10
            nn.Upsample(scale_factor=2, mode='nearest'),    # NN --> 20x20
            nn.Conv2d(d*8, d*4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 5 --> 20x20
            nn.Upsample(scale_factor=2, mode='nearest'),    # NN --> 40x40
            nn.Conv2d(d*4, d*4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 6 --> 40x40
            nn.Upsample(scale_factor=2, mode='nearest'),    # NN --> 80x80
            nn.Conv2d(d*4, d*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 7 --> 80x80
            nn.Upsample(scale_factor=2, mode='nearest'),    # NN --> 160x160
            nn.Conv2d(d*2, d*2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 8 --> 160x160
            nn.Conv2d(d*2, d, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),    # NN --> 320x320
            nn.Conv2d(d, d, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),                # Finish Layer 9 --> 320x320
            nn.Conv2d(d, out_channel*self.nsl, 3, 1, 1),    # Finish Layer 10 --> 320x320
        )

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.total_params = pytorch_total_params
        print("Created generator with ", pytorch_total_params, " parameters")
        print('It is %.2f times the image size' % ((pytorch_total_params/320/320)))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, input):
        x = self.conv_blocks(input)
        x = x[:, 0:self.nsl] + 1j*x[:, self.nsl:2*self.nsl]
        x = x.permute((0, 2, 3, 1))
        x = x.unsqueeze(1)
        return x

    # ------------------------------------------------------------------
    # NEW: layer freezing utilities for the transfer experiment
    # ------------------------------------------------------------------
    def conv_index_map(self):
        """Positions of the Conv2d layers inside self.conv_blocks.

        Returns a list of (conv_number, module_index, spatial_size_after) so you
        can see exactly where the freeze boundary falls.
        """
        sizes = [1, 3, 5, 10, 20, 40, 80, 160, 160, 320, 320]
        out, k = [], 0
        for i, m in enumerate(self.conv_blocks):
            if isinstance(m, nn.Conv2d):
                out.append((k, i, sizes[k] if k < len(sizes) else None))
                k += 1
        return out

    def n_convs(self):
        return len(self.conv_index_map())

    # named presets, resolved against the 11 conv layers
    PRESETS = {
        'all':        None,                       # nothing frozen (control)
        'first2':     [0, 1],
        'first3':     [0, 1, 2],
        'first4':     [0, 1, 2, 3],
        'first5':     [0, 1, 2, 3, 4],
        'first6':     [0, 1, 2, 3, 4, 5],
        # band freezes: early layers + output layers trainable, middle frozen.
        # These let each slice re-adapt its fine detail instead of inheriting the
        # source slice's noise pattern through frozen 320x320 convs.
        'ends1':      [0, 1, 2, 3, 10],
        'ends2':      [0, 1, 2, 3, 9, 10],
        'ends3':      [0, 1, 2, 3, 8, 9, 10],
        'ends_wide':  [0, 1, 2, 3, 4, 7, 8, 9, 10],
        'out_only':   [9, 10],                    # only the 320x320 output convs
    }

    def set_trainable_convs(self, spec):
        """Choose which conv layers train. Everything else is frozen.

        spec may be:
          None or 'all'      -> train everything
          int n              -> train the FIRST n convs (suffix freeze)
          list/tuple/set     -> train exactly these conv indices, e.g. [0,1,2,3,9,10]
          str preset name    -> one of self.PRESETS

        Returns the sorted list of trainable conv indices (or None for 'all').
        """
        cmap = self.conv_index_map()
        total = len(cmap)

        if isinstance(spec, str):
            if spec not in self.PRESETS:
                raise ValueError(f'Unknown preset {spec!r}. Options: {sorted(self.PRESETS)}')
            spec = self.PRESETS[spec]

        if spec is None or (isinstance(spec, int) and spec >= total):
            for p in self.parameters():
                p.requires_grad_(True)
            self.trainable_set = None
            return None

        if isinstance(spec, int):
            trainable = set(range(max(0, spec)))
        else:
            trainable = set(int(i) for i in spec)
            bad = [i for i in trainable if i < 0 or i >= total]
            if bad:
                raise ValueError(f'conv indices out of range 0..{total-1}: {bad}')

        # module index -> conv number
        module_of_conv = {mi: cn for cn, mi, _ in cmap}
        for i, m in enumerate(self.conv_blocks):
            if i in module_of_conv:
                flag = module_of_conv[i] in trainable
                for p in m.parameters():
                    p.requires_grad_(flag)
            else:
                for p in m.parameters():          # activations/upsample have none
                    p.requires_grad_(True)

        self.trainable_set = sorted(trainable)
        return self.trainable_set

    def trainable_report(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fr = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        tot = tr + fr
        pct = 100.0*tr/tot if tot else 0.0
        cmap = self.conv_index_map()
        ts = getattr(self, 'trainable_set', None)
        if ts is None:
            where = 'all 11 conv layers trainable (nothing frozen)'
        else:
            froz = [c for c in range(len(cmap)) if c not in ts]
            where = (f'trainable convs {ts}\n'
                     f'                     frozen convs    {froz}')
        return (f'  trainable params : {tr:,}  ({pct:.1f}% of {tot:,})\n'
                f'  frozen params    : {fr:,}\n'
                f'  layout           : {where}')

    # ------------------------------------------------------------------
    # regularization  (now restricted to trainable weights)
    # ------------------------------------------------------------------
    def weightl1norm(self):
        L1norm = 0
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                L1norm = L1norm + torch.norm(param, 1)
        return (self.gen_reg*L1norm)
