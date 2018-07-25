# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvRes


class SpectrogramModality(nn.Module):
    """Modality for spectrogram-like input."""
    def __init__(self, N, kernel):
        super(SpectrogramModality, self).__init__()
        self.layers = nn.ModuleList(
            [ConvRes(2**l, 2**(l+1), kernel) for l in range(N)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = torch.squeeze(x, 2)
        return x
