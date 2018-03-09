# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def ConvTranspose2d(in_channels, out_channels, kernel_size,
                    weight_normalization=True, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
    m.weight.data.fill_(1.0 / freq_axis_kernel_size)
    m.bias.data.zero_()
    if weight_normalization:
        return nn.utils.weight_norm(m)
    else:
        return m


def Conv1d1x1(in_channels, out_channels, bias=True, weight_normalization=True):
    """1-by-1 convolution layer
    """
    if weight_normalization:
        from deepvoice3_pytorch.modules import Conv1d
        assert bias
        return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                      dilation=1, bias=bias, std_mul=1.0)
    else:
        from deepvoice3_pytorch.conv import Conv1d
        return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                      dilation=1, bias=bias)


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None,
                 cin_channels=-1, gin_channels=-1,
                 dropout=1 - 0.95, padding=None, dilation=1, causal=True,
                 bias=True, weight_normalization=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        if weight_normalization:
            from deepvoice3_pytorch.modules import Conv1d
            assert bias
            self.conv = Conv1d(residual_channels, gate_channels, kernel_size,
                               dropout=dropout, padding=padding, dilation=dilation,
                               bias=bias, std_mul=1.0, *args, **kwargs)
        else:
            from deepvoice3_pytorch.conv import Conv1d
            self.conv = Conv1d(residual_channels, gate_channels, kernel_size,
                               padding=padding, dilation=dilation,
                               bias=bias, *args, **kwargs)

        # local conditioning
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels,
                                      bias=bias,
                                      weight_normalization=weight_normalization)
        else:
            self.conv1x1c = None

        # global conditioning
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=bias,
                                      weight_normalization=weight_normalization)
        else:
            self.conv1x1g = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias,
                                     weight_normalization=weight_normalization)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias,
                                      weight_normalization=weight_normalization)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """Forward

        Args:
            x (Variable): B x C x T
            c (Variable): B x C x T, Local conditioning features
            g (Variable): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not

        Returns:
            Variable: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb

        x = F.tanh(a) * F.sigmoid(b)

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for conv in [self.conv, self.conv1x1_out, self.conv1x1_skip,
                     self.conv1x1c, self.conv1x1g]:
            if conv is not None:
                self.conv.clear_buffer()


def pad2d(inputs, kernel_size, stride, padding):
    """Return the required padding in both C and T dimensions."""
    # In PyTorch pad is (pad_left, pad_right, pad_up, pad_bottom).
    # Impose that temporal length is kept.
    B, _, C, T = inputs.size()
    pad_T = int(0.5 * (kernel_size - 1))
    if padding is "SAME":
        pad_C = int(0.5 * (stride*C - 1 + kernel_size - C))
        return (pad_T, pad_T, pad_C, pad_C)
    else:
        return (pad_T, pad_T, 0, 0)


class SepConv(nn.Module):
    """Depthwise Separable Conv layer.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: kernel size of the conv.
        stride: stride over the freq axis.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SepConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels,
            kernel_size, stride=(stride, 1), groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs, padding):
        pad = pad2d(inputs, self.kernel_size, self.stride, padding=padding)
        inputs = F.pad(inputs, pad)
        h = self.depthwise_conv(inputs)
        h = self.pointwise_conv(h)
        return h


class ConvStep(nn.Module):
    """ReLU + 2 x SepConv + BatchNorm.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: kernel size of the second conv
        stride: stride along the freq dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvStep, self).__init__()
        self.conv = SepConv(in_channels, out_channels, kernel_size, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, padding):
        h = F.relu(inputs)
        h = self.conv(h, padding=padding)
        h = self.batch_norm(h)
        return h


class ConvRes(nn.Module):
    """ConvStep + (2xConvStep + MaxPool).

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        is_stride: Whether there is stride=2 in the freq axis.
    """
    def __init__(self, in_channels, out_channels, is_stride=False):
        super(ConvRes, self).__init__()
        stride = 1 if is_stride is False else 2 
        self.stride = stride

        self.left_conv_step = ConvStep(in_channels, out_channels, 1, stride=stride)
        self.right_conv_1 = ConvStep(in_channels, out_channels, 3)
        self.right_conv_2 = ConvStep(out_channels, out_channels, 3)

    def forward(self, inputs):
        h_left = self.left_conv_step(inputs, padding=None)
        h_right = self.right_conv_1(inputs, padding="SAME")
        h_right = self.right_conv_2(h_right, padding="SAME")
        h_right = F.max_pool2d(h_right, 3, stride=(self.stride,1), padding=1)
        h = h_left + h_right
        return h


class SpectrogramModality(nn.Module):
    """ N x ConvRes.

    Args:
        N: Number of layers in the modality.
        is_stride: Whether or not there is stride along freq axis.
    """
    def __init__(self, N, is_stride=False):
        super(SpectrogramModality, self).__init__()
        self.N = N + 1
        self.convs = nn.ModuleList([])
        for n in range(1, N + 1):
            self.convs.append(ConvRes(2**(n-1), 2**n, is_stride=is_stride))

    def forward(self, inputs):
        h = inputs
        for layer in self.convs:
            h = layer(h)
        return h


class BodyNet(nn.Module):
    """ Bi-LSTM + ConvRes + Linear.

    Args:
        input_size (int): number of input features per sample, 
            in_channels times vector length.
        hidden_size (int): state and cell vector length.
        out_channels (int): number of output channels in the inner
            convolution.
        features_size (int): number of features in the conditioning.
            In our case, typicalle 80 melspecs.
    """
    def __init__(self, input_size, hidden_size, 
        out_channels, cin_channels):
        super(BodyNet, self).__init__()

        self.biLSTM = nn.LSTM(input_size, hidden_size,
            batch_first=True, bidirectional=True)
        self.CNN = ConvRes(1, out_channels)
        self.lineal = nn.Linear(2 * hidden_size * out_channels,
         cin_channels)

    def forward(self, inputs, cin_channels):
        # B x F x C' x T 
        B, _, C_p, T = inputs.size()
        inner = Variable(torch.rand(B, cin_channels, T)).cuda()

        h = inputs.view(B, -1, T)
        h = F.relu(h)
        h = h.permute(0,2,1)
        h, _ = self.biLSTM(h)
        h = h.permute(0,2,1).unsqueeze(1)
        h = self.CNN(h)
        h = h.view(B, -1, T)
        for i in range(B):
            for j in range(T):
                inner[i,:,j] = self.lineal(h[i,:,j])
        h = F.relu(inner)
        return h


