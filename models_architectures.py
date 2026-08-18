"""
models_architectures.py
==========================

Two additional predictor architectures for the admission-control benchmark,
both trained with the existing CS_MSELoss (models.py, unchanged):

    - EdgeGRU: identical role to EdgeLSTM, but nn.GRU instead of nn.LSTM
      (no separate cell state -- isolates whether the cell state actually
      matters for detecting non-Markovian noise patterns in the channel
      telemetry, or whether the GRU's simpler gating is enough).

    - EdgeTCN: a Temporal Convolutional Network with STRICTLY CAUSAL,
      DILATED nn.Conv1d layers. Causal: output at time t depends only on
      inputs at times <= t (implemented via LEFT-ONLY padding -- never pad
      or look at the right/future side, which would leak future
      information the model shouldn't have at inference time). Dilated:
      each layer's receptive field doubles (dilation = 2**layer_index),
      covering long windows with few layers. The appeal for edge inference
      is that convolutions are simple matrix multiplications, fully
      parallelizable across the sequence dimension (unlike LSTM/GRU's
      inherently sequential recurrence) -- this benchmark measures whether
      that translates into a real single-sample CPU latency win.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGRU(nn.Module):
    """Same role as EdgeLSTM, but nn.GRU (no separate cell state)."""

    def __init__(self, input_size: int, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.activation(self.head(out[:, -1, :]))


class CausalDilatedConv1d(nn.Module):
    """
    A single strictly-causal, dilated 1D convolution.

    Causality is enforced by padding ONLY on the left (past) side with
    `(kernel_size - 1) * dilation` zeros, then convolving with padding=0 --
    NOT by using nn.Conv1d's built-in symmetric padding and truncating the
    right side after the fact (a common but easy-to-get-backwards mistake
    that can silently leak future timesteps into the current prediction).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class EdgeTCN(nn.Module):
    """
    Temporal Convolutional Network: a stack of strictly causal, dilated
    1D convolutions (dilation doubling each layer), followed by the same
    linear + sigmoid head used by every other Edge* model in this project.
    """

    def __init__(self, input_size: int, hidden_channels: int = 16, kernel_size: int = 3, num_layers: int = 3):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(CausalDilatedConv1d(in_ch, hidden_channels, kernel_size, dilation=dilation))
            layers.append(nn.ReLU())
            in_ch = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last_step = out[:, :, -1]
        return self.activation(self.head(last_step))
