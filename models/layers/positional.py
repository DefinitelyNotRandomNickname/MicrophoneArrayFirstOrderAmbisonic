import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        sinusoid = torch.outer(positions, self.inv_freq)

        pe = torch.zeros(
            seq_len,
            x.size(-1),
            device=x.device,
            dtype=self.inv_freq.dtype,
        )
        pe[:, 0::2] = sinusoid.sin()
        pe[:, 1::2] = sinusoid.cos()[:, : pe[:, 1::2].size(-1)]

        return x + pe.to(dtype=x.dtype).unsqueeze(0)
