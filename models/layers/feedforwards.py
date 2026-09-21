import torch.nn as nn

from models.layers.activations import get_activation


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout, activation):
        super().__init__()

        hidden_dim = int(d_model * expansion_factor)

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvFeedForwardModule(nn.Module):
    """Feed-forward module whose hidden units are mixed by a grouped convolution.

    Input/output: [B, L, D]
    """

    def __init__(
        self,
        d_model,
        expansion_factor,
        dropout,
        activation,
        kernel_size=3,
        groups=1,
        eps=1e-5,
    ):
        super().__init__()

        hidden_dim = int(d_model * expansion_factor)
        if d_model < 1 or hidden_dim < 1:
            raise ValueError("d_model and d_model * expansion_factor must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if groups < 1 or hidden_dim % groups != 0:
            raise ValueError("d_model * expansion_factor must be divisible by groups")

        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, kernel_size=1),
            get_activation(activation, channels=hidden_dim),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding="same",
                groups=groups,
            ),
            get_activation(activation, channels=hidden_dim),
            nn.Conv1d(hidden_dim, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("ConvFeedForwardModule expects input with shape [B, L, D]")

        x = self.norm(x).transpose(1, 2)
        return self.net(x).transpose(1, 2)
