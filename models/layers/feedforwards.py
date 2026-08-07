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
