import torch.nn as nn

from models.conformer.encoders import ConformerConvModule
from models.layers.feedforwards import FeedForwardModule


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_expansion_factor,
        conv_kernel_size,
        dropout,
        activation,
        conv_norm,
    ):
        super().__init__()

        self.ff1 = FeedForwardModule(
            d_model,
            ff_expansion_factor,
            dropout,
            activation,
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(
            d_model,
            conv_kernel_size,
            dropout,
            activation,
            conv_norm,
        )
        self.ff2 = FeedForwardModule(
            d_model,
            ff_expansion_factor,
            dropout,
            activation,
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)

        attn_in = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(attn_out)

        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers,
        num_heads,
        ff_expansion_factor,
        conv_kernel_size,
        dropout,
        activation,
        conv_norm,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_expansion_factor=ff_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    activation=activation,
                    conv_norm=conv_norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
