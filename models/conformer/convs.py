import torch.nn as nn

from models.layers.activations import get_activation
from models.layers.convs import SamePadDepthwiseConv1d
from models.layers.norms import get_norm_1d


class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout, activation, norm):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = SamePadDepthwiseConv1d(d_model, kernel_size)
        self.norm = get_norm_1d(norm, d_model)
        self.activation = get_activation(activation)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_in(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return x.transpose(1, 2)
