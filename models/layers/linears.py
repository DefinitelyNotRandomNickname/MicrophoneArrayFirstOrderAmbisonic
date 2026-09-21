import math

import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    """Independent linear maps for every group along the second-to-last axis.

    Input:  [..., num_groups, in_features]
    Output: [..., num_groups, out_features]
    """

    def __init__(self, in_features, out_features, num_groups, bias=True):
        super().__init__()

        if in_features < 1 or out_features < 1 or num_groups < 1:
            raise ValueError(
                "in_features, out_features, and num_groups must be positive"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.weight = nn.Parameter(torch.empty(num_groups, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Same bounds as nn.Linear so every group starts as an independent layer.
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.dim() < 2:
            raise ValueError(
                "GroupedLinear expects input with shape [..., num_groups, in_features]"
            )
        if x.size(-2) != self.num_groups or x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected trailing shape ({self.num_groups}, {self.in_features}), "
                f"got {tuple(x.shape[-2:])}"
            )

        output = torch.einsum("...gi,goi->...go", x, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_groups={self.num_groups}, bias={self.bias is not None}"
        )
