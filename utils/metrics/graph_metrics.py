from typing import Union
import torch
from torch_geometric.nn import MessagePassing


class GraphLaplaceOperator(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        x_with_ind = torch.cat([x.new_ones(x.shape[0], 1), x], dim=1)
        prop = self.propagate(edge_index, x=x_with_ind)
        return prop[:, 1:] - prop[:, 0:1] * x

    def message(self, x_j):
        return x_j


class GraphLaplaceVariance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = GraphLaplaceOperator()
        for param in self.parameters():
            param.requires_grad = False

    def grayscale(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(self, x, edge_indices):
        gray = self.grayscale(x)
        return torch.var(self.filter(gray, edge_indices), dim=0, unbiased=False)


def graph_total_variation(x, edge_indices):
    h, w = x.shape
    tv = torch.abs(x[edge_indices[0]] - x[edge_indices[1]]).sum()
    tv /= h * w
    return tv


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1.0,
         convert_to_greyscale: bool = False) -> torch.Tensor:
    r"""Compute Peak Signal-to-Noise Ratio between two graphs.
    Supports both greyscale and color vertices with RGB channel order.

    Args:
        x: Predicted graph vertex color. Shape (N, C).
        y: Target graph vertex color. Shape (N, C).
        data_range: Value range of input color (usually 1.0 or 255). Default: 1.0
        convert_to_greyscale: Convert vertex color to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.

    Returns:
        PSNR: Index of similarity betwen two graphs.

    References:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    # Constant for numerical stability
    EPS = 1e-8

    x = x / data_range
    y = y / data_range

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[0, 1])
    score: torch.Tensor = - 10 * torch.log10(mse + EPS)

    return score

