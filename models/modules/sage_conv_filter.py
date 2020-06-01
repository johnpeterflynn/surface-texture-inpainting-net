"""Function definitions for graph conv modules used in our DCM Net
"""
import torch
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, InstanceNorm1d
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import InstanceNorm, BatchNorm
import torch_geometric.nn.conv.sage_conv as sage
from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_geometric.typing import Union, Tuple

from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul


class SAGEConvTransInv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConvTransInv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # TODO: Remove hard-coding on position and normal dims
        x_j[:, 3:9] = x_j[:, 3:9] - x_i[:, 3:9]
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def get_gcn_filter(input_size: int, output_size, activation: torch.nn.Module = None,
                   inplace: bool = False, aggregation: str = 'mean', bias: bool = True,
                   module: MessagePassing = None, double_input=False):
    """returns graph conv module with specified arguments and type

    Arguments:
        input_size {int} -- input size (2 * current vertex feature size!)
        output_size {[type]} -- feature size of new vertex features
        activation {torch.nn.Module} -- activation function for internal MLP

    Keyword Arguments:
        inplace {bool} -- (default: {False})
        aggregation {str} -- permutation-invariant feature aggregation of adjacent vertices (default: {'mean'})
        module {MessagePassing} -- graph convolutional module (default: {edge.EdgeConv})
    """

    assert input_size >= 0
    assert output_size >= 0

    if module is None:
        module = sage.SAGEConv

    class SumSAGEConv(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super(SumSAGEConv, self).__init__()
            self.sage1 = module(*args, **kwargs)
            #self.sage2 = module(*args, **kwargs)
            #self.sage3 = module(*args, **kwargs)

        def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> torch.Tensor:
            x1 = self.sage1(x, edge_index, size)
            #x2 = self.sage2(x, edge_index, size)
            #x3 = self.sage3(x, edge_index, size)
            return x1# + x2 + x3

    return SumSAGEConv(input_size, output_size, normalize=False, root_weight=True, bias=bias)
