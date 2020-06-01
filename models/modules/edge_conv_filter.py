"""Function definitions for graph conv modules used in our DCM Net
"""
import torch
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, InstanceNorm1d
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import InstanceNorm, BatchNorm
import torch_geometric.nn.conv.edge_conv as edge


def get_gcn_filter(input_size: int, output_size, activation: torch.nn.Module = torch.nn.ReLU,
                   inplace: bool = False, aggregation: str = 'mean', bias: bool = True,
                   module: MessagePassing = None, double_input=True, with_norm=False):
    """returns graph conv module with specified arguments and type

    Arguments:
        input_size {int} -- input size current vertex feature size
        output_size {[type]} -- feature size of new vertex features
        activation {torch.nn.Module} -- activation function for internal MLP

    Keyword Arguments:
        inplace {bool} -- (default: {False})
        aggregation {str} -- permutation-invariant feature aggregation of adjacent vertices (default: {'mean'})
        module {MessagePassing} -- graph convolutional module (default: {edge.EdgeConv})
    """

    assert input_size >= 0
    assert output_size >= 0

    double_input_size = 2 * input_size if double_input else input_size

    if module is None:
        module = edge.EdgeConv

    if with_norm:
        inner_module = Seq(
            Lin(double_input_size, 2 * output_size, bias=False),
            # TODO: Should norms be 1D or geometric? If geometric, should we use GraphNorm?
            #InstanceNorm(2 * output_size),
            BatchNorm1d(2 * output_size),#, track_running_stats=False, affine=False),
            activation(inplace=inplace),
            Lin(2 * output_size, output_size, bias=False),
            #InstanceNorm(output_size)
            BatchNorm1d(output_size),#, track_running_stats=False, affine=False)
        )
    else:
        inner_module = Seq(
            Lin(double_input_size, 2 * output_size),
            # TODO: Should norms be 1D or geometric? If geometric, should we use GraphNorm?
            #InstanceNorm(2 * output_size),
            #BatchNorm1d(2 * output_size),#, track_running_stats=False, affine=False),
            activation(inplace=inplace),
            Lin(2 * output_size, output_size),
            #InstanceNorm(output_size)
            #BatchNorm1d(output_size),#, track_running_stats=False, affine=False)
        )

    return module(inner_module, aggr=aggregation)
    #return module(InnerModule(double_input_size, output_size), aggr=aggregation)
