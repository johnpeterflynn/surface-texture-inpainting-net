import torch
from torch.nn import Sequential as Seq, Linear as Lin, functional as F, ReLU, LeakyReLU, BatchNorm1d
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_max
from base.base_model import BaseModel
from models.modules.edge_conv_filter import get_gcn_filter
from models.modules.edge_conv_translation_invariance import EdgeConvTransInv


class SingleConvMeshNet(BaseModel):
    """U-Net architecture which only operate in either the geodesic or Euclidean domain."""

    def __init__(self, feature_number, num_propagation_steps, filter_sizes, num_classes=3,
                 pooling_method='mean', aggr='mean'):
        super(SingleConvMeshNet, self).__init__()

        activation = 'ReLU'

        curr_size = feature_number

        inplace = False
        self._pooling_method = pooling_method

        if activation == 'ReLU':
            self._activation = ReLU
            self._act = F.relu
        elif activation == 'LeakyReLU':
            self._activation = LeakyReLU
            self._act = F.leaky_relu
        else:
            raise NotImplementedError(f"{activation} is not implemented")

        self.left_geo_cnns = []

        self.right_geo_cnns = []

        self.pooling_cnns = []
        self.squeeze_cnns = []

        self._graph_levels = len(filter_sizes)

        for level in range(len(filter_sizes)):
            if level < len(filter_sizes) - 1:
                if level == 0:
                    # First level needs translation invariant version of edge conv
                    left_geo = [get_gcn_filter(
                        curr_size, filter_sizes[level], self._activation, aggregation=aggr, module=EdgeConvTransInv,
                        double_input=False, with_norm=True)]
                else:
                    left_geo = [get_gcn_filter(
                        curr_size, filter_sizes[level], self._activation, aggregation=aggr, with_norm=True)]

                for _ in range(num_propagation_steps - 1):
                    left_geo.append(get_gcn_filter(filter_sizes[level], filter_sizes[level],
                                                   self._activation, aggregation=aggr, with_norm=True))

                # DECODER branch of U-NET
                curr_size = filter_sizes[level] + filter_sizes[level + 1]

                right_geo = [get_gcn_filter(
                    curr_size, filter_sizes[level], self._activation, aggregation=aggr, with_norm=True)]

                for _ in range(num_propagation_steps - 1):
                    right_geo.append(get_gcn_filter(filter_sizes[level], filter_sizes[level],
                                                    self._activation, aggregation=aggr, with_norm=True))

                right_block = self.ResBlock(right_geo, self._act)
                self.right_geo_cnns.append(right_block)

                curr_size = filter_sizes[level]
            else:
                left_geo = [get_gcn_filter(
                    curr_size, filter_sizes[level], self._activation, aggregation=aggr, with_norm=True)]
                for _ in range(num_propagation_steps - 1):
                    left_geo.append(get_gcn_filter(filter_sizes[level], filter_sizes[level],
                                                   self._activation, aggregation=aggr, with_norm=True))

            left_block = self.ResBlock(left_geo, self._act)
            self.left_geo_cnns.append(left_block)

        self.final_convs = [
            Seq(
                Lin(filter_sizes[0], filter_sizes[0] // 2),
                BatchNorm1d(filter_sizes[0] // 2),
                self._activation(inplace=inplace),
                Lin(filter_sizes[0] // 2, num_classes)
            )
        ]

        self.left_geo_cnns = torch.nn.ModuleList(self.left_geo_cnns)
        self.right_geo_cnns = torch.nn.ModuleList(self.right_geo_cnns)
        self.final_convs = torch.nn.ModuleList(self.final_convs)

    class ResBlock(torch.nn.Module):
        def __init__(self, filters, act):
            super(SingleConvMeshNet.ResBlock, self).__init__()
            self.filters = torch.nn.ModuleList(filters)
            self._act = act

        def forward(self, vertex_features, geo_edges, inplace=False):
            residual_geo = self.filters[0](vertex_features, geo_edges)
            vertex_features = self._act(residual_geo, inplace=inplace)

            for step in range(1, len(self.filters)):
                residual_geo = self.filters[step](vertex_features, geo_edges)
                vertex_features += residual_geo
                vertex_features = self._act(vertex_features, inplace=inplace)
            return vertex_features

    def _pooling(self, vertex_features, edges):
        if self._pooling_method == 'mean':
            return scatter_mean(vertex_features, edges, dim=0)
        if self._pooling_method == 'max':
            return scatter_max(vertex_features, edges, dim=0)[0]

        raise ValueError(f"Unkown pooling type {self._pooling_method}")

    def forward(self, sample):
        levels = []
        #level1 = torch.cat((sample.pos, sample.x), dim=-1)
        #level1 = torch.cat([sample.x, sample.mask.unsqueeze(1)], dim=-1)
        level1 = self.left_geo_cnns[0](sample.x, sample.edge_index)

        levels.append(level1)

        # ENCODER BRANCH
        for level in range(1, self._graph_levels):
            curr_level = self._pooling(levels[-1],
                                       sample[f"hierarchy_trace_index_{level}"])
            curr_level = checkpoint.checkpoint(self.left_geo_cnns[level],
                                               curr_level, sample[f"hierarchy_edge_index_{level}"],
                                               preserve_rng_state=False)

            levels.append(curr_level)

        current = levels[-1]

        # DECODER BRANCH
        for level in range(1, self._graph_levels):
            back = current[sample[f"hierarchy_trace_index_{self._graph_levels - level}"]]
            fused = torch.cat((levels[-(level + 1)], back), -1)

            if level == self._graph_levels - 1:
                fused = self.right_geo_cnns[-level](fused, sample.edge_index)
            else:
                fused = checkpoint.checkpoint(self.right_geo_cnns[-level], fused,
                                              sample[f"hierarchy_edge_index_{self._graph_levels - level - 1}"],
                                              preserve_rng_state=False)
            current = fused

        result = current

        for conv in self.final_convs:
            result = conv(result)

        return result
