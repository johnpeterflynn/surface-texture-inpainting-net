import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils import checkpoint
from torch_geometric.typing import OptTensor
from torch_geometric.nn import BatchNorm as GeometricBatchNorm
from torch_geometric.nn import InstanceNorm as GeometricInstanceorm
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_mean, scatter_max
import functools
from models.modules import sage_conv_filter, edge_conv_filter, edge_conv_translation_invariance
from models.modules.singlebatchgroupnorm import SingleBatchGraphNorm
from models.modules.fastinstancenorm import FastInstanceNorm


def forward_conv(in_c, out_c, n_repeated=1, dilation=1, bias=True,
                 receptive_field_type='normal', padding_type='zero'):
    if receptive_field_type == 'large':
        kernel_size = 7
        padding = 3
    elif receptive_field_type == 'dilated':
        kernel_size = 3
        padding = dilation
    elif receptive_field_type == 'normal':
        kernel_size = 3
        padding = 1
    else:
        raise NotImplementedError('receptive field type [%s] is not implemented' % receptive_field_type)

    conv_block = []
    for i in range(n_repeated):
        out_channels_per_conv = out_c if i == n_repeated - 1 else in_c
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(padding)]
        elif padding_type == 'zero':
            p = padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(in_c, out_channels_per_conv,
                                 kernel_size=kernel_size, padding=p,
                                 dilation=dilation, bias=bias)]
    return conv_block


def down_conv(in_c, out_c, bias=True, pooling_type='stride'):
    if pooling_type == 'stride':
        conv_block = [nn.Conv2d(in_c, out_c,
                                kernel_size=3, stride=2,
                                padding=1, bias=bias)]
    else:
        if pooling_type == 'mean':
            conv_block = [nn.AvgPool2d(2)]
        elif pooling_type == 'max':
            conv_block = [nn.MaxPool2d(2)]
        else:
            raise NotImplementedError('pooling type [%s] is not implemented' % pooling_type)
        conv_block += forward_conv(in_c, out_c, bias=bias)
    return conv_block


def up_conv(in_c, out_c, bias=True, pooling_type='stride'):
    if pooling_type == 'stride':
        conv_block = [nn.ConvTranspose2d(in_c, out_c,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=bias)]
    else:
        conv_block = [nn.Upsample(scale_factor=2, mode='nearest')]
        conv_block += forward_conv(in_c, out_c, bias=bias)
    return conv_block


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # TODO: Parallel for both conv2d and graph conv
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    #init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, filter_type, norm='batch', dilation_order=0, use_dropout=False, n_blocks=6, n_levels=2,
             n_repeated_io_convs=1, init_type='normal', pooling_type='stride', io_receptive_field_type='large',
             checkpoint_bottleneck=False, num_blocks_per_uncheckpointed_block=1,
             use_label_embedding=False, num_classes=None, num_embedding=None, dilations=None,
             init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        filter_type(str)    -- type of filter to use: conv2d | edge | sage
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        dilation_order (int) -- max order of exponentially increasing dilation on Resnet layers (ResnetGenerator only)
        use_dropout (bool) -- if use dropout layers.
        n_blocks (int)      -- the number of ResNet blocks
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    if filter_type == 'conv2d':
        norm_layer_2d = get_norm_layer(norm_type=norm)
        net = Resnet2D(input_nc, output_nc, ngf, norm_layer=norm_layer_2d, use_dropout=use_dropout, n_blocks=n_blocks,
                          n_levels=n_levels, dilation_order=dilation_order, n_repeated_io_convs=n_repeated_io_convs,
                          pooling_type=pooling_type, io_receptive_field_type=io_receptive_field_type)
    elif filter_type == 'cfconv2d':
        norm_layer_2d = get_norm_layer(norm_type=norm)
        net = CoarseFineResnet2D(input_nc, output_nc, ngf, norm_layer=norm_layer_2d, use_dropout=use_dropout, n_blocks=n_blocks,
                          n_levels=n_levels, dilation_order=dilation_order, n_repeated_io_convs=n_repeated_io_convs,
                          pooling_type=pooling_type, io_receptive_field_type=io_receptive_field_type)
    else:
        net = SurfaceTextureInpaintingNet(input_nc, output_nc, filter_type, ngf, norm_type=norm, n_blocks=n_blocks,
                          n_levels=n_levels, n_repeated_io_convs=n_repeated_io_convs, pooling_type=pooling_type,
                          checkpoint_bottleneck=checkpoint_bottleneck,
                          num_blocks_per_uncheckpointed_block=num_blocks_per_uncheckpointed_block,
                          use_label_embedding=use_label_embedding, num_classes=num_classes, num_embedding=num_embedding,
                          dilations=dilations)

    return init_net(net, init_type, init_gain, gpu_ids)


class SurfaceTextureInpaintingNet(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, filter_type, ngf=64, norm_type='instance', n_blocks=6,
                 n_levels=2, n_repeated_io_convs=1, pooling_type='mean', checkpoint_bottleneck=False,
                 num_blocks_per_uncheckpointed_block=1, use_label_embedding=False, num_classes=None, num_embedding=None,
                 dilations=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            n_blocks (int)      -- the number of ResNet blocks
            n_levels (int)      -- the number of encoder-decoder steps
            n_repeated_io_convs (int)      -- the number of size-preserving convs at input/output layers
            pooling_type (str)  -- the type of pooling layer in encoder/decoder steps: mean | max
            checkpoint_bottleneck (bool) -- if true, recalculate forward pass of bottleneck during backward pass
            num_blocks_per_uncheckpointed_block (int) -- number of resnet blocks in bottleneck per block that is not checkpointed (1 = all blocks checkpointed)
        """
        assert(n_blocks >= 0)
        super(SurfaceTextureInpaintingNet, self).__init__()

        if filter_type == 'edgeconv' or filter_type == 'edgeconvtransinv':
            get_gcn_filter = edge_conv_filter.get_gcn_filter
        elif filter_type == 'sageconv' or filter_type == 'sageconvtransinv':
            get_gcn_filter = sage_conv_filter.get_gcn_filter
        else:
            raise NotImplementedError('No filter implemented for gcn filter type {}'.format(filter_type))

        class BatchNorm2Param(GeometricBatchNorm):
            def __init__(self, *args, **kwargs):
                super(BatchNorm2Param, self).__init__(*args, **kwargs)

            def forward(self, input: torch.Tensor, batch: OptTensor = None) -> torch.Tensor:
                return super(BatchNorm2Param, self).forward(input)

        if norm_type == 'batch':
            self.norm = BatchNorm2Param
            self.using_norm = True
        elif norm_type == 'instance':
            #self.norm = GeometricInstanceorm
            self.norm = FastInstanceNorm
            self.using_norm = True
        elif norm_type == 'graph':
            # TODO: It may not be appropriate to apply GraphNorm to the final fully connected layer
            self.norm = SingleBatchGraphNorm
            #self.norm = GraphNorm
            self.using_norm = True
        else:
            # Identity isn't implemented for Pytorch 1.8 so we'll implement it manually
            class Identity(nn.Module):
                def __init__(self, *args, **kwargs):
                    super(Identity, self).__init__()
                def forward(self, input: torch.Tensor, batch: OptTensor = None) -> torch.Tensor:
                    return input
            self.norm = Identity
            self.using_norm = False

        # TODO: Import these values
        # TODO: Difference between functional and object?
        self._act = nn.functional.elu
        self._output_activation = nn.Tanh()
        self.use_bias = True
        self._pooling_type = pooling_type
        self._inplace = False
        self.checkpoint_bottleneck = checkpoint_bottleneck
        self.num_blocks_per_uncheckpointed_block = num_blocks_per_uncheckpointed_block
        self._use_embedding = use_label_embedding
        self.dilations = dilations if dilations is not None else np.ones(n_blocks)

        if self._use_embedding:
            self.label_embedding = nn.Embedding(num_classes, num_embedding, padding_idx=0)

        #self.input_geo_filters = []
        self.input_blocks = []
        for i in range(n_repeated_io_convs):
            out_channels_per_conv = ngf if i == n_repeated_io_convs - 1 else input_nc

            if i == 0:
                if filter_type == 'edgeconvtransinv':
                    first_filter = edge_conv_translation_invariance.EdgeConvTransInv
                    double_input = False
                elif filter_type == 'edgeconv':
                    first_filter = None  # default
                    double_input = True
                elif filter_type == 'sageconvtransinv':
                    first_filter = sage_conv_filter.SAGEConvTransInv
                    double_input = False
                elif filter_type == 'sageconv':
                    first_filter = None  # default
                    double_input = False
                else:
                    first_filter = None  # default
                    double_input = False
                #self.input_geo_filters += [get_gcn_filter(input_nc, out_channels_per_conv,
                #                                          inplace=self._inplace, bias=self.use_bias,
                #                                          module=first_filter, double_input=double_input)]
                self.input_blocks += [GraphResnetBlock(input_nc, out_channels_per_conv, get_gcn_filter,
                                                       self.norm, self._inplace, self.use_bias,
                                                       module=first_filter, double_input=double_input)]
            else:
                #self.input_geo_filters += [get_gcn_filter(input_nc, out_channels_per_conv,
                #                                          inplace=self._inplace, bias=self.use_bias)]
                self.input_blocks += [GraphResnetBlock(input_nc, out_channels_per_conv, get_gcn_filter,
                                                       self.norm, self._inplace, self.use_bias)]
        # TODO: Norm per-conv or for the whole batch?
        #self.input_geo_filters = nn.ModuleList(self.input_geo_filters)
        #self.input_norm = self.norm(ngf)
        self.input_blocks = nn.ModuleList(self.input_blocks)

        #self.encoder_pooling_blocks = []
        self.encoder_blocks = []
        for i in range(n_levels):  # add downsampling layers
            mult = 2 ** i
            in_size = ngf * mult
            if i == 0 and self._use_embedding:
                in_size += num_embedding
            #self.encoder_pooling_blocks += [get_gcn_filter(in_size, in_size, inplace=self._inplace, bias=self.use_bias)]
            self.encoder_blocks += [GraphResnetBlock(in_size, ngf * mult * 2, get_gcn_filter, self.norm, self._inplace, self.use_bias)]
        #self.encoder_pooling_blocks = nn.ModuleList(self.encoder_pooling_blocks)
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.bottleneck_blocks = []
        mult = 2 ** n_levels
        for i in range(n_blocks):  # add ResNet blocks
            self.bottleneck_blocks += [GraphResnetBlock(ngf * mult, ngf * mult, get_gcn_filter, self.norm, self._inplace, self.use_bias, is_checkpointed=self.checkpoint_bottleneck)]
        self.bottleneck_blocks = nn.ModuleList(self.bottleneck_blocks)

        self.decoder_blocks = []
        for i in range(n_levels):  # add upsampling layers
            mult = 2 ** (n_levels - i)
            self.decoder_blocks += [
                GraphResnetBlock(ngf * mult, int(ngf * mult / 2), get_gcn_filter, self.norm, self._inplace, self.use_bias)]
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # TODO: No norm, right? What about if there is an FC layer on the output?
        #self.output_geo_filters = []
        self.output_blocks = []
        for i in range(n_repeated_io_convs):
            out_channels_per_conv = ngf#output_nc if i == n_repeated_io_convs - 1 else ngf
            #self.output_geo_filters += [get_gcn_filter(ngf, out_channels_per_conv,
            #                                           inplace=self._inplace, bias=self.use_bias)]
            self.output_blocks += [
                GraphResnetBlock(ngf, out_channels_per_conv, get_gcn_filter, self.norm, self._inplace,
                                 self.use_bias)]
        #self.output_geo_filters = nn.ModuleList(self.output_geo_filters)
        #self.output_norm = self.norm(ngf)
        self.output_blocks = nn.ModuleList(self.output_blocks)

        self.final_linear1 = nn.Linear(ngf, ngf, bias=self.use_bias)
        self.final_norm1 = self.norm(ngf)
        self.final_linear2 = nn.Linear(ngf, output_nc)

        def init_weights(m):
            classname = m.__class__.__name__
            if classname.find('EdgeConv') != -1:
                pass
                #assert m.nn[0].__class__.__name__.find('Linear') != -1, '0 not Linear'
                #assert m.nn[3].__class__.__name__.find('Linear') != -1, '3 not Linear'
                #nn.init.xavier_uniform_(m.nn[0].weight, nn.init.calculate_gain('relu'))
                #nn.init.xavier_uniform_(m.nn[2].weight, nn.init.calculate_gain('relu'))
                #nn.init.zeros_(m.nn[0].bias)
                #nn.init.zeros_(m.nn[3].bias)
            elif classname.find('Linear') != -1:
                #nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain('relu'))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights)

        #nn.init.xavier_uniform_(self.final_linear1.weight, nn.init.calculate_gain('relu'))
        #if hasattr(self.final_linear1, 'bias') and self.final_linear1.bias is not None:
        #    nn.init.zeros_(self.final_linear1.bias)
        #nn.init.xavier_uniform_(self.final_conv[-2].weight, nn.init.calculate_gain('tanh'))
        #nn.init.zeros_(self.final_conv[-2].bias)

    def _pooling(self, vertex_features, traces, num_pooled_vertices):
        if self._pooling_type == 'mean':
            return scatter_mean(vertex_features, traces, dim=0, dim_size=num_pooled_vertices)
        if self._pooling_type == 'max':
            return scatter_max(vertex_features, traces, dim=0, dim_size=num_pooled_vertices)[0]

        raise ValueError(f"Unknown pooling type {self._pooling_method}")

    def _unpooling(self, vertex_features, traces):
        return vertex_features[traces]

    def _forward_block(self, vertex_features, geo_filter, geo_edges, geo_norm, batch=None):
        geo = geo_filter(vertex_features, geo_edges)
        geo = geo_norm(geo, batch=batch)
        return self._act(geo, inplace=self._inplace)

    def forward(self, sample):
        """Standard forward"""
        #assert len(sample.num_vertices) == 1, 'Error: pooling not implemented for batch size > 1'

        num_levels = len(self.decoder_blocks) + 1

        out = sample.x
        #for i, filter in enumerate(self.input_geo_filters):
        for i, filter in enumerate(self.input_blocks):
            out = filter(out, sample.edge_index)
        #out = self.input_norm(out, batch=sample.batch)
        #if self._use_embedding:
        #    out = torch.cat([out, self.label_embedding(sample.labels)], dim=-1)
        #out = self._act(out)

        total_num_vertices = sample.num_vertices.sum(dim=0)

        # Don't use batch indices if batch size is 1
        batch = sample.batch if sample.batch.max() > 0 else None

        for i, block in enumerate(self.encoder_blocks):
            level = i + 1
            trace_indices = sample[f"hierarchy_trace_index_{level}"]
            if batch is not None:
                batch = scatter_max(batch, trace_indices, dim=0, dim_size=total_num_vertices[level])[0]
            out = self._pooling(out, trace_indices, total_num_vertices[level])
            #with torch.no_grad():
            #    pooling_edge_indices = torch.stack([torch.arange(0, len(trace_indices)).to(out.get_device()),
            #                                        trace_indices], dim=1).t().contiguous()
            #out = self.encoder_pooling_blocks[i](out, pooling_edge_indices)[:total_num_vertices[level]]
            #out = block(out, sample[f"hierarchy_edge_index_{level}"], batch)
            out = checkpoint.checkpoint(block, out, sample[f"hierarchy_edge_index_{level}"], batch, preserve_rng_state=False)

        for i, block in enumerate(self.bottleneck_blocks):
            if self.dilations[i] > 1:
                edge_set_name = f"hierarchy_dil_{self.dilations[i]}_edge_index_{num_levels - 1}"
            else:
                edge_set_name = f"hierarchy_edge_index_{num_levels - 1}"
            if self.checkpoint_bottleneck and (i + 1) % self.num_blocks_per_uncheckpointed_block == 0:
                # False indicates that checkpoint does not save RNG state (random number generator).
                out = checkpoint.checkpoint(block, out, sample[edge_set_name], batch, preserve_rng_state=False)
            else:
                out = block(out, sample[edge_set_name], batch)

        for i, block in enumerate(self.decoder_blocks):
            level = i + 1
            trace_indices = sample[f"hierarchy_trace_index_{num_levels - level}"]
            out = self._unpooling(out, trace_indices)
            if batch is not None:
                batch = batch.index_select(0, trace_indices)

            if level == num_levels - 1:
                #out = block(out, sample.edge_index, batch)
                out = checkpoint.checkpoint(block, out, sample.edge_index, batch, preserve_rng_state=False)
            else:
                #out = block(out, sample[f"hierarchy_edge_index_{num_levels - level - 1}"], batch)
                out = checkpoint.checkpoint(block, out, sample[f"hierarchy_edge_index_{num_levels - level - 1}"], batch,
                                            preserve_rng_state=False)

        # TODO: Maybe add a ReLU to output of graph conv, then a fully connected layer then Tanh
        #for i, filter in enumerate(self.output_geo_filters):
        for i, filter in enumerate(self.output_blocks):
            out = filter(out, sample.edge_index)
        #out = self.output_norm(out, batch=sample.batch)
        #out = self._act(out)

        out = self.final_linear1(out)
        out = self.final_norm1(out, batch=sample.batch)
        out = self._act(out)
        out = self.final_linear2(out)

        out = self._output_activation(out)

        return out


class GraphResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim_in, dim_out, get_gcn_filter, norm_layer, inplace, use_bias, is_checkpointed=False,
                 module=None, double_input=None):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(GraphResnetBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.act = nn.ELU()
        if module is not None:
            self.first_filter = get_gcn_filter(dim_in, dim_out, inplace=inplace, bias=use_bias, module=module, double_input=double_input)
        else:
            self.first_filter = get_gcn_filter(dim_in, dim_out, inplace=inplace, bias=use_bias)
        #self.second_filter = get_gcn_filter(dim_out, dim_out, inplace=inplace, bias=use_bias)
        if is_checkpointed and issubclass(norm_layer, GeometricBatchNorm):
            # Account for two forward passes through BM.
            # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
            self.first_norm = norm_layer(dim_out, momentum=math.sqrt(0.1))
            #self.second_norm = norm_layer(dim_out, momentum=math.sqrt(0.1))
        else:
            self.first_norm = norm_layer(dim_out)
            #self.second_norm = norm_layer(dim_out)
        if dim_in != dim_out:
            self.shortcut = nn.Linear(dim_in, dim_out)

    def forward(self, x, edges, batch=None):
        """Forward function (with skip connections)"""
        # NOTE: It is assumed that this forward pass is completely deterministic
        out = self.first_filter(x, edges)
        out = self.first_norm(out, batch)
        out = self.act(out)
        #out = self.second_filter(out, edges)

        if self.dim_in != self.dim_out:
            x = self.shortcut(x)

        # TODO: Are graphs better with or without activation on output?
        # return self.act(x + self.second_norm(out))
        #return x + self.second_norm(out, batch)  # add skip connections
        return x + out


class Resnet2D(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 dilation_order=0, n_levels=2, n_repeated_io_convs=1, padding_type='reflect', pooling_type='stride',
                 io_receptive_field_type='large'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            dilation_order (int)-- max order of exponentially increasing dilation on Resnet layers
            n_levels (int)      -- the number of encoder-decoder steps
            n_repeated_io_convs (int)      -- the number of size-preserving convs at input/output layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            pooling_type (str)  -- the type of pooling layer in encoder/decoder steps: stride | mean | max
            io_receptive_field_type (str)  -- desired conv size type in encoder/decoder steps: large | normal | dilated
        """
        assert(n_blocks >= 0)
        assert(dilation_order < n_blocks)
        super(Resnet2D, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = forward_conv(input_nc, ngf,
                             bias=use_bias, n_repeated=n_repeated_io_convs,
                             receptive_field_type=io_receptive_field_type,
                             padding_type=padding_type)
        model += [norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_levels):  # add downsampling layers
            mult = 2 ** i

            #model += [ResnetBlock(ngf * mult, ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer,
            #                      use_dropout=use_dropout, use_bias=use_bias)]
            model += down_conv(ngf * mult, ngf * mult * 2,
                               bias=use_bias, pooling_type=pooling_type)
            model += [norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_levels
        d_start = n_blocks - dilation_order - 1  # int(np.ceil(n_blocks / 2) - 1 - np.floor((dilation_order + (n_blocks % 2)) // 2))
        for i in range(n_blocks):       # add ResNet blocks
            if d_start <= i <= d_start + dilation_order:
                dilation = 2 ** (i - d_start)
            else:
                dilation = 1

            model += [ResnetBlock(ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, dilation=dilation)]

        for i in range(n_levels):  # add upsampling layers
            mult = 2 ** (n_levels - i)
            #model += [ResnetBlock(ngf * mult, int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer,
            #                      use_dropout=use_dropout, use_bias=use_bias)]
            model += up_conv(ngf * mult, int(ngf * mult / 2),
                             bias=use_bias, pooling_type=pooling_type)
            model += [norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += forward_conv(ngf, output_nc, n_repeated=n_repeated_io_convs,
                              receptive_field_type=io_receptive_field_type, padding_type=padding_type)
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, sample=None):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias, dilation=1):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.conv_block = self.build_conv_block(dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias,
                                                dilation)

        if self.dim_in != self.dim_out:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=use_bias)

    def build_conv_block(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias, dilation):
        """Construct a convolutional block.

        Parameters:
            dim_in (int)           -- the number of channels in the input conv layer.
            dim_out (int)           -- the number of channels in the output conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
            dilation (int)      -- dilation of first convolution

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """

        conv_block = forward_conv(dim_in, dim_out,
                                  receptive_field_type='dilated', dilation=dilation,
                                  bias=use_bias, padding_type=padding_type)
        conv_block += [norm_layer(dim_out), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        #conv_block += forward_conv(dim_out, dim_out,
        #                           receptive_field_type='normal', bias=use_bias,
        #                           padding_type=padding_type)
        #conv_block += [norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.conv_block(x)  # add skip connections
        if self.dim_in != self.dim_out:
            x = self.shortcut(x)
        return x + out