# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
"""
Implementation of Where2comm Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()

        self.smooth = False
        self.thre = args['thre']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size,
                                             stride=1,
                                             padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center: k_size - center,
                   0 - center: k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(
                -(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g

        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]

            ori_communication_maps = \
            batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(
                1)  # dim1=2 represents the confidence of two anchors

            if self.smooth:
                communication_maps = self.gaussian_filter(
                    ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(
                communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(
                communication_maps.device)
            communication_mask = torch.where(communication_maps > self.thre,
                                             ones_mask, zeros_mask)

            communication_rate = communication_mask[0].sum() / (H * W)

            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(
                communication_mask.device)
            communication_mask_nodiag[::2] = ones_mask[::2]

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(
                ori_communication_maps * communication_mask_nodiag)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks,


def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0,
                                           1)  # (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]


class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v, confidence_map=None):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        if confidence_map is not None:
            context, weight = self.attn(q, k, v,
                                        quality_map=confidence_map)  # (1, H*W, C)
        else:
            context, weight = self.attn(q, k, v)  # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1  # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2


class TransformerFusion(nn.Module):
    def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True,
                 dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        self.with_spe = with_spe
        self.with_scm = with_scm

    def forward(self, batch_neighbor_feature, batch_neighbor_feature_pe,
                batch_confidence_map, record_len):
        x_fuse = []
        B = len(record_len)
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            neighbor_feature = batch_neighbor_feature[b]
            _, C, H, W = neighbor_feature.shape
            neighbor_feature_flat = neighbor_feature.view(N, C,
                                                          H * W)  # (N, C, H*W)

            if self.with_spe:
                neighbor_feature_pe = batch_neighbor_feature_pe[b]
                neighbor_feature_flat_pe = neighbor_feature_pe.view(N, C,
                                                                    H * W)  # (N, C, H*W)
                query = neighbor_feature_flat_pe[0:1, ...].permute(0, 2,
                                                                   1)  # (1, H*W, C)
                key = neighbor_feature_flat_pe.permute(0, 2, 1)  # (N, H*W, C)
            else:
                query = neighbor_feature_flat[0:1, ...].permute(0, 2,
                                                                1)  # (1, H*W, C)
                key = neighbor_feature_flat.permute(0, 2, 1)  # (N, H*W, C)

            value = neighbor_feature_flat.permute(0, 2, 1)

            if self.with_scm:
                confidence_map = batch_confidence_map[b]
                fused_feature = self.encode_layer(query, key, value,
                                                  confidence_map)  # (1, H*W, C)
            else:
                fused_feature = self.encode_layer(query, key,
                                                  value)  # (1, H*W, C)

            fused_feature = fused_feature.permute(0, 2, 1).reshape(1, C, H, W)

            x_fuse.append(fused_feature)
        x_fuse = torch.concat(x_fuse, dim=0)
        return x_fuse


def add_pe_map(x):
    # scale = 2 * math.pi
    temperature = 10000
    num_pos_feats = x.shape[-3] // 2  # positional encoding dimension. C = 2d

    mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool,
                       device=x.device)  # [H, W]
    not_mask = ~mask
    y_embed = not_mask.cumsum(0, dtype=torch.float32)  # [H, W]
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [H, W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32,
                         device=x.device)  # [0,1,2,...,d]
    dim_t = temperature ** (2 * (
                dim_t // 2) / num_pos_feats)  # 10000^(2k/d), k is [0,0,1,1,...,d/2,d/2]

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                        dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                        dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]

    if len(x.shape) == 4:
        x_pe = x + pos[None, :, :, :]
    elif len(x.shape) == 5:
        x_pe = x + pos[None, None, :, :, :]
    return x_pe


class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()

        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4
        self.downsample_rate = args[
            'downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]

        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
                elif self.agg_mode == 'MAX':
                    fuse_network = MaxFusion()
                elif self.agg_mode == 'Transformer':
                    fuse_network = TransformerFusion(
                        channels=num_filters[idx],
                        n_head=args['agg_operator']['n_head'],
                        with_spe=args['agg_operator']['with_spe'],
                        with_scm=args['agg_operator']['with_scm'])
                self.fuse_modules.append(fuse_network)
        else:
            if self.agg_mode == 'ATTEN':
                self.fuse_modules = AttenFusion(
                    args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'MAX':
                self.fuse_modules = MaxFusion()
            elif self.agg_mode == 'Transformer':
                self.fuse_network = TransformerFusion(
                    channels=args['agg_operator']['feature_dim'],
                    n_head=args['agg_operator']['n_head'],
                    with_spe=args['agg_operator']['with_spe'],
                    with_scm=args['agg_operator']['with_scm'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None,
                heads=None):
        """
        Fusion forwarding.

        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :,
                            [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                    self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                    self.downsample_rate * self.discrete_ratio * H) * 2

        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(x)

            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)

                ############ 1. Communication (Mask the features) #########
                if i == 0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(rm, record_len)
                        _, communication_masks, communication_rates = self.naive_communication(
                            batch_confidence_maps, record_len,
                            pairwise_t_matrix)
                        x = x * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)

                ############ 2. Split the confidence map #######################
                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]
                    # (N,N,4,4)
                    # t_matrix[i, j]-> from i to j
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features,
                                                          t_matrix[0, :, :, :],
                                                          (H, W))
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                ############ 4. Deconv ####################################
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len)
            batch_confidence_maps = self.regroup(rm, record_len)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                _, communication_masks, communication_rates = self.naive_communication(
                    batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)

            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                      t_matrix[0, :, :, :],
                                                      (H, W))
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

        return x_fuse, communication_rates, {}