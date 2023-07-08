# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

# This file only contains CoAlign's multiscale intermediate feature fusion.
# If you want build the agent-object pose graph to correct pose error, 
# please refer to https://github.com/yifanlu0227/CoAlign

# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.res_bev_backbone import ResBEVBackbone
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.fuse_modules.coalign_fuse import Att_w_Warp, normalize_pairwise_tfm

class PointPillarCoAlign(nn.Module):
    def __init__(self, args):
        super(PointPillarCoAlign, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = ResBEVBackbone(args['res_bev_backbone'], 64)

        self.voxel_size = args['voxel_size']

        # multiscale fusion network modules
        self.fusion_net = nn.ModuleList()
        for i in range(len(args['res_bev_backbone']['layer_nums'])):
             # If proj_first = True, no actual warping is performed
            self.fusion_net.append(Att_w_Warp(args['res_bev_backbone']['num_filters'][i]))
        self.out_channel = sum(args['res_bev_backbone']['num_upsample_filter'])

        self.compression = False
        if 'compression' in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(args['res_bev_backbone']['num_filters'][0], args['compression'])

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        # get affine matrix for feature warping
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0.
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']

        # multiscale fusion
        # The first scale feature 'feature_list[0]' for transmission. Default 100*352*64*32/1000000 = 72.0896
        feature_list = []
        feature_list.append(self.backbone.get_layer_i_feature(spatial_features, layer_i=0)) 
        if self.compression:
            feature_list[0] = self.naive_compressor(feature_list[0])
        for i in range(1, self.backbone.num_levels):
            feature_list.append(self.backbone.get_layer_i_feature(feature_list[i-1], layer_i=i))
                            

        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, normalized_affine_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 

        # downsample feature to reduce memory
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict