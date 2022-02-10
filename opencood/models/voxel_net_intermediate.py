"""
VoxelNet for intermediate fusion
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.models.sub_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder


# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


class NaiveFusion(nn.Module):

    def __init__(self):
        super(NaiveFusion, self).__init__()
        self.conv1 = Conv2d(128 * 5, 256, 3, 1, 1,
                            batch_norm=False, bias=False)
        self.conv2 = Conv2d(256, 128, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self, cin, cout, T):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)
        self.T = T

    def forward(self, x, mask):
        # point-wise feature
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, self.T, 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self, T):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32, T)
        self.vfe_2 = VFE(32, 128, T)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


# Region Proposal Network
class RPN(nn.Module):
    def __init__(self, anchor_num=2):
        super(RPN, self).__init__()
        self.anchor_num = anchor_num

        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),
                                      nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, self.anchor_num, 1, 1, 0,
                                 activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * self.anchor_num, 1, 1, 0,
                               activation=False, batch_norm=False)

    def forward(self, x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0, x_1, x_2), 1)
        return self.score_head(x), self.reg_head(x)


class VoxelNetIntermediate(nn.Module):
    def __init__(self, args):
        super(VoxelNetIntermediate, self).__init__()
        self.svfe = PillarVFE(args['pillar_vfe'],
                              num_point_features=4,
                              voxel_size=args['voxel_size'],
                              point_cloud_range=args['lidar_range'])
        self.cml = CML()
        self.fusion_net = AttFusion(128)
        self.rpn = RPN(args['anchor_num'])

        self.N = args['N']
        self.D = args['D']
        self.H = args['H']
        self.W = args['W']
        self.T = args['T']
        self.anchor_num = args['anchor_num']

        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.compression_layer = AutoEncoder(128, args['compression'])

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(
            torch.zeros(dim, self.N, self.D, self.H, self.W).cuda())

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2],
        coords[:, 3]] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def regroup(self, dense_feature, record_len):
        """
        Regroup the data based on the record_len.

        Parameters
        ----------
        dense_feature : torch.Tensor
            N, C, H, W
        record_len : list
            [sample1_len, sample2_len, ...]

        Returns
        -------
        regroup_feature : torch.Tensor
            B, 5C, H, W
        """
        cum_sum_len = list(np.cumsum(record_len))
        split_features = torch.tensor_split(dense_feature,
                                            cum_sum_len[:-1])
        regroup_features = []

        for split_feature in split_features:
            # M, C, H, W
            feature_shape = split_feature.shape

            # the maximum M is 5 as most 5 cavs
            padding_len = 5 - feature_shape[0]
            padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                         feature_shape[2], feature_shape[3])
            padding_tensor = padding_tensor.to(split_feature.device)

            split_feature = torch.cat([split_feature, padding_tensor],
                                      dim=0)

            # 1, 5C, H, W
            split_feature = split_feature.view(-1,
                                               feature_shape[2],
                                               feature_shape[3]).unsqueeze(0)
            regroup_features.append(split_feature)

        # B, 5C, H, W
        regroup_features = torch.cat(regroup_features, dim=0)

        return regroup_features

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        if voxel_coords.is_cuda:
            record_len_tmp = record_len.cpu()

        record_len_tmp = list(record_len_tmp.numpy())

        self.N = sum(record_len_tmp)

        # feature learning network
        vwfs = self.svfe(batch_dict)['pillar_features']

        voxel_coords = torch_tensor_to_numpy(voxel_coords)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)

        # convolutional middle network
        vwfs = self.cml(vwfs)
        # convert from 3d to 2d N C H W
        vmfs = vwfs.view(self.N, -1, self.H, self.W)

        # compression layer
        if self.compression:
            vmfs = self.compression_layer(vmfs)

        # information naive fusion
        vmfs_fusion = self.fusion_net(vmfs, record_len)

        # region proposal network
        # merge the depth and feature dim into one, output probability score
        # map and regression map
        psm, rm = self.rpn(vmfs_fusion)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets.late_fusion_dataset import \
        LateFusionDataset

    params = load_yaml('../hypes_yaml/voxelnet_late_fusion.yaml')
    opencda_dataset = LateFusionDataset(params, visualize=True)
    data_loader = DataLoader(opencda_dataset,
                             batch_size=params['train_params']['batch_size'],
                             num_workers=4,
                             collate_fn=opencda_dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False)
    model = VoxelNetIntermediate(params['model']['args'])
    model.cuda()

    for j, batch_data in enumerate(data_loader):
        output_dict = model(batch_data['ego']['processed_lidar'])
        print('debug')
