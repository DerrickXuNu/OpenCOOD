
import numpy as np
import torch
import torch.nn as nn
from opencood.models.sub_modules.resblock import ResNetLayers

class ResBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        self.model_cfg = model_cfg

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        self.resnet = ResNetLayers(layer_nums,
                                     layer_strides,
                                     num_filters,
                                     inplanes = input_channels)

        num_levels = len(layer_nums)
        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        This can be used in single agent detection (late fusion)

        For intermediate fusion, use get_multiscale_feature and 
        decode_multiscale_feature.
        """
        spatial_features = data_dict['spatial_features']

        x = self.resnet(spatial_features)  # tuple of features
        ups = []

        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict


    def get_multiscale_feature(self, spatial_features):
        """
        before multiscale intermediate fusion
        """
        x = self.resnet(spatial_features)  # tuple of features
        return x

    def decode_multiscale_feature(self, x):
        """
        after multiscale interemediate fusion
        """
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x
        
    def get_layer_i_feature(self, spatial_features, layer_i):
        """
        before multiscale intermediate fusion
        """
        return eval(f"self.resnet.layer{layer_i}")(spatial_features)  # tuple of features
    