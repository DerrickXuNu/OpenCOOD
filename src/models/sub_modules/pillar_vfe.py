"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# In thsi file, you can see two classes:
#// 1: PFNLayer: PFN stands for Pillar Feature Networ Shown in Pint-Pillar paper.
#// 2: PillarVFE: VFE stands for Voxel Feature Encoder shown in Voxel_Net paper.

class PFNLayer(nn.Module):

    #? This class has PFN structure and used in the PillarVFE
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        #? Set maximu size of batch to 50,000
        self.part = 50000
        print("self.part set to:",self.part)

    def forward(self, inputs):
        """
        Args: 
            inputs: It (should) or (Must) be Stacked Pillars
        
        Return:
            x_concatenated: This is Learned Feature shown in Voxel-Net paper. It must have 2D Dimension.
        """


        #print('Input shape to PFn:',inputs.shape)
        #// Step 1: Run Linear Layer
        #? Create a condition to check input shape > 50,000. If so, for more stability we run nn.linear on smaller parts and
        #? then concatenated the result in x variable.
        if inputs.shape[0] > self.part:
            #! nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        
        #// Step2: Run Batch-Norm on it
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True

        #// Step3: Pass it to Relu activation
        x = F.relu(x)

        #// Step4: Run Max pooling to have Learned feature in 2D dimension.
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            #print('Output shape of PFn:',x_max.shape)
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            #print('Output shape of PFn:',x_concatenated.shape)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        
        # num_point_features = 4
        #// Step1: Read parameters specified in the model_cfg config
        self.model_cfg = model_cfg
        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']
        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        self.num_filters = self.model_cfg['num_filters']

        #! xyz are the direction of a point.
        #! In the base article we are working, use_absolute_xyz is 3
        num_point_features += 6 if self.use_absolute_xyz else 3
        # print(num_point_features) -->4+6 = 10
        if self.with_distance:
            num_point_features += 1
            
        #! number of filters is [64] and num_point_features is X+3. So num_filters=[64,X+3]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        #print('Num_filters which is used for pillar VFE is:',num_filters)
        # num_filters = [10, 64]
        
        #// Step2: RunPFN class for different in and out filters
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]

            pfn_layers.append(
                PFNLayer(in_channels=in_filters, out_channels=out_filters, use_norm=self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        #! We have two offsets:
        #! 1: Pillar offset: the position of points relative to the center of Pillars
        #! 2: Cluster offset: the position of points relative to clusters (summarize all points positions)
        
        #? The following code is for
        #// Pillar Offset

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    #? If pillars or samples reach too much data, do these limitation: 
    #! 1:Limitation on non-empty pillars in each sample
    #! 2:Limitation on number of points in each pillar

    #? If pillars or samples don't reach enough data, do paddaing.
    def get_paddings_indicator(actual_num, max_num, axis=0):
        """
        Args:
            actual_num: The actual number of points in each pillar
            max_num: The maximum number of points we want to allow per pillare
        
        Return: 
            paddings_indicator:A list of Flase and True. e.g: [False,False,True]. True says it is 0 added by padding
        """

        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):

        """
        Args: 
            batch_dict: A List of voxel_features, voxel_num_points,voxel_coords
        Return:
            batch_dict: Including an additional list called feature.

        """

        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']
        #? The following code is for:
        #// Cluster offset.
        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]

        #! Do zero padding function
        mask = self.get_paddings_indicator(actual_num=voxel_num_points,max_num=voxel_count,
                                           axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        #! Do network layer after extracting the features.
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()


        batch_dict['pillar_features'] = features
        #print("features:",features.shape)
        return batch_dict
