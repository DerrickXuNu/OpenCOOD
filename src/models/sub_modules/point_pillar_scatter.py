import torch
import torch.nn as nn

#? This class is responsible for converting pillar features into a 2D BEV (Bird's Eye View) feature map
#! Goal: 
""" 1_Take pillar features (processed by PillarVFE)
    2_Convert them into a 2D BEV feature map
    3_Create a format suitable for 2D CNN processing
"""

class PointPillarScatter(nn.Module):
    """
    Stacking.
    """
    #// Step1: Initialization
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']
        assert self.nz == 1 # Since BEV is 2D

    def forward(self, batch_dict):

        #// Step2: Takes pillar features and their coords
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']
        
        #// Step3: Batch Processing
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            #// Step4: Feature Scattering
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            indices = this_coords[:, 1] + \
                      this_coords[:, 2] * self.nx + \
                      this_coords[:, 3]
            indices = indices.type(torch.long)

            #// Step5: Feature Placement:
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        
        #// Step6: Stack all batch items
        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

