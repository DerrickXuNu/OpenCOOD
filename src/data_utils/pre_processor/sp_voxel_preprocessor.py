# There is no additional loss here.

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch
from cumm import tensorview as tv
from src.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    """
    Convert raw point cloud data into a voxel grid representation.
    his involves dividing the 3D space into a grid of cubes (voxels)
    and assigning point cloud data to these voxels based on their spatial locations

    """

    def __init__(self, preprocess_params, train):
        """
        Initialize a voxel_generator, creating voxels from points
        """
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            # spconv v2.x
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
            self.spconv = 2
            
        self.lidar_range = self.params['cav_lidar_range']   #[-140.8, -40, -3, 140.8, 40, 1]
        self.voxel_size = self.params['args']['voxel_size'] # [0.4, 0.4, 4]
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        #grid_size-->[704,200,1]
        self.grid_size = np.round(grid_size).astype(np.int64) # Remove float

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
        else:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels
            )

    def preprocess(self, pcd_np):
        """
        Args:
            pcd_np: pcd pints which is a np.array (N,4). N= number of pints
        Return:
            data_dict: inlcudes voxel_features with shape(M,32,4), voxel_coords(M,3) and voxel_num_points(M,). M = number of Voxels
        
        """
        data_dict = {}
        if self.spconv == 1:
            voxel_output = self.voxel_generator.generate(pcd_np)
        else:
            pcd_tv = tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Args:
            batch : list
            This a List of dictionary. Each dictionary represent a single frame.
            Each dictionary has 3 keys: voxel_features,voxel_num_points,voxel_coords

        Returns:
            a dictionary which is 
                                {
                                     'voxel_features': voxel_features, # This voxel_features is a concatentae of voxel_features of all batch item
                                     'voxel_coords': voxel_coords,     # The same
                                     'voxel_num_points': voxel_num_points # The same
                                }

            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns:
            a dictionary which is 
                                {
                                     'voxel_features': voxel_features, # This voxel_features is a concatentae of voxel_features of all batch item
                                     'voxel_coords': voxel_coords,     # The same
                                     'voxel_num_points': voxel_num_points # The same
                                }
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}
