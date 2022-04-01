import torch
from torch import nn
import numpy as np

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.vsa import VoxelSetAbstraction
from opencood.models.sub_modules.roi_head import RoIHead
from opencood.models.sub_modules.matcher import Matcher
from opencood.data_utils.post_processor.ciassd_postprocessor import CiassdPostprocessor


class FPVRCNN(nn.Module):
    def __init__(self, args):
        super(FPVRCNN, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'], args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv']['num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])
        self.ssfa = SSFA(args['ssfa'])
        self.head = Head(**args['head'])
        self.post_processor = CiassdPostprocessor(args['post_processer'], train=True)
        self.vsa = VoxelSetAbstraction(args['vsa'], args['voxel_size'], args['lidar_range'],
                                       num_bev_features=128, num_rawpoint_features=3)
        self.matcher = Matcher(args['matcher'], args['lidar_range'])
        self.roi_head = RoIHead(args['roi_head'])

    def forward(self, batch_dict):
        batch_dict['batch_size'] = int(batch_dict['record_len'].sum())
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        out = self.ssfa(batch_dict['processed_lidar']['spatial_features'])
        batch_dict['preds_dict_stage1'] = self.head(out)
        data_dict, output_dict = {}, {}
        data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict
        # The data structure is a little confusing, same data pointer is passed two times,
        # because it should match the format of the main framwork
        pred_box3d_list, scores_list = self.post_processor.post_process(data_dict, output_dict)
        batch_dict['det_boxes'] = pred_box3d_list
        batch_dict['det_scores'] = scores_list
        if pred_box3d_list is not None:
            batch_dict = self.vsa(batch_dict)
            batch_dict = self.matcher(batch_dict)
            batch_dict = self.roi_head(batch_dict)

        return batch_dict


if __name__=="__main__":
    model = SSFA(None)
    print(model)