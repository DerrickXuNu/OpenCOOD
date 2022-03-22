from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.post_processor.bev_postprocessor import BevPostprocessor
from opencood.data_utils.post_processor.ciassd_postprocessor import CiassdPostprocessor

__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'BevPostprocessor': BevPostprocessor,
    'CiassdPostprocessor': CiassdPostprocessor,
}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor', 'BevPostprocessor', 'CiassdPostprocessor']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        train=train
    )

    return anchor_generator
