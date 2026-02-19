# There is no additional fucntion here.
from src.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor


__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        train=train
    )

    return anchor_generator
