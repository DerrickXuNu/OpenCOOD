# There is no unsuful function here.
"""
done
"""
# ------------------------------------------------------------------------------
# This module initializes dataset classes and enables users to instantiate
# different dataset objects based on their configuration.
# ------------------------------------------------------------------------------

from src.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset


# Mapping of dataset class names to their corresponding classes for dynamic instantiation
__all__ = {
    'IntermediateFusionDataset': IntermediateFusionDataset,
}

# Ground truth range:
# X-axis range: [-140, 140]
# Y-axis range: [-40, 40]
# Z-axis range: [-3, 1]
# These bounds are based on the OPV2V dataset specifications
GT_RANGE = [-140, -40, -3, 140, 40, 1]

# Communication range between connected autonomous vehicles (CAVs)
# Defined by the OPV2V benchmark
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    """
    Instantiate and return a dataset object based on the configuration.

    Parameters
    ----------
    dataset_cfg : dict
        Configuration dictionary specifying dataset parameters, including
        the core fusion method.
    
    visualize : bool, optional
        Flag to enable data visualization during loading (default is False).

    train : bool, optional
        Indicates whether the dataset is used for training or evaluation (default is True).

    Returns
    -------
    dataset : object
        An instance of the selected dataset class.
    
    Raises
    ------
    AssertionError
        If the specified dataset name is not registered in the __all__ mapping.
    """
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = (
        f"{dataset_name} is not found. "
        f"Please add your dataset class to src/data_utils/datasets/init.py"
    )

    assert dataset_name in __all__, error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )
    
    return dataset
