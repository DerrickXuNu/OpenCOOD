import os
from collections import OrderedDict

from opencood.hypes_yaml.yaml_utils import load_yaml
from logreplay.scenario_manager.scene_manager import SceneManager


class ScenariosManager:
    """
    Format all scenes in a structured way.

    Parameters
    ----------
    scenario_params: dict
        Overall parameters for the replayed scenes.

    Attributes
    ----------

    """

    def __init__(self, scenario_params):
        # this defines carla world sync mode, weather, town name, and seed.
        self.scene_params = scenario_params

        # e.g. /opv2v/data/train
        root_dir = self.scene_params['root_dir']

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_database = OrderedDict()

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            scene_name = os.path.split(scenario_folder)[-1]
            self.scenario_database.update({scene_name: OrderedDict()})

            # load the collection yaml file
            protocol_yml = [x for x in os.listdir(scenario_folder)
                            if x.endswith('.yaml')]
            collection_params = load_yaml(protocol_yml)

            # create the corresponding scene manager
            cur_sg = SceneManager(scenario_folder,
                                  scene_name,
                                  collection_params)
            self.scenario_database[scene_name].update({'scene_manager':
                                                       cur_sg})






