import os
import sys
from collections import OrderedDict

import carla
import numpy as np

from logreplay.assets.utils import find_town
from logreplay.assets.presave_lib import bcolors


class SceneManager:
    """
    Manager for each scene for spawning, moving and destroying.

    Parameters
    ----------
    folder : str
        The folder to the current scene.

    scene_name : str
        The scene's name.

    collection_params : dict
        The collecting protocol information.
    """
    def __init__(self, folder, scene_name, collection_params):
        self.town_name = find_town(scene_name)
        self.collection_params = collection_params

        # at least 1 cav should show up
        cav_list = sorted([x for x in os.listdir(folder)
                           if os.path.isdir(
                os.path.join(folder, x))])
        assert len(cav_list) > 0

        self.database = OrderedDict()
        # we want to save timestamp as the parent keys for cavs
        cav_sample = cav_list[0]

        yaml_files = \
            sorted([os.path.join(cav_sample, x)
                    for x in os.listdir(cav_sample) if
                    x.endswith('.yaml')])
        self.timestamps = self.extract_timestamps(yaml_files)

        # loop over all timestamps
        for timestamp in self.timestamps:
            self.database[timestamp] = OrderedDict()
            # loop over all cavs
            for (j, cav_id) in enumerate(cav_list):
                self.database[timestamp][cav_id] = OrderedDict()
                cav_path = os.path.join(folder, cav_id)

                yaml_file = os.path.join(cav_path,
                                         timestamp + '.yaml')
                self.database[timestamp][cav_id]['yaml'] = \
                    yaml_file

        # this is used to dynamically save all information of the objects
        self.veh_dict = OrderedDict()
        # used to count timestamp
        self.cur_count = 0

    def start_simulator(self):
        """
        Connect to the carla simulator for log replay.
        """
        simulation_config = self.collection_params['world']

        # simulation sync mode time step
        fixed_delta_seconds = simulation_config['fixed_delta_seconds']
        weather_config = simulation_config[
            'weather'] if "weather" in simulation_config else None

        # setup the carla client
        self.client = \
            carla.Client('localhost', simulation_config['client_port'])
        self.client.set_timeout(10.0)

        # load the map
        if self.town_name != 'Culver_City':
            try:
                self.world = self.client.load_world(self.town_name)
            except RuntimeError:
                print(
                    f"{bcolors.FAIL} %s is not found in your CARLA repo! "
                    f"Please download all town maps to your CARLA "
                    f"repo!{bcolors.ENDC}" % self.town_name)
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit('World loading failed')

        # setup the new setting
        self.origin_settings = self.world.get_settings()
        new_settings = self.world.get_settings()

        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = fixed_delta_seconds

        self.world.apply_settings(new_settings)
        # set weather if needed
        if weather_config is not None:
            weather = self.set_weather(weather_config)
            self.world.set_weather(weather)
        # get map
        self.carla_map = self.world.get_map()

    def tick(self):
        """
        Spawn the vehicle to the correct position
        """
        cur_timestamp = self.timestamps[self.cur_count]
        cur_database = self.database[cur_timestamp]

        for cav_id, cav_content in cur_database.items():
            if cav_id not in self.veh_dict:
                self.spawn_cav(cav_id, cav_content)

    def spawn_cav(self, cav_id, cav_content):
        """
        Spawn the cav based on current content.

        Parameters
        ----------
        cav_id : str
            The saved cav_id.

        cav_content : dict
            The information in the cav's folder.
        """
        # cav always use lincoln
        model = 'vehicle.lincoln.mkz2017'
        cur_pose = cav_content['true_ego_pos']
        # convert to carla needed format
        cur_pose = carla.Transform(carla.Location(x=cur_pose[0],
                                                  y=cur_pose[1],
                                                  z=cur_pose[2]),
                                   carla.Rotation(roll=cur_pose[3],
                                                  yaw=cur_pose[4],
                                                  pitch=cur_pose[5]))


    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def set_weather(weather_settings):
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings['sun_altitude_angle'],
            cloudiness=weather_settings['cloudiness'],
            precipitation=weather_settings['precipitation'],
            precipitation_deposits=weather_settings['precipitation_deposits'],
            wind_intensity=weather_settings['wind_intensity'],
            fog_density=weather_settings['fog_density'],
            fog_distance=weather_settings['fog_distance'],
            fog_falloff=weather_settings['fog_falloff'],
            wetness=weather_settings['wetness']
        )
        return weather


