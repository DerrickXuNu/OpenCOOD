# -*- coding: utf-8 -*-
"""
Sensor Manager for each cav
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import importlib


class SensorManager:
    """
    The manager controls all sensor data streaming and dumping for each cav.
    todo: rsu not considered yet.
    Parameters
    ----------
    agent_id : str
        The cav's original id.

    vehicle_content : dict
        The content of the cav.

    world : carla.World
        Carla simulation server object.

    config_yaml : dict
        Configurations for the sensor manager.

    Attributes
    ----------
    sensor_list : list
        A list of sensors to dump/visualize data.
    """
    def __init__(self, agent_id, vehicle_content, world, config_yaml):
        self.agent_id = agent_id
        self.vehicle = vehicle_content['actor']
        self.world = world
        self.sensor_list = []

        for sensor_content in config_yaml['sensor_list']:
            sensor = None
            sensor_name = sensor_content['name']

            # find the relative path
            sensor_filename = "logreplay.sensors." + sensor_name
            sensor_lib = importlib.import_module(sensor_filename)
            target_sensor_name = sensor_name.replace('_', '')

            # the sensor corresponding class has to have the same
            # name pattern as the file
            for name, cls in sensor_lib.__dict__.items():
                if name.lower() == target_sensor_name.lower():
                    sensor = cls

            assert sensor is not None, 'The sensor class name has to be the' \
                                       'same as the file name. e.g. ' \
                                       'bev_semantic_camera -> ' \
                                       'BevSemanticCamera'
            # todo: rsu is not considered yet
            sensor_instance = sensor(self.agent_id,
                                     self.vehicle,
                                     self.world,
                                     sensor_content['args'],
                                     None)
            self.sensor_list.append(sensor_instance)

    def run_step(self):
        for sensor_instance in self.sensor_list:
            sensor_instance.visualize_data()
            # todo add dump data later

    def destroy(self):
        for sensor_instance in self.sensor_list:
            sensor_instance.destroy()
