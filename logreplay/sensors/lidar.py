import weakref

import carla
import os
import open3d as o3d
import numpy as np
from logreplay.sensors.base_sensor import BaseSensor

class Lidar(BaseSensor):

    def __init__(self, agent_id, vehicle, world, config, global_position):
        super().__init__(agent_id, vehicle, world, config, global_position)

        if vehicle is not None:
            world = vehicle.get_world()

        self.vehicle = vehicle
        self.agent_id = agent_id
        self.name = 'lidar'

        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        blueprint.set_attribute('upper_fov', str(config['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config['lower_fov']))
        blueprint.set_attribute('channels', str(config['channels']))
        blueprint.set_attribute('range', str(config['range']))
        blueprint.set_attribute('points_per_second', str(config['points_per_second']))
        blueprint.set_attribute('rotation_frequency', str(config['rotation_frequency']))

        if vehicle is None:
            spawn_point = self.spawn_point_estimation(None, global_position)
            self.sensor = world.spawn_actor(blueprint, spawn_point)
        else:
            self.relative_position = config['relative_pose']
            self.relative_position_id = ['front', 'right', 'left', 'back']
            spawn_point = self.spawn_point_estimation(self.relative_position, None)
            self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
            self.name += str(self.relative_position)

        # lidar data
        self.thresh = config['thresh']
        self.data = None
        self.timestamp = None
        self.frame = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: Lidar._on_data_event(weak_self, event))
    
    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar  method"""
        self = weak_self()
        if not self:
            return

        # retrieve the raw lidar data and reshape to (N, 4)
        data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        # (x, y, z, intensity)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)

        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])

            carla_rotation = carla.Rotation(pitch=pitch, yaw=global_position[3], roll=0)

        else:

            if relative_position == 'front':
                carla_location = carla.Location(x=carla_location.x + 2.5,
                                                y=carla_location.y,
                                                z=carla_location.z + 1.0)
                yaw = 0

            elif relative_position == 'right':
                carla_location = carla.Location(x=carla_location.x + 0.0,
                                                y=carla_location.y + 0.3,
                                                z=carla_location.z + 1.8)
                yaw = 100

            elif relative_position == 'left':
                carla_location = carla.Location(x=carla_location.x + 0.0,
                                                y=carla_location.y - 0.3,
                                                z=carla_location.z + 1.8)
                yaw = -100
            else:
                carla_location = carla.Location(x=carla_location.x - 2.0,
                                                y=carla_location.y,
                                                z=carla_location.z + 1.5)
                yaw = 180

            carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)

        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    def data_dump(self, output_root, cur_timestamp):

        while not hasattr(self, 'data') or self.data is None:
            continue

        point_cloud = self.data
        point_xyz = point_cloud[:, :-1]
        point_intensity = point_cloud[:, -1]
        point_intensity = np.c_[
            point_intensity,
            np.zeros_like(point_intensity),
            np.zeros_like(point_intensity)
        ]

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(point_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

        # write to pcd file
        if self.vehicle is None:
            pcd_name = f'{cur_timestamp}.pcd'
        else:
            pose_id = self.relative_position_id.index(self.relative_position)
            pcd_name = f'{cur_timestamp}_lidar{pose_id}.pcd'

        o3d.io.write_point_cloud(os.path.join(output_root,
                                              pcd_name),
                                 pointcloud=o3d_pcd,
                                 write_ascii=True)
