import os
import numpy as np
import cv2
import carla
import weakref
from logreplay.sensors.base_sensor import BaseSensor

class Camera(BaseSensor):

    """
    Camera manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : str
        Indicates the sensor is a front or rear camera. option:
        front, left, right.

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, agent_id, vehicle, world, config, global_position=None):

        super().__init__(agent_id, vehicle, world, config, global_position)

        if vehicle is not None:
            world = vehicle.get_world()

        self.vehicle = vehicle
        self.agent_id = agent_id
        self.name = 'camera'

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', str(config['fov']))
        blueprint.set_attribute('image_size_x', str(config['image_size_x']))
        blueprint.set_attribute('image_size_y', str(config['image_size_y']))

        self.relative_position = config['relative_pose']
        self.relative_position_id = ['front', 'right', 'left', 'back', 'bev']

        if vehicle is None:
            spawn_point = self.spawn_point_estimation(None, global_position)
            self.sensor = world.spawn_actor(blueprint, spawn_point)
        else:
            spawn_point = self.spawn_point_estimation(self.relative_position, None)
            self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
            self.name += str(self.relative_position)

        self.image = None
        self.timstamp = None
        self.frame = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: Camera._on_data_event(weak_self, event))

        # camera attributes
        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)

        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            # pitch = -35
            carla_rotation = carla.Rotation(roll=0, yaw=global_position[3], pitch=pitch)
        
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
            elif relative_position == 'back':
                carla_location = carla.Location(x=carla_location.x - 2.0,
                                                y=carla_location.y,
                                                z=carla_location.z + 1.5)
                yaw = 180
            else:
                raise NotImplementedError

            carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_data_event(weak_self, event):
        """CAMERA  method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        # we need to remove the alpha channel
        image = image[:, :, :3]

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp

    def data_dump(self, output_root, cur_timestamp):

        while not hasattr(self, 'image') or self.image is None:
            continue

        if self.vehicle is None: ###
            image_name = f'{cur_timestamp}_camera.png'
        else:

            pose_id = self.relative_position_id.index(self.relative_position)
            image_name = f'{cur_timestamp}_camera{pose_id}.png'
            
        cv2.imwrite(os.path.join(output_root, image_name), self.image)
