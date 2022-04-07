# -*- coding: utf-8 -*-

"""Rasterization drawing functions
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import cv2

# sub-pixel drawing precision constants
CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]
INTERPOLATION_POINTS = 20

AGENT_COLOR = (255, 255, 255)
ROAD_COLOR = (255, 255, 255)
ROAD_COLOR_VIS =  (17, 17, 31)
Lane_COLOR = {'normal': (255, 217, 82),
              'red': (255, 0, 0),
              'yellow': (255, 255, 0),
              'green': (0, 255, 0)}


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int)
    return coords


def draw_agent(agent_list, image):
    """
    Draw agent mask on image.

    Parameters
    ----------
    agent_list : list
        The agent corner list.

    image : np.ndarray
        The image to be drawn.

    Returns
    -------
    Drawn image.
    """
    for agent_corner in agent_list:
        agent_corner = agent_corner.reshape(-1, 2)
        cv2.fillPoly(image, [agent_corner], AGENT_COLOR,
                     **CV2_SUB_VALUES)
    return image


def draw_road(lane_area_list, image, visualize=False):
    """
    Draw poly for road.

    Parameters
    ----------
    visualize : bool
        If set to true, the road segment will be black color.

    lane_area_list : list
        List of lane coordinates

    image : np.ndarray
        image to be drawn

    Returns
    -------
    drawed image.
    """
    color = ROAD_COLOR if not visualize else ROAD_COLOR_VIS

    for lane_area in lane_area_list:
        lane_area = lane_area.reshape(-1, 2)
        cv2.fillPoly(image, [lane_area], color,
                     **CV2_SUB_VALUES)
    return image

def road_exclude(static_road):
    """
    Exclude the road segment that is not connected to the ego vehicle
    position.

    Parameters
    ----------
    static_road : np.ndarray
        The static bev map with road segment.

    Returns
    -------
    The road without unrelated road.
    """
    binary_bev = cv2.cvtColor(static_road, cv2.COLOR_BGR2GRAY)
    _, label, stats, _ = cv2.connectedComponentsWithStats(binary_bev)

    ego_label = label[static_road.shape[0] // 2, static_road.shape[1] // 2]
    static_road[label != ego_label] = 0

    return static_road


def draw_lane(lane_area_list, lane_type_list, image):
    """
    Draw lanes on image (polylines).

    Parameters
    ----------
    lane_area_list : list
        List of lane coordinates

    lane_type_list : list
        List of lane types, normal, red, green or yellow.

    image : np.ndarray
        image to be drawn

    Returns
    -------
    drawed image.
    """
    for (lane_area, lane_type) in zip(lane_area_list, lane_type_list):
        cv2.polylines(image, lane_area, False, Lane_COLOR[lane_type],
                      **CV2_SUB_VALUES)

    return image