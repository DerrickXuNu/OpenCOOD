import numpy as np

from logreplay.assets.presave_lib import TOWN_DICTIONARY, BLUE_PRINT_LIB


def find_town(scenario_name):
    """
    Given the scenario name, find the corresponding town name,

    Parameters
    ----------
    scenario_name : str
        The scenario name.

    Returns
    -------
    The corresponding town's name.
    """
    return TOWN_DICTIONARY[scenario_name]


def find_blue_print(extent):
    pass