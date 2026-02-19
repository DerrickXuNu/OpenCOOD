# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution



def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='V2V',
    packages=find_packages(),
    license='MIT',
    author='Mohammad Amirifard',
    description='An opensource pytorch framework for autonomous driving '
                'cooperative detection',
    long_description="You can find more details about src at ",
    install_requires=_read_requirements_file(),
)
