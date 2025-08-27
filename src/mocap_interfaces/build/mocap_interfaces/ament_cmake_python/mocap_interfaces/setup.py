from setuptools import find_packages
from setuptools import setup

setup(
    name='mocap_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('mocap_interfaces', 'mocap_interfaces.*')),
)
