from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ieerasufcg',
    maintainer_email='carlos.eduardo.carvalho@ee.ufcg.edu.br',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'udp_bridge_node = ros2_bridge.udp_bridge_node:main',
        ],
    },
)