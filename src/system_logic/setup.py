from setuptools import find_packages, setup

package_name = 'system_logic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'target_detector = system_logic.target_detector_node:main',
			'target_subscriber = system_logic.target_subscriber_node:main',
            'picamera_node = system_logic.picamera_node:main',
        ],
    },
)
