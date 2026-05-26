from glob import glob
from setuptools import find_packages, setup

package_name = 'franka_wuji_vr_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mds',
    maintainer_email='mds@example.com',
    description='Meta Quest hand-tracking teleoperation bridge for Franka Panda with Wuji hand.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hts_franka_wuji_teleop = franka_wuji_vr_teleop.hts_franka_wuji_teleop:main',
        ],
    },
)
