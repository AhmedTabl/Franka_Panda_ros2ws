# Franka Panda + surgical hand in MuJoCo (slice 7, Franka attachment).
#
# Wraps franka_bringup's franka_sim.launch.py using its existing CUSTOM
# end-effector hook (no franka_description edits needed):
#   - robot_description: panda + ORCA URDF + the hand's ros2_control block
#     (surgical_hand_ee.urdf.xacro)
#   - MuJoCo scene: generated panda_orca_ng/scene_orca_ng (see
#     surgical_hand_description/scripts/generate_mujoco_scenes.py)
#   - hand controller: spawned here with its type and joint list supplied
#     via spawner flags, so franka_bringup's controllers YAML stays untouched.
#
#   ros2 launch surgical_hand_bringup surgical_hand_franka_mujoco.launch.py
#   ros2 run surgical_hand_skills hand_pose_cli close   # hand
#   ...plus all normal franka_sim controllers for the arm.
#
# The flange mount transform is an untuned placeholder; keep the URDF args
# below and MOUNT_* in generate_mujoco_scenes.py in sync when tuning.

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ATTACHMENT_XYZ = '0 0 0'
ATTACHMENT_RPY = '1.5708 0 0'  # matches MOUNT_EULER in generate_mujoco_scenes.py


def _setup(context, *args, **kwargs):
    description_pkg = get_package_share_directory('surgical_hand_description')
    franka_bringup_pkg = get_package_share_directory('franka_bringup')

    # Controller parameters (type comes via spawner --controller-type; the
    # joint list is generated from hand_joints.yaml, single source of truth).
    config_file = os.path.join(description_pkg, 'config', 'hand_joints.yaml')
    with open(config_file, 'r') as f:
        joint_names = [entry['name'] for entry in yaml.safe_load(f)['joints']]
    controller_params = {
        'hand_joint_position_controller': {
            'ros__parameters': {'interface_name': 'position', 'joints': joint_names}
        }
    }
    fd, params_path = tempfile.mkstemp(prefix='surgical_hand_franka_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(controller_params, f)

    franka_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(franka_bringup_pkg, 'launch', 'sim', 'franka_sim.launch.py')),
        launch_arguments={
            'hand': 'true',
            'end_effector': 'custom',
            'end_effector_urdf': os.path.join(description_pkg, 'robots',
                                              'surgical_hand_ee.urdf.xacro'),
            'end_effector_base_link': 'ForeArmStructure-Model_e18f2368',
            'attachment_xyz': ATTACHMENT_XYZ,
            'attachment_rpy': ATTACHMENT_RPY,
            'end_effector_scene': os.path.join(description_pkg, 'mujoco',
                                               'scene_orca_ng.xml'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'no_render': LaunchConfiguration('no_render'),
        }.items(),
    )

    hand_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['hand_joint_position_controller',
                   '--controller-manager', '/controller_manager',
                   '--controller-manager-timeout', '120',
                   '--controller-type', 'forward_command_controller/ForwardCommandController',
                   '--param-file', params_path],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return [franka_sim, hand_controller_spawner]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='Start RViz alongside the sim'),
        DeclareLaunchArgument('no_render', default_value='false',
                              description='Run the MuJoCo server headless'),
        OpaqueFunction(function=_setup),
    ])
