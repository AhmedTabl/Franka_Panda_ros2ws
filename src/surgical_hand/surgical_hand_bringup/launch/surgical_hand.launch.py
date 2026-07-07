# Standalone surgical hand bringup (no Franka arm).
#
# Backends:
#   backend:=mock (default) - ros2_control mock hardware; no physics, no
#                             external processes. This is the "fake hand"
#                             vertical slice and works on any machine.
#   backend:=mujoco         - reserved; the MuJoCo-backed hand launch will be
#                             added in a later slice (it must run inside the
#                             mujoco_ros2_control server, like the Wuji hand).
#   backend:=real           - refused here on purpose; real hardware bringup
#                             will live in a separate, safety-gated launch.
#
# Verify with:
#   ros2 launch surgical_hand_bringup surgical_hand.launch.py
#   ros2 topic echo /joint_states --once
#   ros2 run surgical_hand_skills hand_pose_cli close
#
# The launch file is Python because ROS 2 launch files are Python by
# convention; all runtime functionality stays in C++.

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_hand_joint_names(config_file):
    with open(config_file, 'r') as f:
        hand_config = yaml.safe_load(f)
    return [entry['name'] for entry in hand_config['joints']]


def _write_generated_controller_params(joint_names):
    """Write a params file holding the controller joint list.

    Generated from hand_joints.yaml at launch time so the joint set has a
    single source of truth (see surgical_hand_controllers.yaml).
    """
    params = {
        'hand_joint_position_controller': {
            'ros__parameters': {
                'joints': joint_names,
            }
        }
    }
    fd, path = tempfile.mkstemp(prefix='surgical_hand_joints_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(params, f)
    return path


def _setup(context, *args, **kwargs):
    backend = LaunchConfiguration('backend').perform(context)
    config_file = LaunchConfiguration('config_file').perform(context)

    if backend == 'real':
        raise RuntimeError(
            'backend:=real is not supported by this launch file. Real '
            'hardware bringup will be a separate, safety-gated launch file.')
    if backend == 'mujoco':
        raise RuntimeError(
            'backend:=mujoco is not wired up yet. The MuJoCo hand backend '
            'must run inside the mujoco_ros2_control server (later slice).')
    if backend != 'mock':
        raise RuntimeError(f'Unknown backend "{backend}". Use backend:=mock.')

    description_pkg = get_package_share_directory('surgical_hand_description')
    bringup_pkg = get_package_share_directory('surgical_hand_bringup')

    xacro_file = os.path.join(description_pkg, 'robots', 'surgical_hand.urdf.xacro')
    controllers_yaml = os.path.join(bringup_pkg, 'config', 'surgical_hand_controllers.yaml')

    import xacro
    robot_description = xacro.process_file(
        xacro_file,
        mappings={'backend': backend, 'config_file': config_file},
    ).toprettyxml(indent='  ')

    joint_names = _load_hand_joint_names(config_file)
    generated_params = _write_generated_controller_params(joint_names)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[
            {'robot_description': robot_description},
            controllers_yaml,
            generated_params,
        ],
    )

    spawners = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager'],
            output='screen',
        )
        for name in ('joint_state_broadcaster', 'hand_joint_position_controller')
    ]

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return [robot_state_publisher, ros2_control_node, *spawners, rviz]


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('surgical_hand_description'),
        'config', 'hand_joints.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'backend', default_value='mock',
            description='Hand hardware backend: mock (fake), mujoco (later slice), real (separate launch)'),
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Hand joint configuration YAML'),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Start RViz alongside the hand stack'),
        OpaqueFunction(function=_setup),
    ])
