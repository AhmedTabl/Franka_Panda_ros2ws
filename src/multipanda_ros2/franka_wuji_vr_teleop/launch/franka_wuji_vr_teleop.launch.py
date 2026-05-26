#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    controller_manager = LaunchConfiguration('controller_manager')

    franka_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['franka_robot_state_broadcaster', '-c', controller_manager],
        condition=IfCondition(LaunchConfiguration('spawn_arm_controllers')),
        output='screen',
    )

    cartesian_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['cartesian_impedance_controller', '-c', controller_manager],
        condition=IfCondition(LaunchConfiguration('spawn_arm_controllers')),
        output='screen',
    )

    wuji_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wuji_joint_position_controller', '-c', controller_manager],
        condition=IfCondition(LaunchConfiguration('spawn_wuji_controller')),
        output='screen',
    )

    teleop_node = Node(
        package='franka_wuji_vr_teleop',
        executable='hts_franka_wuji_teleop',
        name='hts_franka_wuji_teleop',
        output='screen',
        parameters=[{
            'protocol': LaunchConfiguration('protocol'),
            'udp_host': LaunchConfiguration('udp_host'),
            'udp_port': LaunchConfiguration('udp_port'),
            'tcp_host': LaunchConfiguration('tcp_host'),
            'tcp_port': LaunchConfiguration('tcp_port'),
            'control_hand': LaunchConfiguration('control_hand'),
            'pause_hand': LaunchConfiguration('pause_hand'),
            'control_rate': LaunchConfiguration('control_rate'),
            'position_scale': LaunchConfiguration('position_scale'),
            'max_position_delta': LaunchConfiguration('max_position_delta'),
            'position_smoothing': LaunchConfiguration('position_smoothing'),
            'orientation_smoothing': LaunchConfiguration('orientation_smoothing'),
            'control_orientation': LaunchConfiguration('control_orientation'),
            'fist_close_threshold': LaunchConfiguration('fist_close_threshold'),
            'fist_open_threshold': LaunchConfiguration('fist_open_threshold'),
            'pause_debounce_time': LaunchConfiguration('pause_debounce_time'),
            'curl_closed_flexion': LaunchConfiguration('curl_closed_flexion'),
            'arm_enabled': LaunchConfiguration('arm_enabled'),
            'wuji_enabled': LaunchConfiguration('wuji_enabled'),
            'require_robot_state': LaunchConfiguration('require_robot_state'),
            'set_cartesian_stiffness': LaunchConfiguration('set_cartesian_stiffness'),
            'cartesian_param_service': LaunchConfiguration('cartesian_param_service'),
            'pos_stiffness': LaunchConfiguration('pos_stiffness'),
            'rot_stiffness': LaunchConfiguration('rot_stiffness'),
            'robot_state_topic': LaunchConfiguration('robot_state_topic'),
            'arm_command_topic': LaunchConfiguration('arm_command_topic'),
            'wuji_command_topic': LaunchConfiguration('wuji_command_topic'),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('protocol', default_value='tcp'),
        DeclareLaunchArgument('udp_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('udp_port', default_value='9000'),
        DeclareLaunchArgument('tcp_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('tcp_port', default_value='8000'),
        DeclareLaunchArgument('control_hand', default_value='right'),
        DeclareLaunchArgument('pause_hand', default_value='left'),
        DeclareLaunchArgument('control_rate', default_value='100.0'),
        DeclareLaunchArgument('position_scale', default_value='1.0'),
        DeclareLaunchArgument('max_position_delta', default_value='0.45'),
        DeclareLaunchArgument('position_smoothing', default_value='0.2'),
        DeclareLaunchArgument('orientation_smoothing', default_value='0.3'),
        DeclareLaunchArgument('control_orientation', default_value='true'),
        DeclareLaunchArgument('fist_close_threshold', default_value='0.75'),
        DeclareLaunchArgument('fist_open_threshold', default_value='0.45'),
        DeclareLaunchArgument('pause_debounce_time', default_value='0.2'),
        DeclareLaunchArgument('curl_closed_flexion', default_value='1.8'),
        DeclareLaunchArgument('arm_enabled', default_value='true'),
        DeclareLaunchArgument('wuji_enabled', default_value='true'),
        DeclareLaunchArgument('require_robot_state', default_value='true'),
        DeclareLaunchArgument('set_cartesian_stiffness', default_value='true'),
        DeclareLaunchArgument('cartesian_param_service', default_value='/cartesian_impedance_controller/set_parameters'),
        DeclareLaunchArgument('pos_stiffness', default_value='80.0'),
        DeclareLaunchArgument('rot_stiffness', default_value='20.0'),
        DeclareLaunchArgument('robot_state_topic', default_value='/franka_robot_state_broadcaster/robot_state'),
        DeclareLaunchArgument('arm_command_topic', default_value='/cartesian_impedance/pose_desired'),
        DeclareLaunchArgument('wuji_command_topic', default_value='/wuji_joint_position_controller/commands'),
        DeclareLaunchArgument('controller_manager', default_value='/controller_manager'),
        DeclareLaunchArgument('spawn_arm_controllers', default_value='true'),
        DeclareLaunchArgument('spawn_wuji_controller', default_value='false'),
        franka_state_broadcaster,
        cartesian_controller,
        wuji_controller,
        teleop_node,
    ])
