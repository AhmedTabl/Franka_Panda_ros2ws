# Hand-only MuJoCo simulation bringup (slice 7, stage 1).
#
# Starts the mujoco_ros server with the ORCA right-hand scene and runs the
# SAME controllers and topics as the mock backend, so hand_pose_cli and any
# higher layer work unchanged:
#
#   ros2 launch surgical_hand_bringup surgical_hand_mujoco.launch.py
#   ros2 run surgical_hand_skills hand_pose_cli close
#
# Headless (CI / no display):  no_render:=true
# Keep physics paused on start: unpause:=false
#
# How the name mapping works: the URDF (and therefore /joint_states) uses
# the ORCA CAD joint names, while the ORCA MJCF uses semantic names
# (right_i-mcp, ...). hand_joints.yaml carries the mj_joint mapping, the
# ros2_control xacro emits it as the mj_joint_name/actuator_name params,
# and franka_hardware/GenericMjJointPositionHardwareSystem resolves them.

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    description_pkg = get_package_share_directory('surgical_hand_description')
    bringup_pkg = get_package_share_directory('surgical_hand_bringup')
    orca_pkg = get_package_share_directory('orcahand_description')
    franka_bringup_pkg = get_package_share_directory('franka_bringup')

    scene = LaunchConfiguration('scene').perform(context)
    if scene == 'plain':
        scene_file = os.path.join(orca_pkg, 'v2', 'scene_right.xml')
    elif scene == 'objects':
        # Hand + interaction object proxies (tool handle, phantom block,
        # needle and suture proxies). Generated file with absolute mesh
        # paths: see surgical_hand_description/scripts/generate_mujoco_scenes.py.
        scene_file = os.path.join(description_pkg, 'mujoco', 'scene_objects.xml')
    else:
        raise RuntimeError(f'Unknown scene "{scene}". Use scene:=plain or scene:=objects.')
    config_file = os.path.join(description_pkg, 'config', 'hand_joints.yaml')
    xacro_file = os.path.join(description_pkg, 'robots', 'surgical_hand.urdf.xacro')
    static_plugin_yaml = os.path.join(bringup_pkg, 'config', 'surgical_hand_mujoco.yaml')

    import xacro
    robot_description = xacro.process_file(
        xacro_file, mappings={'backend': 'mujoco', 'config_file': config_file}
    ).toprettyxml(indent='  ')

    # Merge the static server/controller config with the controller joint
    # list generated from hand_joints.yaml (single source of truth).
    with open(config_file, 'r') as f:
        joint_names = [entry['name'] for entry in yaml.safe_load(f)['joints']]
    with open(static_plugin_yaml, 'r') as f:
        merged = yaml.safe_load(f)
    merged['hand_joint_position_controller']['ros__parameters']['joints'] = joint_names
    fd, merged_path = tempfile.mkstemp(prefix='surgical_hand_mujoco_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(merged, f)

    # no_render is not a self-contained shorthand at the node level: pass
    # headless/render_offscreen explicitly too, otherwise a Viewer thread
    # starts without a display and stalls the physics loop (which also
    # services controller_manager -> spawners hang).
    no_render = LaunchConfiguration('no_render').perform(context).lower() in ('true', '1')
    mujoco_server = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            os.path.join(franka_bringup_pkg, 'launch', 'sim', 'launch_mujoco_ros_server.launch')),
        launch_arguments={
            'use_sim_time': 'true',
            'modelfile': scene_file,
            'verbose': 'false',
            'unpause': LaunchConfiguration('unpause'),
            'no_render': str(no_render).lower(),
            'headless': str(no_render).lower(),
            'render_offscreen': str(not no_render).lower(),
            'ns': '',
            'mujoco_plugin_config': merged_path,
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    # Model/mesh compilation takes ~10 s, and controller_manager services
    # are only serviced once the sim loop runs: give spawners a generous
    # wait instead of letting them race the startup.
    spawners = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager',
                       '--controller-manager-timeout', '60'],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )
        for name in ('joint_state_broadcaster', 'hand_joint_position_controller')
    ]

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return [mujoco_server, robot_state_publisher, *spawners, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('unpause', default_value='true',
                              description='Start physics running (false = paused)'),
        DeclareLaunchArgument('no_render', default_value='false',
                              description='Disable all rendering (headless verification)'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='Start RViz alongside the sim'),
        DeclareLaunchArgument('scene', default_value='plain',
                              description='plain (hand only) or objects (hand + tool/phantom/needle/suture proxies)'),
        OpaqueFunction(function=_setup),
    ])
