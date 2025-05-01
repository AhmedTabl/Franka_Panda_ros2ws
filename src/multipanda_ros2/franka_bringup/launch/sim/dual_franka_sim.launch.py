#  Copyright (c) 2021 Franka Emika GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
def concatenate_ns(ns1, ns2, absolute=False):
    
    if(len(ns1) == 0):
        return ns2
    if(len(ns2) == 0):
        return ns1
    
    # check for /s at the end and start
    if(ns1[0] == '/'):
        ns1 = ns1[1:]
    if(ns1[-1] == '/'):
        ns1 = ns1[:-1]
    if(ns2[0] == '/'):
        ns2 = ns2[1:]
    if(ns2[-1] == '/'):
        ns2 = ns2[:-1]
    if(absolute):
        ns1 = '/' + ns1
    return ns1 + '/' + ns2

def generate_launch_description():
    arm_id_1_param = "arm_id_1"
    arm_id_2_param = "arm_id_2"
    initial_positions_1_param = 'initial_positions_1'
    initial_positions_2_param = 'initial_positions_2'
    use_rviz_param = 'use_rviz'

    arm_id_1 = LaunchConfiguration(arm_id_1_param)
    arm_id_2 = LaunchConfiguration(arm_id_2_param)
    initial_positions_1 = LaunchConfiguration(initial_positions_1_param)
    initial_positions_2 = LaunchConfiguration(initial_positions_2_param)
    use_rviz = LaunchConfiguration(use_rviz_param)

    # Fixed variables
    load_gripper = True # We make gripper a fixed variable, mainly because parsing the argument 
                        # within generate_launch_description is a fairly unintuitive process, 
                        # and it's not worth doing just for a single boolean.
                        
    if(load_gripper): # mujoco scene file must be manually adjusted since there's no way to pass parameters
        scene_file = 'dual_scene.xml'
    else:
        scene_file = 'dual_scene_ng.xml'
    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'sim',
                                     "dual_panda_arm_sim.urdf.xacro")
    xml_file = os.path.join(get_package_share_directory('franka_description'), 'mujoco', 'franka', scene_file)
    mjros_config_file = os.path.join(get_package_share_directory('franka_bringup'), 'config', 'sim',
                                     'dual_sim_controllers.yaml')
    franka_bringup_path = get_package_share_directory('franka_bringup')
    ns=""

    # Robot state publisher setup
    robot_description = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file, 
            ' arm_id_1:=', arm_id_1, 
            ' arm_id_2:=', arm_id_2,
            ' hand_1:=', str(load_gripper).lower(),
            ' hand_2:=', str(load_gripper).lower(),
            ' initial_positions_1:=', initial_positions_1,
            ' initial_positions_2:=', initial_positions_2])

    params = {'robot_description': robot_description}
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        namespace= ns,
        parameters=[params]
    )

    # Joint state publisher setup
    jsp_source_list = [concatenate_ns(ns, 'joint_states', True)]
    if(load_gripper):
        jsp_source_list.append(concatenate_ns(ns, 'mj_left_gripper_sim_node/joint_states/joint_states', True))
        jsp_source_list.append(concatenate_ns(ns, 'mj_right_gripper_sim_node/joint_states/joint_states', True))

    node_joint_state_publisher = Node( # RVIZ dependency
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            namespace= ns,
            parameters=[
                {'source_list': jsp_source_list,
                 'rate': 30}],
    )

    # Others
    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                             'visualize_dual_franka.rviz')
    

    return LaunchDescription([
        # Launch args
        DeclareLaunchArgument(
            use_rviz_param,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            arm_id_1_param,
            default_value='mj_left',
            description='Unique name of robot 1.'
        ),
        DeclareLaunchArgument(
            arm_id_2_param,
            default_value='mj_right',
            description='Unique name of robot 2.'
        ),
        DeclareLaunchArgument(
            initial_positions_1_param,
            default_value='"0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"',
            description='Initial joint positions of robot 1. Must be enclosed in quotes, and in pure number.'
                        'Defaults to the "communication_test" pose.'),
        DeclareLaunchArgument(
            initial_positions_2_param,
            default_value='"0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"',
            description='Initial joint positions of robot 2. Must be enclosed in quotes, and in pure number.'
                        'Defaults to the "communication_test" pose.'),

        # Mujoco ros2 server launch
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(franka_bringup_path + '/launch/sim/launch_mujoco_ros_server.launch'),
            launch_arguments={
                'use_sim_time': "true",
                'modelfile': xml_file,
                'verbose': "true",
                'ns': ns,
                'mujoco_plugin_config': mjros_config_file
                # 'mujoco_plugin_config': os.path.join(mjr2_control_path, 'example', 'ros2_control_plugins_example.yaml')

            }.items()
        ),

        # Miscellaneous
        node_robot_state_publisher,
        node_joint_state_publisher,

        Node( # RVIZ dependency
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '-c', concatenate_ns(ns, 'controller_manager', True)],
            output='screen',
        ),
        Node(package='rviz2',
             executable='rviz2',
             name='rviz2',
             arguments=['--display-config', rviz_file],
             condition=IfCondition(use_rviz)
             )

    ])
