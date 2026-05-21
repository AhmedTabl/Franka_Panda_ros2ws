from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def concatenate_ns(ns1, ns2, absolute=False):
    if len(ns1) == 0:
        return ns2
    if len(ns2) == 0:
        return ns1

    if ns1[0] == '/':
        ns1 = ns1[1:]
    if ns1[-1] == '/':
        ns1 = ns1[:-1]
    if ns2[0] == '/':
        ns2 = ns2[1:]
    if ns2[-1] == '/':
        ns2 = ns2[:-1]
    if absolute:
        ns1 = '/' + ns1
    return ns1 + '/' + ns2


def generate_launch_description():
    arm_id_param = 'arm_id'
    initial_positions_param = 'initial_positions'
    use_rviz_param = 'use_rviz'
    attachment_xyz_param = 'attachment_xyz'
    attachment_rpy_param = 'attachment_rpy'
    end_effector_urdf_param = 'end_effector_urdf'
    end_effector_base_link_param = 'end_effector_base_link'

    arm_id = LaunchConfiguration(arm_id_param)
    initial_positions = LaunchConfiguration(initial_positions_param)
    use_rviz = LaunchConfiguration(use_rviz_param)
    attachment_xyz = LaunchConfiguration(attachment_xyz_param)
    attachment_rpy = LaunchConfiguration(attachment_rpy_param)
    end_effector_urdf = LaunchConfiguration(end_effector_urdf_param)
    end_effector_base_link = LaunchConfiguration(end_effector_base_link_param)

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        'sim',
        'franka_wuji.urdf.xacro',
    )
    xml_file = os.path.join(
        get_package_share_directory('franka_description'),
        'mujoco',
        'franka',
        'scene_wuji_ng.xml',
    )
    mjros_config_file = os.path.join(
        get_package_share_directory('franka_bringup'),
        'config',
        'sim',
        'single_sim_controllers.yaml',
    )
    franka_bringup_path = get_package_share_directory('franka_bringup')
    wuji_description_path = get_package_share_directory('wuji_hand_description')
    ns = ''

    robot_description = Command(
        [
            FindExecutable(name='xacro'),
            ' ',
            franka_xacro_file,
            ' arm_id:=',
            arm_id,
            ' initial_positions:=',
            initial_positions,
            ' end_effector_urdf:=',
            end_effector_urdf,
            ' end_effector_base_link:=',
            end_effector_base_link,
            ' attachment_xyz:=',
            '"',
            attachment_xyz,
            '"',
            ' attachment_rpy:=',
            '"',
            attachment_rpy,
            '"',
        ]
    )

    params = {'robot_description': robot_description}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        namespace=ns,
        parameters=[params],
    )

    node_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=ns,
        parameters=[
            {
                'source_list': [concatenate_ns(ns, 'joint_states', True)],
                'rate': 30,
            }
        ],
    )

    rviz_file = os.path.join(
        get_package_share_directory('franka_description'),
        'rviz',
        'visualize_franka.rviz',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                use_rviz_param,
                default_value='false',
                description='Visualize the robot in Rviz',
            ),
            DeclareLaunchArgument(
                arm_id_param,
                default_value='panda',
                description='The name of the robot. Defaults to panda.',
            ),
            DeclareLaunchArgument(
                initial_positions_param,
                default_value='"0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"',
                description='Initial arm joint positions.',
            ),
            DeclareLaunchArgument(
                end_effector_urdf_param,
                default_value=os.path.join(wuji_description_path, 'urdf', 'right-ros.urdf'),
                description='URDF file for the attached end-effector.',
            ),
            DeclareLaunchArgument(
                end_effector_base_link_param,
                default_value='right_palm_link',
                description='Root/base link of the attached end-effector.',
            ),
            DeclareLaunchArgument(
                attachment_xyz_param,
                default_value='0 0 0',
                description='Fixed joint translation from Panda attachment link to end-effector base.',
            ),
            DeclareLaunchArgument(
                attachment_rpy_param,
                default_value='0 0 3.141592653589793',
                description='Fixed joint rotation from Panda attachment link to end-effector base.',
            ),
            IncludeLaunchDescription(
                FrontendLaunchDescriptionSource(
                    franka_bringup_path + '/launch/sim/launch_mujoco_ros_server.launch'
                ),
                launch_arguments={
                    'use_sim_time': 'true',
                    'modelfile': xml_file,
                    'verbose': 'true',
                    'ns': ns,
                    'mujoco_plugin_config': mjros_config_file,
                }.items(),
            ),
            node_robot_state_publisher,
            node_joint_state_publisher,
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', '-c', concatenate_ns(ns, 'controller_manager', True)],
                output='screen',
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['--display-config', rviz_file],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
