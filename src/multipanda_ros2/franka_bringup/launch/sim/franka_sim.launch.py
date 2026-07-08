from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

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
    # Parameters as launch arguments
    arm_id_param = 'arm_id'
    hand_param = 'hand'
    end_effector_param = 'end_effector'
    initial_positions_param = 'initial_positions'
    use_rviz_param = 'use_rviz'
    end_effector_urdf_param = 'end_effector_urdf'
    end_effector_base_link_param = 'end_effector_base_link'
    attachment_xyz_param = 'attachment_xyz'
    attachment_rpy_param = 'attachment_rpy'
    end_effector_scene_param = 'end_effector_scene'
    
    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'sim',
                                     'panda_arm_sim.urdf.xacro')
    franka_scene_file = os.path.join(get_package_share_directory('franka_description'), 'mujoco', 'franka',
                                     'scene.xml')
    wuji_scene_file = os.path.join(get_package_share_directory('franka_description'), 'mujoco', 'franka',
                                   'scene_wuji_ng.xml')
    no_hand_scene_file = os.path.join(get_package_share_directory('franka_description'), 'mujoco', 'franka',
                                      'scene_ng.xml')
    mjros_config_file = os.path.join(get_package_share_directory('franka_bringup'), 'config', 'sim',
                                     'single_sim_controllers.yaml')
    franka_bringup_path = get_package_share_directory('franka_bringup')
    wuji_description_path = get_package_share_directory('wuji_hand_description')
    ns = ''     # this must match the namespace argument under mujoco_ros2_control in the plugin's parameter yaml file. 
                # See the ros2_control_plugins_example_with_ns.yaml file for more details.

    # Others
    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                             'visualize_franka.rviz')

    def launch_setup(context, *args, **kwargs):
        hand = LaunchConfiguration(hand_param).perform(context).lower() in ('true', '1', 'yes', 'on')
        no_render = LaunchConfiguration('no_render').perform(context).lower() in ('true', '1', 'yes', 'on')
        end_effector = LaunchConfiguration(end_effector_param).perform(context).lower()
        end_effector_scene = LaunchConfiguration(end_effector_scene_param).perform(context)

        if hand and end_effector not in ('franka', 'wuji', 'custom'):
            raise RuntimeError("Unsupported end_effector '{}'. Use 'franka', 'wuji', or 'custom'.".format(end_effector))

        if not hand:
            xml_file = no_hand_scene_file
        elif end_effector_scene:
            xml_file = end_effector_scene
        elif end_effector == 'franka':
            xml_file = franka_scene_file
        elif end_effector == 'wuji':
            xml_file = wuji_scene_file
        else:
            raise RuntimeError(
                "end_effector:=custom requires end_effector_scene to point at a MuJoCo scene."
            )

        hand_value = 'true' if hand else 'false'
        robot_description = Command(
            [
                FindExecutable(name='xacro'), ' ', franka_xacro_file,
                ' arm_id:=', LaunchConfiguration(arm_id_param),
                ' hand:=', hand_value,
                ' end_effector:=', end_effector,
                ' initial_positions:=', LaunchConfiguration(initial_positions_param),
                ' end_effector_urdf:=', LaunchConfiguration(end_effector_urdf_param),
                ' end_effector_base_link:=', LaunchConfiguration(end_effector_base_link_param),
                ' attachment_xyz:=', '"', LaunchConfiguration(attachment_xyz_param), '"',
                ' attachment_rpy:=', '"', LaunchConfiguration(attachment_rpy_param), '"',
            ]
        )

        params = {'robot_description': robot_description}

        node_robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            namespace=ns,
            parameters=[params]
        )

        jsp_source_list = [concatenate_ns(ns, 'joint_states', True)]
        if hand and end_effector == 'franka':
            jsp_source_list.append(concatenate_ns(ns, 'panda_gripper_sim_node/joint_states', True))

        node_joint_state_publisher = Node( # RVIZ dependency
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher',
                namespace=ns,
                parameters=[
                    {'source_list': jsp_source_list,
                     'rate': 30}],
        )

        launch_actions = [
            IncludeLaunchDescription(
                FrontendLaunchDescriptionSource(franka_bringup_path + '/launch/sim/launch_mujoco_ros_server.launch'),
                launch_arguments={
                    'use_sim_time': "true",
                    'modelfile': xml_file,
                    'verbose': "true",
                    'ns': ns,
                    'mujoco_plugin_config': mjros_config_file,
                    # headless support (default false = unchanged behavior);
                    # all three flags must be coherent, see
                    # launch_mujoco_ros_server.launch arg descriptions.
                    'no_render': str(no_render).lower(),
                    'headless': str(no_render).lower(),
                    'render_offscreen': str(not no_render).lower(),
                    'unpause': 'true' if no_render else 'false',
                }.items()
            ),
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
                 condition=IfCondition(LaunchConfiguration(use_rviz_param))
                 )
        ]

        if hand and end_effector == 'wuji':
            launch_actions.append(
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['wuji_joint_position_controller', '-c',
                               concatenate_ns(ns, 'controller_manager', True)],
                    output='screen',
                )
            )

        return launch_actions
    

    return LaunchDescription([
        
        # Launch args
        DeclareLaunchArgument(
            use_rviz_param,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            'no_render',
            default_value='false',
            description='Run the MuJoCo server headless (no viewer, unpaused on start).'),
        DeclareLaunchArgument(
            hand_param,
            default_value='true',
            description='Attach an end-effector to the Panda flange.'),
        DeclareLaunchArgument(
            end_effector_param,
            default_value='franka',
            description='End-effector type to mount when hand is true: franka, wuji, or custom.'),
        DeclareLaunchArgument(
            arm_id_param,
            default_value='panda',
            description='The name of the robot. Defaults to panda.'),
        DeclareLaunchArgument(
            initial_positions_param,
            default_value='"0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"',
            description='Initial joint positions of the robot. Must be enclosed in quotes, and in pure number.'
                        'Defaults to the "communication_test" pose.'),
        DeclareLaunchArgument(
            end_effector_urdf_param,
            default_value=os.path.join(wuji_description_path, 'urdf', 'right-ros.urdf'),
            description='URDF file for wuji/custom end-effector robot_description attachment.'),
        DeclareLaunchArgument(
            end_effector_base_link_param,
            default_value='right_palm_link',
            description='Root/base link for wuji/custom end-effector attachment.'),
        DeclareLaunchArgument(
            attachment_xyz_param,
            default_value='0 0 0',
            description='Fixed joint translation from Panda attachment link to the end-effector base.'),
        DeclareLaunchArgument(
            attachment_rpy_param,
            default_value='0 0 3.141592653589793',
            description='Fixed joint rotation from Panda attachment link to the end-effector base.'),
        DeclareLaunchArgument(
            end_effector_scene_param,
            default_value='',
            description='Optional MuJoCo scene override for custom end-effectors.'),

        OpaqueFunction(function=launch_setup),
    ])
