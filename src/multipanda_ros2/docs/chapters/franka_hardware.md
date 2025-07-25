# franka_hardware
This package is the core of the `multipanda_ros2`, implementing the interfaces between `ros2_control` and the real/sim robots.

Each configuration has a dedicated flavor of `franka_hardware`. The real robot interface is:
- `FrankaMultiHardwareInterface` implemented in `franka_multi_hardware_interface.cpp/h`.
    Interface for connecting to $n \geq 1$ Panda robots. 
The corresponding MuJoCo alternative is:
- `FrankaMjHardwareSystem` implemented in `franka_mj_hardware_system.cpp/h`.
    Interface for simulating $n\geq 1$ Panda robots, and inherits from the `mujoco_ros2_control::MujocoRos2SystemInterface`. 

While the low-level implementations are different, with the real one using `libfranka` and the simulation using MuJoCo, the exposed interfaces on the ROS2 level are the same, i.e. you can run the exact same controller that you write for the real robot on the simulated robot, and vice versa. 

## Interface for real robots
The real robot interface `FrankaMultiHardwareInterface` implements a wrapper around the `libfranka` to connect to the robot. The core logic of the wrapper is in the `Robot` class in `robot.cpp/hpp`. `libfranka` offers 5 different control modes (joint torque, position, velocity and Cartesian position, velocity). For each of the control mode, there is a corresponding callback function that returns the corresponding mode's variable. These callback functions do not implement any inherent logic; they simply write the data passed from the `FrankaMultiHardwareInterface` to the robot. As such, all safety features and command calculation needs to be handled by the controller that make use of these. Additionally, they implement several ROS2 service callbacks for changing the internal parameters of the robot on the FCI level.

Each `Robot` class additionally contains a `ModelFranka`, which provides a wrapper around the `franka::Model` class function calls offered by `libfranka`. This `ModelFranka` class is inherited from `ModelBase`, which provides a shared interface to enable controllers that require model data to be agnostic to whether the robot is real or simulated.

The `FrankaMultiHardwareInterface` class, upon being started, goes through the following stages:
- `on_init`
Initializes the connection to the robots, given their names and IP addresses. FCI needs to be activated.
It first performs a preliminary check on whether the loaded URDF and `arm.ros2_control.xacro` files are correctly configured.
It also starts up the parameter service nodes and the error recovery nodes.
- `export_state_interfaces` and `export_command_interfaces`
States are robot states that are updated at 1khz, while the commands are values that will be written to the robot at 1khz. This essentially advertises to the `ros2_control` framework that these interfaces are available, which can be then claimed by a `ros2_control` controller. 
- `on_activate`
Activates the connected robots. By default, all robots will run the `initializeContinuousReading` function, which does not execute any control loops but simply read off the data so that the `FrankaState` and `ModelFranka` objects are updated. This function internally calls the `libfranka`'s `read` function, which cannot be run simultaneously with a `control` function.
- `read`
Continuously reads the `FrankaState` and populates the `ros2_control` states that are exported in the `export_state_interfaces` class. This is what enables the 1khz state read.
- `write`
Continuously reads the command variables that have been exposed by `export_command_interfaces`, which the controllers write their values to. These values are then passed onto each `Robot`. Depending on which controller is active, the corresponding values are written to the robot for execution.

When a controller is activated, the `FrankaMultiHardwareInterface` executes two functions in series:
- `prepare_command_mode_switch`
This ensures that the controller you are trying to activate is valid, i.e. it targets valid command and state interfaces, they are all of the same type and of the same number, and so on. Additionally, it prevents a new controller from being activated if another controller is currently running. A controller in this context refers to a `ros2-controller` that attempts to take control of one of the command interfaces that are exposed.

- `perform_command_mode_switch`
Once the check has passed, the robot's control mode is switched by first executing the `stopRobot` function to stop the currently running thread, then starting the corresponding `initialize...Control` function. Even if the robot is not running a controller, the `initializeContinuousReading` function calls the `libfranka`'s `read` function, which in practice is a control loop, so stopping is necessary.
After the command mode is switched, the corresponding command variable that is passed to the robot in the `write` loop will be written to the robot.

### Error recovery
As the robot runs, it may run into various errors (e.g. joint velocity violation, etc.), which will abort the currently running control loop. In this case, you can recover the robot from the error without having to restart the framework by publishing an empty service to the error recovery node. In the terminal, this looks like:
``` bash
ros2 service call /panda_error_recovery_service_server/error_recovery franka_msgs/srv/ErrorRecovery
```
Current behavior will cause the running controller to be deactivated. You can of course add this service call into your own code to automate this process, and re-activate the controller again.
Errors that cause the robot to lock, e.g. severe torque violations, will require unlocking the robot and restarting the framework.

## Interface for simulated robots
The simulated robot interface `FrankaMjHardwareSystem` works mostly identical to the real robot interfaces. Unlike the real robot, where the robot state and model calculations are handled by the master controller of the robot and thus they can be obtained by their corresponding `libfranka` functions, they are performed entirely in the `FrankaMjHardwareSystem` class' `read` and `write` functions. The `RobotSim` class serves as a replacement for the `Robot` class in the real scenario. This is simply a container for various MuJoCo indices that correspond to that particular robot. 


Likewise, `ModelSim` class implemented in `model_sim.hpp` serves as a replacement for the `ModelFranka`. Since it also inherits from `ModelBase`, it offers the same functions as `ModelFranka` -- the main reason for the real-sim interoperability. Each `ModelSim` instance corresponds to a robot in simulation, and queries the MuJoCo's `mjModel` and `mjData` to extract the robot's parameters that match their real-robot equivalents.

While the `ModelSim` class' inherited functions all have arguments, they are not used, since all the necessary data is contained in `mjData` and `mjModel`. It was simply left as-is for the sake of having identical interfaces.


An important caveat is the gravity and Coriolis-related functions. Since these values are simply presented as a summed array in `qfrc_bias` in MuJoCo, it is difficult to ascertain the accuracy of these functions. For Coriolis, it simply returns `qfrc_bias - qfrc_gravcomp`, while gravity just returns `qfrc_gravcomp`. 

`FrankaState` published by the simulation is not fully populated. Currently, only the following parameters are:
- `q`
- `dq`
- `tau_J`
- `O_T_EE`
- `O_F_ext_hat_K` (untested)
More state values will be added based on need/request.

You can change the initial pose of the robot by setting the `initial_position` parameter in the `ros2_control.xacro` file of the robot setup that you wish to load. These values are then processed at `initSim` stage, and used to set `mjData* d->qpos` of the individual joints to that value.

The simulated robot only supports joint-level states and commands, i.e. joint torque, position and velocity. Cartesian-level commands and states are currently not planned.

The `FrankaMjHardwareSystem` has a few differences to the real one:
- `initSim`
This function is inherited from the base class `mujoco_ros2_control::MujocoRos2SystemInterface`. It is where `const mjModel* m` and `mjData* d` variables are passed and assigned to the class.
`RobotSim` class instances are created here, and `populateIndices` is executed to figure out the indices of the robot's various bodies, joints, actuators, sites, etc., corresponding to the `ns` argument passed to the **xacro** file. This is also the reason why your robot's name in the MuJoCo `xml` file must match the **xacro's `ns` value**. 
Naturally, it does not start parameter service nodes or error recovery service nodes.

Also, if `hand` is set to true (no-hand version is not supported at the moment; this will require a dedicated MuJoCo XML file that does not mount the hand), it starts up the gripper action servers, which again have identical interfaces to the real robot. 

- `read`
Popluates the `FrankaState` as well as the `ros2_control` state interfaces. The `read` function is executed within `mujoco_ros_pkg`'s corresponding loop function. 
- `write`
Writes the command values based on the currently active controller. Again, this function is executed within `mujoco_ros_pkg`'s corresponding loop function.
- `perform_command_mode_switch`
Sets the currently active controller's gains to a fixed value, and sets all other gains to 0. This ensures that no two controllers can be running at the same time.
This is where you can change the gain values for the various actuators if you deem them to be not stiff enough, like position and velocity.

The rest of the functions remain largely the same.

## Other chapters
- [Back to main](../main.md)
- [franka_bringup](./franka_bringup.md)
- [franka_description](./franka_description.md)
- [franka_hardware](./franka_hardware.md)
- [franka_multi_mode_controller](./franka_multi_mode_controller.md)
- [garmi_packages](./garmi.md)