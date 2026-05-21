# Franka Bringup End-Effector Selection

This package contains the launch files and controller configuration used to bring up the Franka Panda in simulation and on hardware. The single-arm MuJoCo launch path now supports a modular end-effector selector so the same launch file can spawn:

- the Panda arm with the stock Franka gripper
- the Panda arm with the Wuji right hand
- the Panda arm with no end effector
- future custom hands or tools, as long as they provide matching URDF and MuJoCo description files

The canonical simulation launch file is:

```bash
ros2 launch franka_bringup franka_sim.launch.py
```

## Launch Arguments

The end-effector behavior is controlled by two main arguments:

- `hand`: whether any end effector should be attached.
- `end_effector`: which end effector to attach when `hand:=true`.

Supported `end_effector` values are:

- `franka`: stock Franka gripper.
- `wuji`: Wuji right hand.
- `custom`: user-provided URDF and MuJoCo scene.

Useful additional arguments for URDF-based end effectors:

- `end_effector_urdf`: path to the attached end-effector URDF/xacro.
- `end_effector_base_link`: root/base link of the attached end effector.
- `attachment_xyz`: fixed-joint translation from Panda attachment link to end-effector base.
- `attachment_rpy`: fixed-joint rotation from Panda attachment link to end-effector base.
- `end_effector_scene`: MuJoCo scene XML override for custom end effectors.

## Common Commands

Launch the Panda with the stock Franka gripper:

```bash
ros2 launch franka_bringup franka_sim.launch.py hand:=true end_effector:=franka
```

Launch the Panda with the Wuji hand:

```bash
ros2 launch franka_bringup franka_sim.launch.py hand:=true end_effector:=wuji
```

Launch the Panda with no end effector:

```bash
ros2 launch franka_bringup franka_sim.launch.py hand:=false
```

Launch with RViz:

```bash
ros2 launch franka_bringup franka_sim.launch.py hand:=true end_effector:=wuji use_rviz:=true
```

## Files Changed For The Modular End-Effector Path

The unified robot description lives in:

```text
franka_description/robots/sim/panda_arm_sim.urdf.xacro
```

It now:

- keeps `hand:=true/false` as the switch for whether an end effector exists at all
- adds `end_effector:=franka|wuji|custom`
- mounts the stock Franka gripper when `end_effector:=franka`
- includes and attaches the Wuji/custom URDF when `end_effector:=wuji` or `end_effector:=custom`
- only enables the stock Franka gripper hardware path when `hand:=true end_effector:=franka`

The unified launch path lives in:

```text
franka_bringup/launch/sim/franka_sim.launch.py
```

It now selects the MuJoCo model according to the requested end effector:

- `hand:=true end_effector:=franka` loads `franka_description/mujoco/franka/scene.xml`
- `hand:=true end_effector:=wuji` loads `franka_description/mujoco/franka/scene_wuji_ng.xml`
- `hand:=false` loads `franka_description/mujoco/franka/scene_ng.xml`
- `hand:=true end_effector:=custom` requires `end_effector_scene:=/path/to/custom_scene.xml`

The older Wuji-only launch/xacro wrappers were removed so `franka_sim.launch.py` and `panda_arm_sim.urdf.xacro` are the canonical path.

## Why Wuji Does Not Use The Stock `hand_1` Hardware Flag

In the existing Franka MuJoCo hardware plugin, `hand_1=True` means "the stock Franka gripper is present." When enabled, the plugin expects stock gripper MuJoCo names such as:

- `panda_finger_joint1`
- `panda_finger_joint2`
- `panda_act_gripper`

The Wuji hand does not use those names or that actuator model, so the unified xacro intentionally keeps `hand_1=False` for `end_effector:=wuji`. This avoids accidentally starting the stock gripper action server for a non-Franka hand.

Wuji hand control should be added separately later with Wuji-specific joints, actuators, controller interfaces, and controller YAML.

## End-Effector Weight And Gravity Compensation

The stock Franka gripper is represented in MuJoCo by `scene.xml`, which includes gripper bodies with `gravcomp="1"`.

The Wuji hand is represented in MuJoCo by:

```text
franka_description/mujoco/franka/scene_wuji_ng.xml
franka_description/mujoco/franka/panda_wuji_ng.xml
```

The Wuji MuJoCo body tree also uses `gravcomp="1"` on the palm and finger bodies. This matches the stock gripper convention and prevents the attached hand mass from pulling the arm down just because the no-gripper Panda model was used.

For future hands, make sure the MuJoCo model includes realistic inertials and uses the same gravity-compensation convention if the arm is expected to hold its nominal pose before end-effector control is implemented.

## Adding A Custom End Effector

To add another hand or tool, create both a ROS robot-description attachment and a MuJoCo model attachment.

### 1. Add the description package

Create or add a package that contains the end-effector URDF/xacro and meshes. The URDF should have:

- a clear root/base link
- valid mesh paths, preferably `package://...`
- inertial tags on physical links
- collision tags where needed
- fixed/revolute/prismatic joints with correct axes and limits

Do not edit vendor URDFs directly if a wrapper xacro can adapt them.

### 2. Attach it in the robot description

Use `end_effector:=custom` and provide the URDF and base link:

```bash
ros2 launch franka_bringup franka_sim.launch.py \
  hand:=true \
  end_effector:=custom \
  end_effector_urdf:=/absolute/path/to/custom_hand.urdf.xacro \
  end_effector_base_link:=custom_base_link \
  attachment_xyz:="0 0 0" \
  attachment_rpy:="0 0 0" \
  end_effector_scene:=/absolute/path/to/custom_scene.xml
```

Tune `attachment_xyz` and `attachment_rpy` until the hand/tool is visually aligned with the Panda flange.

### 3. Add a MuJoCo scene

The ROS URDF controls `robot_description` and RViz. MuJoCo uses its own MJCF scene, so a custom end effector also needs a matching MuJoCo model.

Recommended structure:

```text
franka_description/mujoco/franka/panda_custom.xml
franka_description/mujoco/franka/scene_custom.xml
```

`scene_custom.xml` should include the custom Panda model:

```xml
<mujoco model="panda custom scene">
  <include file="panda_custom.xml"/>
  ...
</mujoco>
```

`panda_custom.xml` should be based on the Panda model and attach the custom end-effector body under `panda_link8`. Include inertials, meshes, collision geoms, and `gravcomp="1"` on the attached bodies if you want the same hold-up behavior as the stock Franka gripper and Wuji hand.

### 4. Validate before launching

Check the generated URDF:

```bash
ros2 run xacro xacro \
  src/multipanda_ros2/franka_description/robots/sim/panda_arm_sim.urdf.xacro \
  hand:=true \
  end_effector:=custom \
  end_effector_urdf:=/absolute/path/to/custom_hand.urdf.xacro \
  end_effector_base_link:=custom_base_link \
  > /tmp/panda_custom.urdf

check_urdf /tmp/panda_custom.urdf
```

Check the MuJoCo model:

```bash
python3 -c "import mujoco; mujoco.MjModel.from_xml_path('/absolute/path/to/scene_custom.xml'); print('ok')"
```

Then launch:

```bash
ros2 launch franka_bringup franka_sim.launch.py \
  hand:=true \
  end_effector:=custom \
  end_effector_urdf:=/absolute/path/to/custom_hand.urdf.xacro \
  end_effector_base_link:=custom_base_link \
  end_effector_scene:=/absolute/path/to/scene_custom.xml \
  use_rviz:=true
```

## Current Limitation

This modular path is for spawning, visualization, and arm-side simulation behavior. It does not yet implement Wuji or custom hand control. Control for non-Franka end effectors should be added later through dedicated ros2_control interfaces, MuJoCo actuators, controller YAML, and launch wiring.
