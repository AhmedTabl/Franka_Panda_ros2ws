# Franka Wuji VR Teleop

This package receives Hand Tracking Streamer telemetry from a Meta Quest headset and turns it into:

- Franka Cartesian impedance targets on `/cartesian_impedance/pose_desired`
- Wuji hand joint targets on `/wuji_joint_position_controller/commands`

The first implementation uses:

- right hand wrist pose for relative Franka end-effector motion
- right hand landmark curl for Wuji finger curl
- left hand fist closure to pause only arm motion
- TCP input on port `8000` by default, with UDP still available

## Launch

Build and source the workspace:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select \
  franka_hardware \
  franka_description \
  franka_bringup \
  franka_wuji_vr_teleop \
  --symlink-install \
  --base-paths src
source install/setup.bash
```

Start the Franka + Wuji sim first:

```bash
ros2 launch franka_bringup franka_sim.launch.py hand:=true end_effector:=wuji use_rviz:=true
```

Then start VR teleop in a second terminal:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch franka_wuji_vr_teleop franka_wuji_vr_teleop.launch.py \
  protocol:=tcp \
  tcp_host:=10.202.52.252 \
  tcp_port:=8000
```

The VR launch starts `franka_robot_state_broadcaster` and `cartesian_impedance_controller` by default. The Wuji controller is normally already started by `franka_sim.launch.py`; if needed, pass:

```bash
spawn_wuji_controller:=true
```

If the Cartesian impedance controller is already active from another launch or test, prevent this launch from spawning arm controllers again:

```bash
ros2 launch franka_wuji_vr_teleop franka_wuji_vr_teleop.launch.py \
  protocol:=tcp \
  tcp_host:=10.202.52.252 \
  tcp_port:=8000 \
  spawn_arm_controllers:=false
```

## Quest Setup

In Hand Tracking Streamer:

- Protocol: TCP Wireless
- IP: your workstation IP, for example `10.202.52.252`
- Port: `8000`
- Hands: both hands

Keep the headset and workstation on the same network. If the firewall blocks inbound traffic, allow TCP port `8000`.

The `tcp_host` launch value is the local interface to bind on the workstation. Use your workstation's hotspot/LAN IP, for example:

```bash
tcp_host:=10.202.52.252
```

UDP remains available for comparison:

```bash
ros2 launch franka_wuji_vr_teleop franka_wuji_vr_teleop.launch.py \
  protocol:=udp \
  udp_host:=0.0.0.0 \
  udp_port:=9000
```

## Gestures

- Right hand wrist motion controls the Franka end effector relative to the pose captured at startup/resume.
- Right hand finger curl controls the Wuji fingers.
- Closing the left fist pauses arm updates only.
- The Wuji hand continues following right-hand finger curl while the arm is paused.
- Opening the left fist resumes control without jumping by rebasing the current right hand pose.

## Useful Topics

```text
/cartesian_impedance/pose_desired
/wuji_joint_position_controller/commands
/vr/control_wrist_pose
/vr/robot_target_pose
/vr/right_landmarks
/vr/left_landmarks
/vr_teleop/paused
/vr_teleop/status
/vr_teleop/diagnostics
```

Quick health checks:

```bash
ros2 topic echo /vr_teleop/diagnostics
ros2 topic echo /vr_teleop/paused
ros2 topic hz /cartesian_impedance/pose_desired
ros2 topic hz /wuji_joint_position_controller/commands
ros2 control list_controllers
```

The diagnostics line reports the HTS receive rate, arm command rate, Wuji command rate, pause metric, and whether arm teleop is active or paused.

## Tuning

Common launch arguments:

```bash
position_scale:=1.0
max_position_delta:=0.45
position_smoothing:=0.2
orientation_smoothing:=0.3
fist_close_threshold:=0.75
fist_open_threshold:=0.45
pause_debounce_time:=0.2
curl_closed_flexion:=1.8
control_orientation:=true
pos_stiffness:=80.0
rot_stiffness:=20.0
```

For early testing, you can disable arm or hand output independently:

```bash
arm_enabled:=false
wuji_enabled:=false
```

If arm motion feels delayed, try less smoothing and higher Cartesian stiffness:

```bash
ros2 launch franka_wuji_vr_teleop franka_wuji_vr_teleop.launch.py \
  protocol:=tcp \
  tcp_host:=10.202.52.252 \
  tcp_port:=8000 \
  position_smoothing:=0.05 \
  orientation_smoothing:=0.1 \
  pos_stiffness:=120.0 \
  rot_stiffness:=30.0
```

If left-fist pause flickers, increase debounce or spread the hysteresis thresholds:

```bash
pause_debounce_time:=0.35 fist_close_threshold:=0.8 fist_open_threshold:=0.4
```

## Troubleshooting

If the Quest cannot connect over TCP:

- Confirm the Quest and workstation are on the same network.
- Confirm HTS is set to TCP Wireless, not UDP.
- Confirm the IP entered in HTS matches the `tcp_host` address.
- Allow inbound TCP traffic on port `8000`.
- Start the ROS launch before pressing start in HTS.

If the hand moves but the arm does not:

- Check that `cartesian_impedance_controller` is active with `ros2 control list_controllers`.
- Check `/cartesian_impedance/pose_desired` with `ros2 topic hz`.
- Check that `/franka_robot_state_broadcaster/robot_state` is publishing.

If the arm moves but the Wuji fingers do not:

- Check that `wuji_joint_position_controller` is active.
- Check `/wuji_joint_position_controller/commands` with `ros2 topic hz`.
- Relaunch VR teleop with `spawn_wuji_controller:=true` if the controller was not started.
