# Franka Panda ROS 2 Workspace

This repository contains my ROS 2 workspace (`ros2_ws`) for controlling and experimenting with the Franka Emika Panda robotic arm, supporting both simulation and real hardware.

---

## Table of Contents

- [Overview](#overview)
- [Workspace Structure](#workspace-structure)
- [Main Features](#main-features)
- [Getting Started](#getting-started)
- [Hand Follower GUI](#hand-follower-gui)
- [External Packages & Attribution](#external-packages--attribution)
- [License](#license)

---

## Overview

This workspace enables intuitive control of the Franka Panda arm using hand tracking, live parameter editing, and robust safety features. It is designed for both research and practical robotics applications.

---

## Workspace Structure

```
ros2_ws/
├── src/
│   ├── mujoco_ros_pkgs/         # External: MuJoCo ROS integration (see their README)
│   └── multipanda_ros2/         # External: Multi-arm Franka integration (see their README)
├── scripts/
│   ├── hand_follower_gui.py     # Main GUI for hand tracking and parameter editing (mine)
│   ├── aruco_marker_follower.py # ArUco marker following (mine)
│   ├── joint_teleop.py          # Joint teleoperation (mine)
│   ├── keyboard_cartesian_publisher.py # Cartesian teleop (mine)
│   └── camera-calibration/      # External: Camera calibration toolkit (see their README)
├── install/
├── build/
├── log/
└── README.md
```

- **src/**: Contains external ROS 2 packages (see Attribution).
- **scripts/**: All custom scripts and tools for this workspace.
- **scripts/camera-calibration/**: External, see Attribution.

---

## Main Features

- Unified PyQt5 GUI for hand tracking and parameter editing
- MediaPipe-based hand tracking and OpenCV camera integration
- ROS 2 control of Franka arm (real and simulation)
- Live editing of workspace mapping, gripper, and controller parameters
- Safety features (position jump filtering, error feedback)
- Easy switching between simulation and real hardware
- Additional tools: ArUco marker following, joint and cartesian teleop

---

## Getting Started

- **Prerequisites:** ROS 2 Humble, Python 3.8+, OpenCV, PyQt5, MediaPipe, etc.
- **Build the workspace:**
  ```bash
  cd ~/ros2_ws
  colcon build --symlink-install
  source install/setup.bash
  ```
- **Launch the main GUI:**
  ```bash
  python3 scripts/hand_follower_gui.py
  ```
- **Connect to the robot or simulation** as needed.

---

## Hand Follower GUI

The main script, `hand_follower_gui.py`, provides:
- Real-time hand tracking using MediaPipe and OpenCV
- Live camera feed and parameter editing in a PyQt5 GUI
- ROS 2 integration for controlling the Franka arm (real/sim)
- Gripper control via pinch gesture
- Safety: position jump filtering, error messages for connection issues, etc.

**Screenshots:**  
_Add screenshots of the GUI in action here!_

---

## External Packages & Attribution

- **src/mujoco_ros_pkgs/**: Cloned from [ubi-agni/mujoco_ros_pkgs](https://github.com/ubi-agni/mujoco_ros_pkgs) (see their README and license).
- **src/multipanda_ros2/**: Cloned from [tenfoldpaper/multipanda_ros2](https://github.com/tenfoldpaper/multipanda_ros2) (see their README and license).
- **scripts/camera-calibration/**: Cloned from [original repo, e.g. github.com/someuser/camera-calibration] (see their README and license).

All other scripts in `scripts/` are original work.

---

## License

- This workspace: _[Your License Here, e.g. MIT, Apache 2.0, etc.]_
- External packages are under their own licenses (see their READMEs).
