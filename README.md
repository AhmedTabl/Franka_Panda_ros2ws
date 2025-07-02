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

- **Step 1: Panda ROS 2 Configuration**
  - Before using this workspace, you must follow the setup and configuration steps described in the [multipanda_ros2 README](src/multipanda_ros2/README.md). This ensures you have the correct ROS 2 configuration and drivers for the Franka Panda arm (real or simulation).

- **Step 2: Additional Prerequisites for Scripts**
  - Python 3.8+
  - OpenCV
  - PyQt5
  - MediaPipe
  - (Optional) Other dependencies as required by individual scripts

- **Build the workspace:**
  ```bash
  cd ~/ros2_ws
  colcon build
  source install/setup.bash
  ```
- **Launch the main GUI:**
  > **Tip:** It is recommended to run the GUI in a Python virtual environment (venv) to avoid conflicts between PyQt5 and OpenCV versions installed system-wide and those required by the scripts.
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
- Safety: position jump filtering, error messages for connection issues

**Example:**  

![Demo](images/franka_demo.gif)

---

## External Packages & Attribution

- **src/mujoco_ros_pkgs/**: Cloned from [ubi-agni/mujoco_ros_pkgs](https://github.com/ubi-agni/mujoco_ros_pkgs) (see their README and license).
- **src/multipanda_ros2/**: Cloned from [tenfoldpaper/multipanda_ros2](https://github.com/tenfoldpaper/multipanda_ros2) (see their README and license).
- **scripts/camera-calibration/**: Cloned from [niconielsen32/camera-calibration](https://github.com/niconielsen32/camera-calibration) (see their README and license).

All other scripts in `scripts/` are original work.

---

## License

- This workspace: _[Your License Here, e.g. MIT, Apache 2.0, etc.]_
- External packages are under their own licenses (see their READMEs).
