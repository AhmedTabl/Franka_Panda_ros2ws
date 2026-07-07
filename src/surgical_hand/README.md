# Surgical Hand Stack

ROS 2 software stack for a cable-driven 3-finger robotic hand for surgical
tool manipulation research (suturing / CABG-inspired primitives: needle-driver
acquisition, needle loading, needle driving, suture pull, knot approximation).

Built on top of the existing Franka/Wuji/MuJoCo workspace and follows its
conventions (ros2_control, `forward_command_controller` for hand joints, the
generic MuJoCo joint-position hardware plugin from `franka_hardware`).

## Target Architecture

```
surgical primitive state machines        (slice 8, planned)
        │  hand skill commands
        ▼
named poses / skills  ──────────────────  surgical_hand_skills (hand_pose_cli today)
        │  Float64MultiArray on /hand_joint_position_controller/commands
        ▼
ros2_control controller_manager           joint_state_broadcaster
        │                                 hand_joint_position_controller
        ▼
interchangeable hardware backend (selected by xacro arg `backend`)
        ├── mock    mock_components/GenericSystem              [WORKING]
        ├── mujoco  franka_hardware/GenericMjJointPositionHardwareSystem
        │           (same plugin the Wuji hand uses)           [PLANNED, slice 7]
        └── real    surgical_hand_hw/TendonHandHardwareSystem
                    (Arduino Due + DYNAMIXEL XC330, tendon drive)
                                                               [PLANNED, slices 4-5]
feedback (planned): motor current → tendon tension estimator (slice 3),
eFlesh tactile interface stub (slice 6)
```

The layers above the controller manager never know which backend is active —
the same commands and topics work against fake, simulated, and real hardware.

## Packages

| Package | Language | Purpose |
|---|---|---|
| `surgical_hand_description` | xacro/YAML | Robot description wrapper + backend-switchable ros2_control block. **`config/hand_joints.yaml` is the single source of truth for the joint set** (names, aliases, limits, neutral pose). |
| `surgical_hand_bringup` | launch/YAML | Standalone hand bringup (mock backend today). Controller joint list is generated at launch time from `hand_joints.yaml` — never hand-edited. |
| `surgical_hand_skills` | C++ | Named hand poses (`config/named_poses.yaml`, keyed by joint *aliases*) and the `hand_pose_cli` tool. Will grow into the skills/primitives layer. |
| `surgical_hand_msgs` | msg | Interfaces (`TendonTension.msg`; tactile/hardware status messages later). |
| `surgical_hand_estimation` | C++ | Current→tension estimator: ROS-free `tension_estimator` library (unit-tested, reusable inside the future hardware plugin) + `tension_estimator_node`. **An observer, not a sensor** — gearbox/spool friction, slack, hysteresis, and heating all bias it; consume `status`/`confidence` alongside `tension_n`. |
| `surgical_hand_serial` | C++ | Host↔Arduino Due protocol: CRC16-framed packet codec, POSIX serial wrapper, `DueClient`, and the `hand_serial_cli` bench tool. ROS-free; loopback-tested over a pty. |
| `firmware/arduino_due` | Arduino | Due firmware **skeleton** (ping/status/heartbeat/write-lock implemented; DYNAMIXEL bus stubbed). Not a colcon package; not yet compiled/flashed. Wiring assumptions + unknowns in its README. |
| `../orcahand_description` | vendored | ORCA v2 hand model, used as the **placeholder** until the custom 3-finger hand CAD/URDF exists. See "Local patches" below. |

Planned packages (not yet created): `surgical_hand_hw` (real-hardware
ros2_control plugin + Arduino serial protocol), `surgical_hand_estimation`
(current→tension estimator), `surgical_hand_tactile` (eFlesh interface stub),
firmware for the Arduino Due.

## Placeholder model and joint mapping

The ORCA v2 **right** hand URDF is the placeholder model (17 revolute joints:
wrist, 4-DOF thumb, 4 × 3-DOF fingers). Its auto-generated CAD joint names are
mapped to stable semantic aliases (`thumb_mcp`, `index_tip`, …) in
`surgical_hand_description/config/hand_joints.yaml`. Everything else (poses,
controllers, future tendon maps) uses the aliases, so swapping in the custom
hand later should only require a new joint YAML (and URDF).

Only the right hand is wired up; the left ORCA URDF uses different generated
names and would need its own YAML.

The eventual custom hand is 3 fingers × 4 DOF (12 DOF). The extra placeholder
fingers simply follow the named poses until then.

## Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select \
  orcahand_description \
  surgical_hand_description \
  surgical_hand_bringup \
  surgical_hand_skills \
  --base-paths src
source install/setup.bash
```

**Gotcha:** do not build with the workspace Python venv active
(`deactivate` first). The venv python lacks `catkin_pkg` and breaks
ament CMake package parsing.

## Run and verify (fake backend)

```bash
ros2 launch surgical_hand_bringup surgical_hand.launch.py            # add use_rviz:=true for RViz
ros2 control list_controllers                                        # both controllers active
ros2 topic echo /joint_states --once
ros2 run surgical_hand_skills hand_pose_cli list
ros2 run surgical_hand_skills hand_pose_cli close                    # then re-echo /joint_states
ros2 run surgical_hand_skills hand_pose_cli open
```

## Tension estimator (slice 3)

Model (per the hand spec doc; quasi-static reduction of the Sang et al. 2017
current-based force estimation approach):
`i_eff = i - offset` (zeroed inside the deadband) → `tau = kt * i_eff` →
`tension = direction * efficiency * tau / spool_radius` → low-pass →
clamp ≥ 0. An empirical `(current, tension)` calibration table can replace
the analytic step once the motor is characterized. Statuses: OK /
BELOW_DEADBAND / SLACK / SATURATED, plus an independent
`over_safety_threshold` flag (default 4 N, the suture design capacity).

```bash
ros2 run surgical_hand_estimation tension_estimator_node --ros-args \
  --params-file $(ros2 pkg prefix surgical_hand_estimation)/share/surgical_hand_estimation/config/tension_estimator.yaml
ros2 topic pub -r 50 /tension_estimator/motor_current std_msgs/msg/Float64 "{data: 0.2}"
ros2 topic echo /tension_estimator/tension
```

All numeric parameters in `tension_estimator.yaml` are datasheet-derived
placeholders; the one-motor hardware slice must characterize kt, offset,
deadband, and efficiency before estimates are trusted for control. Note:
rclcpp rejects explicit empty arrays, so omit (don't `[]`) the calibration
parameters when unused. One node = one tendon; run one instance per motor.

## Serial link to the Arduino Due (slice 4)

Frame: `AA 55 len seq cmd payload crc16` (CCITT-FALSE over len..payload),
responses echo `seq` with `cmd|0x80` and lead with a status byte. Safety is
enforced twice: the CLI refuses write commands without `--enable-torque`
*before opening the port*, and the firmware boots write-locked (unlock magic
`0x5AFE`) with a 500 ms heartbeat watchdog that kills torque and re-locks.

```bash
ros2 run surgical_hand_serial hand_serial_cli --port /dev/ttyACM0 ping        # read-only
ros2 run surgical_hand_serial hand_serial_cli --port /dev/ttyACM0 status
ros2 run surgical_hand_serial hand_serial_cli --port /dev/ttyACM0 read-motor 1
ros2 run surgical_hand_serial hand_serial_cli --port /dev/ttyACM0 goal-position 1 2048 --enable-torque
```

No default port is assumed. The full client↔firmware path is verified by a
pty loopback test against a fake firmware that mirrors the lock behavior;
the real firmware skeleton lives in `firmware/arduino_due/` (not yet
compiled/flashed — see its README for wiring assumptions and open questions,
notably 3.3 V Due vs 5 V DYNAMIXEL TTL level shifting).

## Backend selection

`backend` is a xacro arg on `surgical_hand.urdf.xacro` and a launch arg on
`surgical_hand.launch.py`:

- `mock` — works today, no physics, commands mirrored to states.
- `mujoco` — plugin string is wired in the xacro, but the MJCF scene
  (position actuators named `<joint>_actuator`) and the MuJoCo-server launch
  path do not exist yet. The launch file refuses this backend for now.
- `real` — reserved. The plugin does not exist yet; selecting it fails loudly
  at startup rather than pretending to work. Real-hardware bringup will be a
  separate, safety-gated launch file.

## Safety rules for the real-hardware slices (binding, not yet implemented)

- Startup is dry-run/read-only: no torque enable, no EEPROM writes, no motion,
  no operating-mode/ID/baud changes.
- Read-only tools (ping, state dump) come before any write tools.
- Any write/motion path requires explicit flags (e.g. `--enable-torque`,
  `--write-eeprom`) and prints what it is about to do first.
- Arduino link gets timeout/heartbeat handling; wiring assumptions get
  documented with the firmware.
- XC330-M288-T first mode: current-based position control (operating mode 5),
  goal current clamped well below the 1.8 A stall.

## Slice status

| Slice | Status |
|---|---|
| 1. Repo integration + architecture | done (this document; branch was already pushed clean before edits) |
| 2. Fake hand backend | done, verified end-to-end (build + launch + pose command + joint-state check) |
| 3. Current→tension estimator (C++ lib + node + tests) | done: 13 gtests pass; node verified numerically (0.2 A → 9.984 N, deadband → status 1) |
| 4. Arduino serial protocol skeleton | done: 13 tests pass (codec + pty loopback incl. write-lock behavior); CLI verified read-only-by-default; firmware skeleton written but NOT compiled/flashed |
| 5. XC330 one-motor safe path | next |
| 6. eFlesh/tactile interface stub | planned |
| 7. MuJoCo hand backend (ORCA MJCF exists upstream: `v2/models/mjcf/`) | planned |
| 8. Skills + surgical primitives | seeded (named poses only) |

## What is stubbed / unverified right now

- Named poses in `surgical_hand_skills/config/named_poses.yaml` are numeric
  placeholders checked against joint limits but **not visually tuned**.
- `mujoco` and `real` backends are declared but not implemented.
- Tension estimator numbers are datasheet placeholders (no bench data yet).
- Due firmware skeleton is written but not compiled/flashed; its DYNAMIXEL
  bus path returns NOT_IMPLEMENTED. The Due↔motor level-shifting circuit is
  an open hardware question (see `firmware/arduino_due/README.md`).
- No tactile interface yet.
- ORCA URDF inertials look non-physical (upstream issue); irrelevant for the
  mock backend, must be revisited for MuJoCo (the upstream MJCF models are the
  likely source of truth there).

## Local patches to vendored `orcahand_description`

Vendored from https://github.com/orcahand/orcahand_description at upstream
commit `9fa90c180c30b593df43c3b0f8499ab5ee866649` (2026-04-09), nested `.git`
removed to match this workspace's vendoring convention (plain files, like
`wuji-hand-description`).

1. `CMakeLists.txt`: install `assets v1 v2 launch rviz` as directories
   (upstream's trailing slashes flattened the tree, breaking its own
   `package://orcahand_description/v2/...` mesh paths and launch file).
2. `package.xml`: added `<build_type>ament_cmake</build_type>` export so
   colcon does not classify the package as ROS 1 catkin (the `catkin`
   buildtool_depend otherwise wins auto-detection and the package never
   reaches `AMENT_PREFIX_PATH`, breaking `$(find orcahand_description)`).

Both are worth upstreaming.
