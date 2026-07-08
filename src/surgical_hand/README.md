# Surgical Hand Stack

ROS 2 software stack for a cable-driven 3-finger robotic hand for surgical
tool manipulation research (suturing / CABG-inspired primitives: needle-driver
acquisition, needle loading, needle driving, suture pull, knot approximation).

Built on top of the existing Franka/Wuji/MuJoCo workspace and follows its
conventions (ros2_control, `forward_command_controller` for hand joints, the
generic MuJoCo joint-position hardware plugin from `franka_hardware`).

## Target Architecture

```
surgical primitive sequences             primitive_cli (5 CABG primitives; scripted,
        │  steps: pose+overrides+guards   feedback logged, guards not yet enforced)
        ▼
named poses / skills  ──────────────────  surgical_hand_skills (pose_library + hand_pose_cli)
        │  Float64MultiArray on /hand_joint_position_controller/commands
        ▼
ros2_control controller_manager           joint_state_broadcaster
        │                                 hand_joint_position_controller
        ▼
interchangeable hardware backend (selected by xacro arg `backend`)
        ├── mock    mock_components/GenericSystem              [WORKING]
        ├── mujoco  franka_hardware/GenericMjJointPositionHardwareSystem
        │           (same plugin the Wuji hand uses)           [WORKING, hand-only]
        └── real    surgical_hand_hw/TendonHandHardwareSystem
                    (XC330 over U2D2, tendon drive)            [tools ready, plugin pending]
feedback: motor current → tendon tension estimator [WORKING, uncalibrated],
eFlesh tactile interface [SIMULATED stub]
```

The layers above the controller manager never know which backend is active —
the same commands and topics work against fake, simulated, and real hardware.

## Packages

| Package | Language | Purpose |
|---|---|---|
| `surgical_hand_description` | xacro/YAML | Robot description wrapper + backend-switchable ros2_control block. **`config/hand_joints.yaml` is the single source of truth for the joint set** (names, aliases, limits, neutral pose). |
| `surgical_hand_bringup` | launch/YAML | Standalone hand bringup (mock backend today). Controller joint list is generated at launch time from `hand_joints.yaml` — never hand-edited. |
| `surgical_hand_skills` | C++ | Skills layer: `pose_library` (joints + named poses + command building, ROS-free), `primitive_engine` (validated primitive sequences from `config/primitives.yaml`), and the `hand_pose_cli` / `primitive_cli` tools. |
| `surgical_hand_msgs` | msg | Interfaces (`TendonTension.msg`; tactile/hardware status messages later). |
| `surgical_hand_estimation` | C++ | Current→tension estimator: ROS-free `tension_estimator` library (unit-tested, reusable inside the future hardware plugin) + `tension_estimator_node`. **An observer, not a sensor** — gearbox/spool friction, slack, hysteresis, and heating all bias it; consume `status`/`confidence` alongside `tension_n`. |
| `surgical_hand_serial` | C++ | Host↔Arduino Due protocol: CRC16-framed packet codec, POSIX serial wrapper, `DueClient`, and the `hand_serial_cli` bench tool. ROS-free; loopback-tested over a pty. |
| `firmware/arduino_due` | Arduino | Due firmware **skeleton** (ping/status/heartbeat/write-lock implemented). Not compiled/flashed. The Due is NOT in the motor loop (see below); this stays reserved for future tactile/aux electronics. |
| `surgical_hand_xc330` | C++ | XC330-M288-T bench tools over the U2D2: `xc330_cli` (read-only scan/ping/read/monitor + CSV logging; torque writes behind `--enable-torque`, EEPROM writes behind `--write-eeprom` with read-back verification) and the read-only `xc330_state_publisher` that feeds the tension estimator. |
| `../DynamixelSDK` | vendored | Official ROBOTIS SDK (`ros2` branch, upstream `c9b5fda`, `.git` removed), used by `surgical_hand_xc330`. |
| `surgical_hand_tactile` | C++ | Tactile interface stub: `TactileSource` interface + `SimulatedTactileSource` (synthetic contact episodes) + `tactile_publisher` node (per-fingertip `TactileState` topics, every message flagged `simulated`). Real eFlesh plugs in behind the same interface later. |
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

## XC330 one-motor bench path (slice 5)

Hardware: **XC330-M288-T → U2D2 → PC over USB**, U2D2 powered from an
external 5 V supply. The Arduino Due is not in the motor loop.

Bench procedure (first four steps are read-only):

```bash
ls /dev/ttyUSB*                                                     # U2D2 (FTDI) device
ros2 run surgical_hand_xc330 xc330_cli --port /dev/ttyUSB0 scan     # finds id + baud
ros2 run surgical_hand_xc330 xc330_cli --port /dev/ttyUSB0 --baud 57600 read 1
ros2 run surgical_hand_xc330 xc330_cli --port /dev/ttyUSB0 --baud 57600 monitor 1 --hz 50 --csv xc330_log.csv

# one-time safe configuration (EEPROM; torque must be off; read-back verified)
ros2 run surgical_hand_xc330 xc330_cli --port ... set-current-limit 1 --ma 300 --write-eeprom
ros2 run surgical_hand_xc330 xc330_cli --port ... set-mode 1 current-position --write-eeprom

# first motion, deliberately weak (current-capped)
ros2 run surgical_hand_xc330 xc330_cli --port ... torque-on 1 --enable-torque
ros2 run surgical_hand_xc330 xc330_cli --port ... goal-current 1 --ma 100 --enable-torque
ros2 run surgical_hand_xc330 xc330_cli --port ... goal-position 1 --deg 10 --enable-torque
ros2 run surgical_hand_xc330 xc330_cli --port ... torque-off 1
```

Chain the real motor into the tension estimator (both read-only):

```bash
ros2 run surgical_hand_xc330 xc330_state_publisher --ros-args \
  -p port:=/dev/ttyUSB0 -p baud:=57600 -p id:=1 \
  -r ~/motor_current:=/tension_estimator/motor_current
ros2 run surgical_hand_estimation tension_estimator_node --ros-args \
  --params-file $(ros2 pkg prefix surgical_hand_estimation)/share/surgical_hand_estimation/config/tension_estimator.yaml
ros2 topic echo /tension_estimator/tension
```

The `monitor --csv` logs are what the estimator characterization needs
(no-load current offset, deadband, and — once a spool and load cell or known
weights exist — Kt and efficiency).

## Backend selection

`backend` is a xacro arg on `surgical_hand.urdf.xacro` and a launch arg on
`surgical_hand.launch.py`:

- `mock` — works, no physics, commands mirrored to states
  (`surgical_hand.launch.py`).
- `mujoco` — works, hand-only physics via the ORCA MJCF
  (`surgical_hand_mujoco.launch.py`, below).
- `real` — reserved. The plugin does not exist yet; selecting it fails loudly
  at startup rather than pretending to work. Real-hardware bringup will be a
  separate, safety-gated launch file.

## MuJoCo hand simulation (slice 7, stage 1)

```bash
ros2 launch surgical_hand_bringup surgical_hand_mujoco.launch.py            # with viewer
ros2 launch surgical_hand_bringup surgical_hand_mujoco.launch.py no_render:=true   # headless
ros2 run surgical_hand_skills hand_pose_cli close                           # same CLI as mock
```

How it works: the ORCA MJCF uses semantic joint names (`right_i-mcp`) while
the URDF uses CAD-generated ones, so `hand_joints.yaml` carries a per-joint
`mj_joint` field, the ros2_control xacro emits it as `mj_joint_name`, and
the (locally extended) `GenericMjJointPositionHardwareSystem` resolves the
MJCF joint/actuator through it. The scene is upstream's
`orcahand_description/v2/scene_right.xml`, unmodified.

Verified headless: controllers activate, `close`/`precision_pinch`/`open`
track through real physics (pinky mcp exact, index lags ~0.15 rad in a fist
due to finger self-contact against ORCA's soft default actuators — kp=2,
±1 N·m).

Interaction objects (`scene:=objects`): tool-handle cylinder, phantom
block, straight needle proxy, and a 4-segment ball-joint suture chain,
placed near the hand (placements untuned for actual grasp demos):

```bash
ros2 launch surgical_hand_bringup surgical_hand_mujoco.launch.py scene:=objects
```

Franka attachment — the hand on the Panda flange via franka_bringup's
existing *custom end-effector* hook (no franka_description edits):

```bash
ros2 launch surgical_hand_bringup surgical_hand_franka_mujoco.launch.py   # add no_render:=true for headless
ros2 run surgical_hand_skills hand_pose_cli close
```

Pieces: [surgical_hand_ee.urdf.xacro](surgical_hand_description/robots/surgical_hand_ee.urdf.xacro)
(ORCA URDF + hand ros2_control block), generated `panda_orca_ng.xml` /
`scene_orca_ng.xml` (namespaced defaults/materials, gravcomp on all hand
bodies, per-body contact excludes), and a wrapper launch that spawns the
hand controller via `spawner --controller-type --param-file` so
franka_bringup's controller YAML stays untouched. Verified headless: 24
joints in `/joint_states`, hand tracks `needle_driver_grasp` while the arm
holds its start pose. Caveats: the flange mount transform is an untuned
placeholder (keep the launch args and `MOUNT_*` in the generator in sync);
start an arm controller for real arm work — with none active the panda's
default position actuators slowly pull it to the zero pose; franka_sim's
`joint_state_publisher` aggregator also publishes zeros on `/joint_states`
alongside the broadcaster (pre-existing franka_sim quirk — filter by
publisher or use the controller topics).

All scene files regenerate with
`python3 surgical_hand_description/scripts/generate_mujoco_scenes.py`
(absolute mesh paths — rerun after moving the workspace).

## Surgical primitives (slice 8)

The five basic CABG primitives from the design doc as validated, scripted
pose sequences ([primitives.yaml](surgical_hand_skills/config/primitives.yaml)):
needle_driver_acquisition, needle_loading, needle_driving (wrist arc),
suture_pull, knot_approximation.

```bash
ros2 run surgical_hand_skills primitive_cli list
ros2 run surgical_hand_skills primitive_cli needle_driving
```

Each step = named pose + per-alias overrides + duration + optional guards
(`max_tension_n`, `require_contact`). The runner republishes commands at
20 Hz per step, logs the latest tendon-tension estimate and tactile state
at every transition, and reports guard violations as warnings — guards are
**deliberately not enforced yet** (that needs the bench-calibrated
estimator and real tactile hardware). Works against any backend; verified
on the fake backend with the estimator + simulated tactile running.

## Tactile stub (slice 6)

```bash
ros2 run surgical_hand_tactile tactile_publisher
ros2 topic echo /tactile_publisher/index
```

Publishes eFlesh-shaped `TactileState` per fingertip (raw magnetometer
array + contact/forces/slip/location) from a synthetic episode generator.
Everything is marked `simulated: true`; the real eFlesh driver later
implements the same `TactileSource` interface without changing topics,
messages, or consumers.

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
| 5. XC330 one-motor safe path | tools done: 7 conversion gtests pass; CLI gating verified (writes refused before port open); scan/ping/read/monitor/state-publisher ready. **Bench run against the real motor pending — U2D2 was not plugged in during development** |
| 6. eFlesh/tactile interface stub | done: 9 gtests pass; publisher runtime-verified (per-sensor topics, periodic synthetic contacts, `simulated` flag) |
| 7. MuJoCo hand backend | done: hand-only, interaction objects, and Franka attachment all verified headless (mount transform + object placements untuned) |
| 8. Skills + surgical primitives | done as skeletons: 5 primitives sequenced with feedback logging; 8 engine gtests; guards parsed but not enforced |

## What is stubbed / unverified right now

- Named poses in `surgical_hand_skills/config/named_poses.yaml` are numeric
  placeholders checked against joint limits but **not visually tuned**.
- `mujoco` and `real` backends are declared but not implemented.
- Tension estimator numbers are datasheet placeholders (no bench data yet).
- Due firmware skeleton is written but not compiled/flashed; its DYNAMIXEL
  bus path returns NOT_IMPLEMENTED (the Due is out of the motor loop anyway).
- Tactile data is entirely synthetic (`simulated: true` on every message);
  real eFlesh needs a magnetometer-board driver + calibration model behind
  the existing `TactileSource` interface.
- ORCA's default MJCF actuators are soft (kp=2, ±1 N·m); expect fingertip
  lag under contact until retuned. The Franka flange mount transform and
  the interaction-object placements are untuned placeholders; the needle
  proxy is straight (curved needle needs a mesh/multi-capsule weld).
- Primitives are scripted pose sequences with placeholder durations;
  tension/tactile guards are logged, not enforced.

Local patches to other vendored/workspace packages (beyond orcahand):

- `multipanda_ros2/franka_hardware` `GenericMjJointPositionHardwareSystem`:
  added optional per-joint `mj_joint_name` parameter (default = joint name,
  Wuji path unchanged).
- `mujoco_ros_pkgs/mujoco_ros` `main.cpp`: fixed headless+GLFW builds never
  spinning the ROS executor (all services, incl. controller_manager, hung).
  Worth upstreaming to ubi-agni/mujoco_ros_pkgs.
- `multipanda_ros2/franka_bringup` `franka_sim.launch.py`: added an optional
  `no_render` arg (default false = unchanged) for headless runs.
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
