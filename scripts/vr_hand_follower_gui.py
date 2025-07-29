#!/usr/bin/env python3
# -----------------------------
# VR Hand Follower GUI Script
# -----------------------------
# This script provides a PyQt5 GUI for VR hand tracking and parameter editing for a Franka robot.
# It subscribes to the 'hand_pose' topic for pose data and allows live editing of mapping parameters.

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray
from franka_msgs.action import Move
from franka_msgs.srv import SetForceTorqueCollisionBehavior
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from rclpy.parameter import Parameter as RclpyParameter
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QCheckBox, QGroupBox
)
from PyQt5.QtCore import QTimer, Qt

# Simple Kalman Filter for Smoothing
class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state_estimate = 0.0
        self.error_estimate = 1.0
        self.kalman_gain = 0.0
    def update(self, measurement):
        self.kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_noise)
        self.state_estimate = self.state_estimate + self.kalman_gain * (measurement - self.state_estimate)
        self.error_estimate = (1 - self.kalman_gain) * self.error_estimate + self.process_noise
        return self.state_estimate

class VRHandFollowerNode(Node):

    def __init__(self, gui_callback=None):
        super().__init__('vr_hand_follower')
        # Declare and get parameters for workspace mapping and simulation mode
        self.declare_parameter('sim_gripper', True)
        self.declare_parameter('ee_x_min', -1.0)
        self.declare_parameter('ee_x_max', 1.0)
        self.declare_parameter('ee_y_max', 1.5)
        self.declare_parameter('ee_y_min', -1.0)
        self.declare_parameter('ee_z_min', -0.7)
        self.declare_parameter('ee_z_max', 3.0)

        self.sim_gripper = self.get_parameter('sim_gripper').get_parameter_value().bool_value
        self.ee_x_min = self.get_parameter('ee_x_min').get_parameter_value().double_value
        self.ee_x_max = self.get_parameter('ee_x_max').get_parameter_value().double_value
        self.ee_y_min = self.get_parameter('ee_y_min').get_parameter_value().double_value
        self.ee_y_max = self.get_parameter('ee_y_max').get_parameter_value().double_value
        self.ee_z_min = self.get_parameter('ee_z_min').get_parameter_value().double_value
        self.ee_z_max = self.get_parameter('ee_z_max').get_parameter_value().double_value
        self.add_on_set_parameters_callback(self.on_param_change)
        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)
        if self.sim_gripper:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper_sim_node/move')
        else:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper/move')
        self.collision_srv = self.create_client(SetForceTorqueCollisionBehavior, '/panda_param_service_server/set_force_torque_collision_behavior')
        self.cartesian_param_srv = self.create_client(
            SetParameters,
            '/cartesian_impedance_controller/set_parameters')
        self.last_error = ""
        self.last_cartesian_error = ""
        self.prev_quat = None
        self.quat_alpha = 0.15
        self.last_sent_pos = None
        self.position_jump_threshold = 100.0  # meters
        self.armed = False  # For enable/disable button
        self.gui_callback = gui_callback
        # Smoothing filters
        self.kalman_x = SimpleKalmanFilter(1e-2, 1e-4)
        self.kalman_y = SimpleKalmanFilter(1e-2, 1e-4)
        self.kalman_z = SimpleKalmanFilter(1e-2, 1e-4)
        self.kalman_orientation = [SimpleKalmanFilter(1e-6, 1e-2) for _ in range(9)]
        self.gripper_width = 0.04
        # Subscribe to VR hand pose topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'hand_pose',
            self.vr_pose_callback,
            10)
        self.last_info = ""

    def map_to_ee(self, x, y, z):
        # VR hand tracking ranges:
        # X: -0.4 to 0.7 (right)
        # Y: 0 to 1.5 (up)
        # Z: -0.1 to 0.6 (forward)
        vr_x_min, vr_x_max = -0.4, 0.7
        vr_y_min, vr_y_max = 0.0, 1.5
        vr_z_min, vr_z_max = -0.1, 0.6
        mapped_x = (x - vr_x_min) / (vr_x_max - vr_x_min) * (self.ee_x_max - self.ee_x_min) + self.ee_x_min
        mapped_x = max(self.ee_x_min, min(mapped_x, self.ee_x_max))
        mapped_x = round(mapped_x, 3)
        mapped_y = (y - vr_y_min) / (vr_y_max - vr_y_min) * (self.ee_y_max - self.ee_y_min) + self.ee_y_min
        mapped_y = max(self.ee_y_min, min(mapped_y, self.ee_y_max))
        mapped_y = round(mapped_y, 3)
        mapped_z = (z - vr_z_min) / (vr_z_max - vr_z_min) * (self.ee_z_max - self.ee_z_min) + self.ee_z_min
        mapped_z = max(self.ee_z_min, min(mapped_z, self.ee_z_max))
        mapped_z = round(mapped_z, 3)
        return mapped_x, mapped_y, mapped_z
    
    def vr_pose_callback(self, msg):
        if not self.armed:
            self.last_info = "[INFO] Hand tracking is disarmed."
            if self.gui_callback:
                self.gui_callback()
            return
        data = msg.data
        if len(data) < 13:
            self.last_info = '[WARN] Received hand_pose message with insufficient data.'
            if self.gui_callback:
                self.gui_callback()
            return
        x, y, z = data[0:3]
        # orientation is expected as a 3x3 rotation matrix (flattened, 9 values)
        rotation_matrix = np.array(data[3:12]).reshape((3, 3))
        # Remap and convert to quaternion as in the original script
        from scipy.spatial.transform import Rotation as R, Slerp
        remapped_rotation_matrix = np.array([
            -rotation_matrix[1],
            rotation_matrix[0],
            rotation_matrix[2]
        ])
        z_offset = R.from_euler('z', -90, degrees=True)
        final_rot = z_offset * R.from_matrix(remapped_rotation_matrix)
        quat = final_rot.as_quat()
        # Slerp smoothing (reuse from original script)
        smoothed_quat = self.smooth_quaternion(quat)
        smoothed_matrix = R.from_quat(smoothed_quat).as_matrix()
        rotation_flattened = smoothed_matrix.flatten().tolist()
        # Placeholder for pinch detection (to be replaced with VR pinch flag)
        is_pinched = False  # TODO: Replace with VR pinch flag
        min_width = 0.0
        max_width = 0.08
        gripper_width = min_width if is_pinched else max_width
        if abs(gripper_width - self.gripper_width) > 0.01:
            self.send_gripper_command(gripper_width)
            self.gripper_width = gripper_width
        filtered_x = self.kalman_x.update(x)
        filtered_y = self.kalman_y.update(y)
        filtered_z = self.kalman_z.update(z)
        filtered_orientation = [
            self.kalman_orientation[i].update(rotation_flattened[i])
            for i in range(9)
        ]
        # Map VR hand position to EE position
        mapped_x, mapped_y, mapped_z = self.map_to_ee(filtered_x, filtered_y, filtered_z)
        self.publish_target(mapped_x, mapped_y, mapped_z, filtered_orientation)
        self.last_info = f"Received VR pose: x={mapped_x:.3f}, y={mapped_y:.3f}, z={mapped_z:.3f}\nGripper: {'Closed' if gripper_width == min_width else 'Open'}"
        if self.gui_callback:
            self.gui_callback()
    def smooth_quaternion(self, quat):
        if self.prev_quat is None:
            self.prev_quat = quat
            return quat
        key_times = [0, 1]
        from scipy.spatial.transform import Rotation as R, Slerp
        rotations = R.from_quat([self.prev_quat, quat])
        slerp = Slerp(key_times, rotations)
        smoothed_r = slerp([self.quat_alpha])[0]
        smoothed_quat = smoothed_r.as_quat()
        self.prev_quat = smoothed_quat
        return smoothed_quat

    def send_gripper_command(self, width, speed=0.1):
        goal_msg = Move.Goal()
        goal_msg.width = width
        goal_msg.speed = speed
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            if self.sim_gripper:
                self.last_error = "[ERROR] Sim arm selected, but sim gripper action server not available."
            else:
                self.last_error = "[ERROR] Real arm selected, but real gripper action server not available."
            return False
        self.last_error = ""
        self.gripper_client.send_goal_async(goal_msg)
        return True

    def publish_target(self, x, y, z, orientation):
        current_pos = np.array([x, y, z])
        if self.last_sent_pos is not None:
            jump = np.linalg.norm(current_pos - self.last_sent_pos)
            if jump > self.position_jump_threshold:
                self.last_error = f"[SAFETY] Position jump ({jump:.3f} m) exceeds threshold. Command not sent."
                return
        self.last_sent_pos = current_pos
        msg = Float64MultiArray()
        msg.data = [z, x, y] + orientation + [1.0]
        self.publisher_.publish(msg)

    def call_error_recovery(self):
        if self.sim_gripper:
            self.last_error = "[INFO] Error recovery is only available for the real arm."
            return False
        from franka_msgs.srv import ErrorRecovery
        error_recovery_client = self.create_client(ErrorRecovery, '/panda_error_recovery_service_server/error_recovery')
        if not error_recovery_client.wait_for_service(timeout_sec=2.0):
            self.last_error = "[ERROR] Error recovery service not available!"
            return False
        req = ErrorRecovery.Request()
        future = error_recovery_client.call_async(req)
        self.last_error = "[INFO] Error recovery requested."
        return True

    def on_param_change(self, params):
        for param in params:
            if param.name == 'sim_gripper':
                self.sim_gripper = param.value
                if self.sim_gripper:
                    self.gripper_client = ActionClient(self, Move, '/panda_gripper_sim_node/move')
                else:
                    self.gripper_client = ActionClient(self, Move, '/panda_gripper/move')
            elif param.name == 'ee_x_min':
                self.ee_x_min = param.value
            elif param.name == 'ee_x_max':
                self.ee_x_max = param.value
            elif param.name == 'ee_y_min':
                self.ee_y_min = param.value
            elif param.name == 'ee_y_max':
                self.ee_y_max = param.value
            elif param.name == 'ee_z_min':
                self.ee_z_min = param.value
            elif param.name == 'ee_z_max':
                self.ee_z_max = param.value
        from rcl_interfaces.msg import SetParametersResult
        return SetParametersResult(successful=True)

    def set_force_torque_thresholds(self, upper_force, upper_torque):
        req = SetForceTorqueCollisionBehavior.Request()
        req.upper_torque_thresholds_nominal = [upper_torque] * 7
        req.upper_force_thresholds_nominal = [upper_force] * 6
        if not self.collision_srv.wait_for_service(timeout_sec=2.0):
            self.last_error = 'Collision behavior service not available!'
            return False
        future = self.collision_srv.call_async(req)
        return True

    def set_cartesian_stiffness(self, pos_stiff, rot_stiff):
        if not self.cartesian_param_srv.wait_for_service(timeout_sec=1.0):
            self.last_cartesian_error = (
                '[ERROR] Cartesian Impedance Controller not running.\n'
                'Start the controller before setting stiffness parameters.'
            )
            return False
        PARAM_TYPE_DOUBLE = 3
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='pos_stiff', value=ParameterValue(type=PARAM_TYPE_DOUBLE, double_value=pos_stiff)),
            Parameter(name='rot_stiff', value=ParameterValue(type=PARAM_TYPE_DOUBLE, double_value=rot_stiff)),
        ]
        self.cartesian_param_srv.call_async(req)
        self.last_cartesian_error = ""
        return True

    def set_armed(self, armed):
        self.armed = armed

class VRHandFollowerGUI(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle("VR Hand Follower GUI")
        self.setFixedWidth(900)
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)
        # Connect node callback to GUI update
        self.node.gui_callback = self.update_frame

    def init_ui(self):
        main_layout = QHBoxLayout()
        # No camera feed, just status/pose info
        param_layout = QVBoxLayout()
        # --- Enable/Disable button ---
        self.arm_btn = QPushButton("Enable Hand Tracking")
        self.arm_btn.setCheckable(True)
        self.arm_btn.setChecked(False)
        self.arm_btn.clicked.connect(self.toggle_arm)
        param_layout.addWidget(self.arm_btn)
        self.sim_gripper_checkbox = QCheckBox("Simulation Mode (sim_gripper)")
        self.sim_gripper_checkbox.setChecked(self.node.sim_gripper)
        self.sim_gripper_checkbox.stateChanged.connect(self.on_param_change)
        param_layout.addWidget(self.sim_gripper_checkbox)
        group = QGroupBox("EE Mapping Ranges")
        group_layout = QVBoxLayout()
        self.x_min = self._make_spinbox("X Min", -2.0, 2.0, self.node.ee_x_min)
        self.x_max = self._make_spinbox("X Max", -2.0, 2.0, self.node.ee_x_max)
        self.y_min = self._make_spinbox("Y Min", -2.0, 0.0, self.node.ee_y_min)
        self.y_max = self._make_spinbox("Y Max", 0.0, 2.0, self.node.ee_y_max)
        self.z_min = self._make_spinbox("Z Min", -1.0, 2.0, self.node.ee_z_min)
        self.z_max = self._make_spinbox("Z Max", 0.0, 5.0, self.node.ee_z_max)
        for widget in [self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]:
            group_layout.addLayout(widget['layout'])
        group.setLayout(group_layout)
        param_layout.addWidget(group)

        # Add force/torque threshold controls
        threshold_group = QGroupBox("Collision Thresholds (Franka only)")
        threshold_layout = QVBoxLayout()
        self.force_spin = QDoubleSpinBox()
        self.force_spin.setRange(0, 50)
        self.force_spin.setDecimals(2)
        self.force_spin.setSingleStep(0.1)
        self.force_spin.setValue(20.0)
        self.force_spin.setSuffix(' N')
        self.torque_spin = QDoubleSpinBox()
        self.torque_spin.setRange(0, 50)
        self.torque_spin.setDecimals(2)
        self.torque_spin.setSingleStep(0.1)
        self.torque_spin.setValue(20.0)
        self.torque_spin.setSuffix(' Nm')
        threshold_layout.addWidget(QLabel('Upper Force Threshold (N):'))
        threshold_layout.addWidget(self.force_spin)
        threshold_layout.addWidget(QLabel('Upper Torque Threshold (Nm):'))
        threshold_layout.addWidget(self.torque_spin)
        threshold_group.setLayout(threshold_layout)
        param_layout.addWidget(threshold_group)

        # Add cartesian impedance controller stiffness controls
        cartesian_group = QGroupBox("Cartesian Impedance Stiffness")
        cartesian_layout = QVBoxLayout()
        self.pos_stiff_spin = QDoubleSpinBox()
        self.pos_stiff_spin.setRange(0, 5000)
        self.pos_stiff_spin.setDecimals(1)
        self.pos_stiff_spin.setSingleStep(10)
        self.pos_stiff_spin.setValue(10.0)
        self.pos_stiff_spin.setSuffix(' N/m')
        self.rot_stiff_spin = QDoubleSpinBox()
        self.rot_stiff_spin.setRange(0, 500)
        self.rot_stiff_spin.setDecimals(1)
        self.rot_stiff_spin.setSingleStep(1)
        self.rot_stiff_spin.setValue(10.0)
        self.rot_stiff_spin.setSuffix(' Nm/rad')
        cartesian_layout.addWidget(QLabel('Position Stiffness (pos_stiff):'))
        cartesian_layout.addWidget(self.pos_stiff_spin)
        cartesian_layout.addWidget(QLabel('Rotation Stiffness (rot_stiff):'))
        cartesian_layout.addWidget(self.rot_stiff_spin)
        cartesian_group.setLayout(cartesian_layout)
        param_layout.addWidget(cartesian_group)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.on_param_change)
        param_layout.addWidget(self.apply_btn)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignTop)
        self.info_label.setWordWrap(True)
        self.info_label.setFixedWidth(220)
        param_layout.addWidget(self.info_label)

        self.error_label = QLabel()
        self.error_label.setStyleSheet('color: red; font-weight: bold;')
        self.error_label.setAlignment(Qt.AlignTop)
        self.error_label.setWordWrap(True)
        self.error_label.setFixedWidth(220)
        param_layout.addWidget(self.error_label)

        self.cartesian_error_label = QLabel()
        self.cartesian_error_label.setStyleSheet('color: red; font-weight: bold;')
        self.cartesian_error_label.setAlignment(Qt.AlignTop)
        self.cartesian_error_label.setWordWrap(True)
        self.cartesian_error_label.setFixedWidth(220)
        param_layout.addWidget(self.cartesian_error_label)

        self.error_recovery_btn = QPushButton("Error Recovery")
        self.error_recovery_btn.setStyleSheet('background-color: orange; font-weight: bold;')
        self.error_recovery_btn.clicked.connect(self.on_error_recovery)
        param_layout.addWidget(self.error_recovery_btn)

        main_layout.addLayout(param_layout)
        self.setLayout(main_layout)

    def _make_spinbox(self, label, minv, maxv, default):
        h = QHBoxLayout()
        l = QLabel(label)
        s = QDoubleSpinBox()
        s.setRange(minv, maxv)
        s.setDecimals(3)
        s.setSingleStep(0.01)
        s.setValue(default)
        h.addWidget(l)
        h.addWidget(s)
        return {'layout': h, 'spinbox': s}

    def on_param_change(self):
        self.node.sim_gripper = self.sim_gripper_checkbox.isChecked()
        self.node.ee_x_min = self.x_min['spinbox'].value()
        self.node.ee_x_max = self.x_max['spinbox'].value()
        self.node.ee_y_max = self.y_max['spinbox'].value()
        self.node.ee_y_min = self.y_min['spinbox'].value()
        self.node.ee_z_min = self.z_min['spinbox'].value()
        self.node.ee_z_max = self.z_max['spinbox'].value()
        # Set force/torque thresholds if not sim_gripper
        if not self.node.sim_gripper:
            upper_force = self.force_spin.value()
            upper_torque = self.torque_spin.value()
            self.node.set_force_torque_thresholds(upper_force, upper_torque)
        # Set cartesian impedance controller parameters
        pos_stiff = self.pos_stiff_spin.value()
        rot_stiff = self.rot_stiff_spin.value()
        self.node.set_cartesian_stiffness(pos_stiff, rot_stiff)
        # Optionally, update ROS2 parameters as well
        self.node.set_parameters([
            RclpyParameter('sim_gripper', RclpyParameter.Type.BOOL, self.node.sim_gripper),
            RclpyParameter('ee_x_min', RclpyParameter.Type.DOUBLE, self.node.ee_x_min),
            RclpyParameter('ee_x_max', RclpyParameter.Type.DOUBLE, self.node.ee_x_max),
            RclpyParameter('ee_y_min', RclpyParameter.Type.DOUBLE, self.node.ee_y_min),
            RclpyParameter('ee_y_max', RclpyParameter.Type.DOUBLE, self.node.ee_y_max),
            RclpyParameter('ee_z_min', RclpyParameter.Type.DOUBLE, self.node.ee_z_min),
            RclpyParameter('ee_z_max', RclpyParameter.Type.DOUBLE, self.node.ee_z_max),
        ])
        if hasattr(self.node, 'last_error'):
            self.node.last_error = ""

    def toggle_arm(self):
        armed = self.arm_btn.isChecked()
        self.node.set_armed(armed)
        if armed:
            self.arm_btn.setText("Disable Hand Tracking")
        else:
            self.arm_btn.setText("Enable Hand Tracking")

    def on_error_recovery(self):
        if not self.node.sim_gripper:
            self.node.call_error_recovery()
        else:
            self.node.last_error = "[INFO] Error recovery is only available for the real arm."
        self.error_label.setText(self.node.last_error)

    def update_frame(self):
        self.info_label.setText(self.node.last_info)
        if hasattr(self.node, 'last_error') and self.node.last_error:
            self.error_label.setText(self.node.last_error)
        else:
            self.error_label.setText("")
        if hasattr(self.node, 'last_cartesian_error') and self.node.last_cartesian_error:
            self.cartesian_error_label.setText(self.node.last_cartesian_error)
        else:
            self.cartesian_error_label.setText("")

    def closeEvent(self, event):
        event.accept()

def main():
    rclpy.init()
    node = VRHandFollowerNode()
    app = QApplication(sys.argv)
    gui = VRHandFollowerGUI(node)
    gui.show()
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.01))
    timer.start(10)
    sys.exit(app.exec_())
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
