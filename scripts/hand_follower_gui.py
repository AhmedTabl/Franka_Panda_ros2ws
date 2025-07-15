#!/usr/bin/env python3
# -----------------------------
# Hand Follower GUI Main Script
# -----------------------------
# This script provides a PyQt5 GUI for hand tracking and parameter editing for a Franka robot.
# It uses MediaPipe for hand tracking, OpenCV for camera capture, and ROS2 for robot communication.
# The GUI displays the camera feed and allows live editing of mapping parameters.

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R, Slerp

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QCheckBox, QGroupBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray
from franka_msgs.action import Move
from franka_msgs.srv import SetForceTorqueCollisionBehavior
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from rclpy.parameter import Parameter as RclpyParameter

# -----------------------------
# Simple Kalman Filter for Smoothing
# -----------------------------
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

# -----------------------------
# ROS2 Node for Hand Following
# -----------------------------
class HandFollowerNode(Node):
    def __init__(self, gui_callback=None):
        super().__init__('hand_follower')
        
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
        
        # ROS2 publisher and action client for gripper
        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)
        if self.sim_gripper:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper_sim_node/move')
        else:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper/move')
        
        # Add service client for force/torque threshold
        self.collision_srv = self.create_client(SetForceTorqueCollisionBehavior, '/panda_param_service_server/set_force_torque_collision_behavior')
        
        # Add client for cartesian_impedance_controller parameter setting
        self.cartesian_param_srv = self.create_client(
            SetParameters,
            '/cartesian_impedance_controller/set_parameters')
        
        self.gripper_open = True
        self.gripper_width = 0.04
        self.gripper_step = 0.01
        # Smoothing filters
        self.prev_quat = None
        self.quat_alpha = 0.15
        # Make Kalman filters more responsive to small changes
        self.kalman_x = SimpleKalmanFilter(1e-2, 1e-4)
        self.kalman_y = SimpleKalmanFilter(1e-2, 1e-4)
        self.kalman_z = SimpleKalmanFilter(1e-2, 1e-4)
        self.kalman_orientation = [SimpleKalmanFilter(1e-6, 1e-2) for _ in range(9)]
        
        # MediaPipe setup for hand tracking
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.gui_callback = gui_callback
        self.last_info = ""
        self.last_error = ""  # Initialize last_error for gripper command feedback
        self.last_cartesian_error = ""
        self.last_sent_pos = None  # For position jump filtering
        self.position_jump_threshold = 1.0  # meters, adjust as needed
        self.armed = False  # For enable/disable button
    
    # Send gripper command via ROS2 action
    def send_gripper_command(self, width, speed=0.1):
        goal_msg = Move.Goal()
        goal_msg.width = width
        goal_msg.speed = speed
        # Try to wait for the correct action server, else set error for GUI
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            if self.sim_gripper:
                self.last_error = "[ERROR] Sim arm selected, but sim gripper action server not available.\nCheck if simulation is running and sim_gripper is checked."
            else:
                self.last_error = "[ERROR] Real arm selected, but real gripper action server not available.\nCheck if real robot is running and sim_gripper is unchecked."
            return False
        self.last_error = ""  # Clear error if successful
        self.gripper_client.send_goal_async(goal_msg)
        return True
    
    # Process a camera frame: detect hand and update pose
    def process_frame(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output = self.hand_pose_estimation(image, results)
        return output
    
    # Compute hand orientation from landmarks
    def compute_hand_orientation(self, landmarks):
        palm_pts = np.array([
            [landmarks[0].x, landmarks[0].y, landmarks[0].z],
            [landmarks[5].x, landmarks[5].y, landmarks[5].z],
            [landmarks[9].x, landmarks[9].y, landmarks[9].z],
            [landmarks[13].x, landmarks[13].y, landmarks[13].z],
            [landmarks[17].x, landmarks[17].y, landmarks[17].z],
        ])
        
        palm_center = palm_pts.mean(axis=0)
        index = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        pinky = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
        middle = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        
        x_axis = pinky - index
        x_axis /= np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else 1
        y_axis = middle - palm_center
        y_axis /= np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 0 else 1
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else 1
        
        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
        remapped_rotation_matrix = np.array([
            -rotation_matrix[1],
            rotation_matrix[0],
            rotation_matrix[2]
        ])
        
        z_offset = R.from_euler('z', -90, degrees=True)
        final_rot = z_offset * R.from_matrix(remapped_rotation_matrix)
        quat = final_rot.as_quat()
        smoothed_quat = self.smooth_quaternion(quat)
        smoothed_matrix = R.from_quat(smoothed_quat).as_matrix()
        rotation_flattened = smoothed_matrix.flatten().tolist()
        return rotation_flattened, palm_center

    def set_armed(self, armed):
        self.armed = armed

    # Main hand pose estimation and gripper logic
    def hand_pose_estimation(self, frame, results):
        info = ""
        if not self.armed:
            info += "[INFO] Hand tracking is disarmed.\n"
            self.last_info = info
            return frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                rotation_flattened, palm_center = self.compute_hand_orientation(hand_landmarks.landmark)
                cx, cy = int(palm_center[0] * w), int(palm_center[1] * h)
                z_raw = -palm_center[2]
                # --- Pinch-based gripper control (binary open/close) ---
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                # --- 3D pinch distance for normalization (distance between thumb and index tip in 3D) ---
                pinch_dist_3d = np.linalg.norm(
                    np.array([
                        index_tip.x - thumb_tip.x,
                        index_tip.y - thumb_tip.y,
                        index_tip.z - thumb_tip.z
                    ])
                )
                # --- Hand scale: use palm width (distance between landmarks 5 and 17) as a normalization factor ---
                palm_left = hand_landmarks.landmark[5]
                palm_right = hand_landmarks.landmark[17]
                palm_width = np.linalg.norm(
                    np.array([
                        palm_left.x - palm_right.x,
                        palm_left.y - palm_right.y,
                        palm_left.z - palm_right.z
                    ])
                )
                # Avoid division by zero for hand scale
                hand_scale = palm_width if palm_width > 1e-5 else 1e-5
                # --- Normalized pinch: pinch distance divided by hand scale, robust to hand distance from camera ---
                normalized_pinch = pinch_dist_3d / hand_scale
                # --- Store a neutral pinch value (when not pinched) for compensation reference ---
                if not hasattr(self, 'neutral_pinch'):
                    self.neutral_pinch = normalized_pinch
                # --- Exponential moving average update for neutral pinch when not pinched ---
                is_pinched = normalized_pinch < 2.0  # Increased threshold for pinch detection; only update neutral when hand is clearly open
                if not is_pinched:
                    # EMA update for neutral pinch (slower update for more stability)
                    self.neutral_pinch = 0.98 * self.neutral_pinch + 0.02 * normalized_pinch
                # --- Compensation variables explained ---
                # pinch_dist_3d: 3D distance between thumb and index tip (in normalized hand coordinates)
                # palm_width: 3D distance between landmarks 5 and 17 (used as a scale reference for hand size)
                # hand_scale: palm_width, but clamped to avoid division by zero
                # normalized_pinch: pinch_dist_3d / hand_scale, robust to hand distance from camera
                # neutral_pinch: running average of normalized_pinch when hand is open (used as reference for 'no pinch')
                # is_pinched: True if normalized_pinch is below threshold (hand is pinched/closed)
                # k: compensation gain (smaller = less aggressive z correction)
                # z_comp: compensation value to cancel out z changes due to pinching
                # z_compensated: final z value sent to robot after compensation
                
                # --- Track neutral z_raw (when hand is open) for compensation ---
                if not hasattr(self, 'neutral_z_raw'):
                    self.neutral_z_raw = z_raw
                # Update neutral_z_raw only when hand is open (not pinched)
                if not is_pinched:
                    self.neutral_z_raw = 0.98 * self.neutral_z_raw + 0.02 * z_raw
                # --- Compensation: add back the change in z_raw due to pinching ---
                z_comp = self.neutral_z_raw - z_raw
                z_compensated = z_raw + z_comp
                # --- Info/debug output for GUI ---
                info += f"Pinch dist (3D): {pinch_dist_3d:.3f}, Palm width: {palm_width:.3f}, Normalized pinch: {normalized_pinch:.3f}\n"
                info += f"z_raw: {z_raw:.3f}, neutral_z_raw: {self.neutral_z_raw:.3f}, z_comp: {z_comp:.3f}, z_compensated: {z_compensated:.3f}\n"
                # --- Use z_compensated for EE z to cancel out pinch-induced z changes ---
                z_to_use = z_compensated
                # Gripper logic (binary open/close)
                min_width = 0.0
                max_width = 0.08
                gripper_width = min_width if is_pinched else max_width
                # Only send if changed significantly to avoid spamming
                if abs(gripper_width - self.gripper_width) > 0.01:
                    self.send_gripper_command(gripper_width)
                    self.gripper_width = gripper_width
                filtered_x = self.kalman_x.update(cx)
                filtered_y = self.kalman_y.update(cy)
                filtered_z = self.kalman_z.update(z_to_use)
                filtered_orientation = [
                    self.kalman_orientation[i].update(rotation_flattened[i])
                    for i in range(9)
                ]
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
                cv2.putText(frame, f"Gripper: {'Closed' if gripper_width == min_width else 'Open'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 7, (0,255,255), -1)
                cv2.putText(frame, "Palm Center", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                self.publish_target(filtered_x, filtered_y, filtered_z, filtered_orientation)
                info += f"Gripper: {'Closed' if gripper_width == min_width else 'Open'}\n"
                break
        self.last_info = info
        return frame
    
    # Smooth quaternion orientation using Slerp
    def smooth_quaternion(self, quat):
        if self.prev_quat is None:
            self.prev_quat = quat
            return quat
        key_times = [0, 1]
        rotations = R.from_quat([self.prev_quat, quat])
        slerp = Slerp(key_times, rotations)
        smoothed_r = slerp([self.quat_alpha])[0]
        smoothed_quat = smoothed_r.as_quat()
        self.prev_quat = smoothed_quat
        return smoothed_quat
    
    # Publish the target pose to the robot
    def publish_target(self, x, y, z, orientation):
        mapped_x, mapped_y, mapped_z = self.map_to_ee(x, y, z)
        # Position jump filter
        current_pos = np.array([mapped_x, mapped_y, mapped_z])
        if self.last_sent_pos is not None:
            jump = np.linalg.norm(current_pos - self.last_sent_pos)
            if jump > self.position_jump_threshold:
                self.get_logger().warn(f"[SAFETY] Position jump ({jump:.3f} m) exceeds threshold. Command not sent.")
                return
        self.last_sent_pos = current_pos
        msg = Float64MultiArray()
        msg.data = [mapped_z, mapped_x, mapped_y] + orientation + [1.0]
        self.publisher_.publish(msg)

    # Error recovery service call for real arm
    def call_error_recovery(self):
        if self.sim_gripper:
            self.last_error = "[INFO] Error recovery is only available for the real arm."
            return False
        # Create client for error recovery service
        from franka_msgs.srv import ErrorRecovery
        error_recovery_client = self.create_client(ErrorRecovery, '/panda_error_recovery_service_server/error_recovery')
        if not error_recovery_client.wait_for_service(timeout_sec=2.0):
            self.last_error = "[ERROR] Error recovery service not available!"
            return False
        req = ErrorRecovery.Request()
        future = error_recovery_client.call_async(req)
        self.last_error = "[INFO] Error recovery requested."
        return True

    # Callback for parameter changes
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
    
    # Map image coordinates to end-effector workspace
    def map_to_ee(self, x, y, z):
        img_w = 640
        img_h = 480
        smoothed_x = self.kalman_x.update(x)
        smoothed_y = self.kalman_y.update(y)
        smoothed_z = self.kalman_z.update(z)
        mapped_x = (smoothed_x / img_w) * (self.ee_x_max - self.ee_x_min) + self.ee_x_min
        mapped_x = max(self.ee_x_min, min(mapped_x, self.ee_x_max))
        mapped_x = round(mapped_x, 3)
        mapped_y = (smoothed_y / img_h) * (self.ee_y_max - self.ee_y_min) + self.ee_y_min
        mapped_y = max(min(self.ee_y_min, self.ee_y_max), min(mapped_y, max(self.ee_y_min, self.ee_y_max)))
        mapped_y = round(mapped_y, 3)
        mapped_z = (smoothed_z - 0.01) / (0.1 - 0.01) * (self.ee_z_max - self.ee_z_min) + self.ee_z_min
        mapped_z = max(self.ee_z_min, min(mapped_z, self.ee_z_max))
        mapped_z = round(mapped_z, 3)
        return mapped_x, mapped_y, mapped_z

    def set_force_torque_thresholds(self, upper_force, upper_torque):
        # Only set upper thresholds, keep lower as None (will not be changed)
        req = SetForceTorqueCollisionBehavior.Request()
        req.upper_torque_thresholds_nominal = [upper_torque] * 7
        req.upper_force_thresholds_nominal = [upper_force] * 6
        # Leave lower thresholds as default (empty)
        # Wait for service if needed
        if not self.collision_srv.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Collision behavior service not available!')
            return False
        future = self.collision_srv.call_async(req)
        # Optionally, wait for result (non-blocking for GUI)
        return True

    def set_cartesian_stiffness(self, pos_stiff, rot_stiff):
        # Try to set pos_stiff and rot_stiff parameters on the controller
        if not self.cartesian_param_srv.wait_for_service(timeout_sec=1.0):
            self.last_cartesian_error = (
                '[ERROR] Cartesian Impedance Controller not running.\n'
                'Start the controller before setting stiffness parameters.'
            )
            return False
        # Prepare parameter list
        PARAM_TYPE_DOUBLE = 3  # rcl_interfaces/msg/ParameterType: PARAMETER_DOUBLE = 3
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='pos_stiff', value=ParameterValue(type=PARAM_TYPE_DOUBLE, double_value=pos_stiff)),
            Parameter(name='rot_stiff', value=ParameterValue(type=PARAM_TYPE_DOUBLE, double_value=rot_stiff)),
        ]
        self.cartesian_param_srv.call_async(req)
        self.last_cartesian_error = ""
        return True

# -----------------------------
# PyQt5 GUI for Camera Feed and Parameter Editing
# -----------------------------
class HandFollowerGUI(QWidget):
    
    def __init__(self, node, cap):
        super().__init__()
        self.node = node
        self.cap = cap
        self.setWindowTitle("Hand Follower GUI")
        self.setFixedWidth(900)
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)
    
    # Set up the GUI layout
    def init_ui(self):
        
        main_layout = QHBoxLayout()
        
        # Camera feed
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        main_layout.addWidget(self.image_label)
        
        # Parameter controls
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
        self.info_label.setWordWrap(True)  # Enable word wrap
        self.info_label.setFixedWidth(220)  # Limit width to keep layout clean
        param_layout.addWidget(self.info_label)

        # Error label for gripper command feedback
        self.error_label = QLabel()
        self.error_label.setStyleSheet('color: red; font-weight: bold;')
        self.error_label.setAlignment(Qt.AlignTop)
        self.error_label.setWordWrap(True)
        self.error_label.setFixedWidth(220)  # Limit width to keep layout clean
        param_layout.addWidget(self.error_label)

        # Error label for cartesian controller feedback
        self.cartesian_error_label = QLabel()
        self.cartesian_error_label.setStyleSheet('color: red; font-weight: bold;')
        self.cartesian_error_label.setAlignment(Qt.AlignTop)
        self.cartesian_error_label.setWordWrap(True)
        self.cartesian_error_label.setFixedWidth(220)
        param_layout.addWidget(self.cartesian_error_label)

        # Error Recovery button
        self.error_recovery_btn = QPushButton("Error Recovery")
        self.error_recovery_btn.setStyleSheet('background-color: orange; font-weight: bold;')
        self.error_recovery_btn.clicked.connect(self.on_error_recovery)
        param_layout.addWidget(self.error_recovery_btn)

        main_layout.addLayout(param_layout)
        self.setLayout(main_layout)
    
    # Helper to create labeled spinboxes
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
    
    # Handle parameter changes from GUI
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
            rclpy.parameter.Parameter('sim_gripper', rclpy.Parameter.Type.BOOL, self.node.sim_gripper),
            rclpy.parameter.Parameter('ee_x_min', rclpy.Parameter.Type.DOUBLE, self.node.ee_x_min),
            rclpy.parameter.Parameter('ee_x_max', rclpy.Parameter.Type.DOUBLE, self.node.ee_x_max),
            rclpy.parameter.Parameter('ee_y_min', rclpy.Parameter.Type.DOUBLE, self.node.ee_y_min),
            rclpy.parameter.Parameter('ee_y_max', rclpy.Parameter.Type.DOUBLE, self.node.ee_y_max),
            rclpy.parameter.Parameter('ee_z_min', rclpy.Parameter.Type.DOUBLE, self.node.ee_z_min),
            rclpy.parameter.Parameter('ee_z_max', rclpy.Parameter.Type.DOUBLE, self.node.ee_z_max),
        ])
        # Clear error on param change
        if hasattr(self.node, 'last_error'):
            self.node.last_error = ""
    
    # Toggle arm/disarm state
    def toggle_arm(self):
        armed = self.arm_btn.isChecked()
        self.node.set_armed(armed)
        if armed:
            self.arm_btn.setText("Disable Hand Tracking")
        else:
            self.arm_btn.setText("Enable Hand Tracking")
    
    # Error Recovery button handler
    def on_error_recovery(self):
        # Only allow if sim_gripper is False (real arm)
        if not self.node.sim_gripper:
            self.node.call_error_recovery()
        else:
            self.node.last_error = "[INFO] Error recovery is only available for the real arm."
        # Update error label immediately
        self.error_label.setText(self.node.last_error)
    
    # Update the camera feed and info label
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = self.node.process_frame(frame)
        self.display_image(frame)
        self.info_label.setText(self.node.last_info)
        # Show error if present
        if hasattr(self.node, 'last_error') and self.node.last_error:
            self.error_label.setText(self.node.last_error)
        else:
            self.error_label.setText("")
        # Show cartesian error if present
        if hasattr(self.node, 'last_cartesian_error') and self.node.last_cartesian_error:
            self.cartesian_error_label.setText(self.node.last_cartesian_error)
        else:
            self.cartesian_error_label.setText("")
    
    # Display the camera frame in the GUI
    def display_image(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
    
    # Clean up on close
    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

# -----------------------------
# Main entry point
# -----------------------------
def main():
    rclpy.init()
    cap = cv2.VideoCapture(0)
    # Set a lower resolution for compatibility (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Please check your camera device.")
        return
    node = HandFollowerNode()
    app = QApplication(sys.argv)
    gui = HandFollowerGUI(node, cap)
    gui.show()
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.01))
    timer.start(10)
    sys.exit(app.exec_())
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
