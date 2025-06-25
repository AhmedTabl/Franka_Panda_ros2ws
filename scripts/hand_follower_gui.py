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
        self.declare_parameter('sim_arm', True)
        self.declare_parameter('ee_x_min', -0.7)
        self.declare_parameter('ee_x_max', 0.7)
        self.declare_parameter('ee_y_min', 1.1)
        self.declare_parameter('ee_y_max', 0.125)
        self.declare_parameter('ee_z_min', 0.4)
        self.declare_parameter('ee_z_max', 1.2)

        self.sim_arm = self.get_parameter('sim_arm').get_parameter_value().bool_value
        self.ee_x_min = self.get_parameter('ee_x_min').get_parameter_value().double_value
        self.ee_x_max = self.get_parameter('ee_x_max').get_parameter_value().double_value
        self.ee_y_min = self.get_parameter('ee_y_min').get_parameter_value().double_value
        self.ee_y_max = self.get_parameter('ee_y_max').get_parameter_value().double_value
        self.ee_z_min = self.get_parameter('ee_z_min').get_parameter_value().double_value
        self.ee_z_max = self.get_parameter('ee_z_max').get_parameter_value().double_value
        self.add_on_set_parameters_callback(self.on_param_change)
        
        # ROS2 publisher and action client for gripper
        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)
        if self.sim_arm:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper_sim_node/move')
        else:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper/move')
        
        self.gripper_open = True
        self.gripper_width = 0.08
        self.gripper_step = 0.005
        
        # Smoothing filters
        self.prev_quat = None
        self.quat_alpha = 0.15
        self.kalman_x = SimpleKalmanFilter(1e-5, 1e-2)
        self.kalman_y = SimpleKalmanFilter(1e-5, 1e-2)
        self.kalman_z = SimpleKalmanFilter(1e-5, 1e-2)
        self.kalman_orientation = [SimpleKalmanFilter(1e-6, 1e-2) for _ in range(9)]
        
        # MediaPipe setup for hand tracking
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.gui_callback = gui_callback
        self.last_frame = None
        self.last_info = ""
    
    # Send gripper command via ROS2 action
    def send_gripper_command(self, width, speed=0.1):
        goal_msg = Move.Goal()
        goal_msg.width = width
        goal_msg.speed = speed
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal_msg)
    
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
    
    # Main hand pose estimation and gripper logic
    def hand_pose_estimation(self, frame, results):
        info = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                rotation_flattened, palm_center = self.compute_hand_orientation(hand_landmarks.landmark)
                cx, cy = int(palm_center[0] * w), int(palm_center[1] * h)
                z = -palm_center[2]
                filtered_x = self.kalman_x.update(cx)
                filtered_y = self.kalman_y.update(cy)
                filtered_z = self.kalman_z.update(z)
                filtered_orientation = [
                    self.kalman_orientation[i].update(rotation_flattened[i])
                    for i in range(9)
                ]
                
                # --- Pinch-based gripper control ---
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                # Calculate Euclidean distance in image space
                pinch_dist = np.linalg.norm(
                    np.array([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y])
                )
                info += f"Pinch distance: {pinch_dist:.3f}\n"
                
                min_pinch = 0.04
                max_pinch = 0.3
                min_width = 0.0
                max_width = 0.08
                pinch_dist_clamped = np.clip(pinch_dist, min_pinch, max_pinch)
                gripper_width = ((pinch_dist_clamped - min_pinch) / (max_pinch - min_pinch)) * (max_width - min_width)
                gripper_width = np.clip(gripper_width, min_width, max_width)
                
                # Only send if changed significantly to avoid spamming
                if abs(gripper_width - self.gripper_width) > 0.001:
                    self.send_gripper_command(gripper_width)
                    self.gripper_width = gripper_width
                
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
                cv2.putText(frame, f"Gripper: {gripper_width:.3f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 7, (0,255,255), -1)
                cv2.putText(frame, "Palm Center", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                self.publish_target(filtered_x, filtered_y, filtered_z, filtered_orientation)
                info += f"Gripper width: {gripper_width:.3f} m\n"
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
        msg = Float64MultiArray()
        msg.data = [mapped_z, mapped_x, mapped_y] + orientation + [1.0]
        self.publisher_.publish(msg)
    
    # Callback for parameter changes
    def on_param_change(self, params):
        for param in params:
            if param.name == 'sim_arm':
                self.sim_arm = param.value
                if self.sim_arm:
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
        self.sim_arm_checkbox = QCheckBox("Simulation Mode (sim_arm)")
        self.sim_arm_checkbox.setChecked(self.node.sim_arm)
        self.sim_arm_checkbox.stateChanged.connect(self.on_param_change)
        param_layout.addWidget(self.sim_arm_checkbox)
        group = QGroupBox("EE Mapping Ranges")
        group_layout = QVBoxLayout()
        
        self.x_min = self._make_spinbox("X Min", -2.0, 2.0, self.node.ee_x_min)
        self.x_max = self._make_spinbox("X Max", -2.0, 2.0, self.node.ee_x_max)
        self.y_min = self._make_spinbox("Y Min", -2.0, 2.0, self.node.ee_y_min)
        self.y_max = self._make_spinbox("Y Max", -2.0, 2.0, self.node.ee_y_max)
        self.z_min = self._make_spinbox("Z Min", 0.0, 2.0, self.node.ee_z_min)
        self.z_max = self._make_spinbox("Z Max", 0.0, 2.0, self.node.ee_z_max)
        
        for widget in [self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]:
            group_layout.addLayout(widget['layout'])
        group.setLayout(group_layout)
        param_layout.addWidget(group)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.on_param_change)
        param_layout.addWidget(self.apply_btn)
        
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignTop)
        param_layout.addWidget(self.info_label)
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
        self.node.sim_arm = self.sim_arm_checkbox.isChecked()
        self.node.ee_x_min = self.x_min['spinbox'].value()
        self.node.ee_x_max = self.x_max['spinbox'].value()
        self.node.ee_y_min = self.y_min['spinbox'].value()
        self.node.ee_y_max = self.y_max['spinbox'].value()
        self.node.ee_z_min = self.z_min['spinbox'].value()
        self.node.ee_z_max = self.z_max['spinbox'].value()
        
        # Optionally, update ROS2 parameters as well
        self.node.set_parameters([
            rclpy.parameter.Parameter('sim_arm', rclpy.Parameter.Type.BOOL, self.node.sim_arm),
            rclpy.parameter.Parameter('ee_x_min', rclpy.Parameter.Type.DOUBLE, self.node.ee_x_min),
            rclpy.parameter.Parameter('ee_x_max', rclpy.Parameter.Type.DOUBLE, self.node.ee_x_max),
            rclpy.parameter.Parameter('ee_y_min', rclpy.Parameter.Type.DOUBLE, self.node.ee_y_min),
            rclpy.parameter.Parameter('ee_y_max', rclpy.Parameter.Type.DOUBLE, self.node.ee_y_max),
            rclpy.parameter.Parameter('ee_z_min', rclpy.Parameter.Type.DOUBLE, self.node.ee_z_min),
            rclpy.parameter.Parameter('ee_z_max', rclpy.Parameter.Type.DOUBLE, self.node.ee_z_max),
        ])
    
    # Update the camera feed and info label
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = self.node.process_frame(frame)
        self.display_image(frame)
        self.info_label.setText(self.node.last_info)
    
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
    cap = cv2.VideoCapture(1)
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
