#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from franka_msgs.action import Move
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
import sys
import time
import mediapipe as mp

class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.state_estimate = 0.0
        self.error_estimate = 1.0

        self.kalman_gain = 0.0

    def update(self, measurement):
        # Prediction update
        self.kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_noise)
        self.state_estimate = self.state_estimate + self.kalman_gain * (measurement - self.state_estimate)
        self.error_estimate = (1 - self.kalman_gain) * self.error_estimate + self.process_noise

        return self.state_estimate

class HandFollower(Node):
    def __init__(self):
        super().__init__('hand_follower')

        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)
        
        self.sim_arm = True
        if self.sim_arm:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper_sim_node/move')
        else:
            self.gripper_client = ActionClient(self, Move, '/panda_gripper/move')
        
        self.gripper_open = True
        self.gripper_width = 0.04
        self.gripper_step = 0.005

        self.cap = cv2.VideoCapture(0)
        # Camera matrix and distortion coefficients (reuse from aruco script)
        self.camera_matirx = np.array(((9.249796138176930071e+02, 0, 3.381357122676608356e+02),(0,9.246916577452374213e+02, 2.177988860666487199e+02),(0,0,1)))
        self.distortion_coeff = np.array((-4.612486733002377388e-03,-8.481694671573700717,6.661624888698883251e+03,-4.659576407542749023e-03, 4.277086795155930332e+01))

        self.kalman_x = SimpleKalmanFilter(1e-5, 1e-2)
        self.kalman_y = SimpleKalmanFilter(1e-5, 1e-2)
        self.kalman_z = SimpleKalmanFilter(1e-5, 1e-2)
        # Stronger filtering for orientation (smoother)
        self.kalman_orientation = [SimpleKalmanFilter(1e-6, 1e-2) for _ in range(9)]

        # MediaPipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

        self.timer = self.create_timer(0.05, self.tracker)
        self.get_logger().info("Hand Follower Initialized.")

    def send_gripper_command(self, width, speed=0.1):
        goal_msg = Move.Goal()
        goal_msg.width = width  # meters, e.g., 0.04 for 4cm open
        goal_msg.speed = speed  # meters/second
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal_msg)

    def tracker(self):
        ret, img = self.cap.read()
        if not ret:
            return

        # BGR to RGB and flip for selfie view
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        output = self.hand_pose_estimation(image, results)
        cv2.imshow('Hand Tracking', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if self.gripper_open:
                self.gripper_width = 0.0  # Fully closed
            else:
                self.gripper_width = 0.08  # Fully open
            self.send_gripper_command(self.gripper_width)
            self.gripper_open = not self.gripper_open
            self.get_logger().info(f"Gripper width set to: {self.gripper_width:.3f} m")
            time.sleep(0.2)  # Debounce to avoid rapid toggling

        # Left arrow (open incrementally)
        elif key == 81:  # Left arrow key code in OpenCV
            self.gripper_width = min(self.gripper_width + self.gripper_step, 0.08)
            self.send_gripper_command(self.gripper_width)
            self.gripper_open = self.gripper_width > 0.0
            self.get_logger().info(f"Gripper width set to: {self.gripper_width:.3f} m")
            time.sleep(0.01)  # Debounce

        # Right arrow (close incrementally)
        elif key == 83:  # Right arrow key code in OpenCV
            self.gripper_width = max(self.gripper_width - self.gripper_step, 0.0)
            self.send_gripper_command(self.gripper_width)
            if self.gripper_width == 0.0:
                self.gripper_open = False
            self.get_logger().info(f"Gripper width set to: {self.gripper_width:.3f} m")
            time.sleep(0.01)  # Debounce

        # Quit
        elif key == ord('q'):
            self.destroy_node()

    def hand_pose_estimation(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                z = -wrist.z  # MediaPipe z is negative towards camera

                # Kalman filter for position
                filtered_x = self.kalman_x.update(cx)
                filtered_y = self.kalman_y.update(cy)
                filtered_z = self.kalman_z.update(z)

                # Compute orientation using landmarks
                orientation_matrix = self.compute_hand_orientation(hand_landmarks.landmark)
                orientation_flat = orientation_matrix.flatten()
                filtered_orientation = [
                    self.kalman_orientation[i].update(orientation_flat[i])
                    for i in range(9)
                ]

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )
                cv2.circle(frame, (cx, cy), 7, (255,0,0), -1)
                cv2.putText(frame, "Wrist", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # Publish the filtered hand position
                self.publish_target(filtered_x, filtered_y, filtered_z, filtered_orientation)
                break  # Only use the first detected hand
        return frame

    def compute_hand_orientation(self, landmarks):
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        middle = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        index = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        pinky = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])

        x_axis = pinky - index
        y_axis = middle - wrist
        z_axis = np.cross(x_axis, y_axis)

        # Normalize axes
        x_axis /= np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else 1
        y_axis /= np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 0 else 1
        z_axis /= np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else 1

        R = np.stack([-y_axis, x_axis, z_axis], axis=1)  # 3x3 rotation matrix
        return R

    def publish_target(self, x, y, z, orientation):
        mapped_x, mapped_y, mapped_z = self.map_to_ee(x, y, z)
        rounded_orientation = [round(value, 4) for value in orientation]
        msg = Float64MultiArray()
        msg.data = [mapped_z, mapped_x, mapped_y] + rounded_orientation + [1.0]
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing target: x={mapped_x}, y={mapped_y}, z={mapped_z}, orientation={rounded_orientation}")

    def map_to_ee(self, x, y, z):
        # Map image coordinates to EE workspace (tune as needed)
        img_w = 640  # adjust if your camera resolution is different
        img_h = 480
        smoothed_x = self.kalman_x.update(x)
        smoothed_y = self.kalman_y.update(y)
        smoothed_z = self.kalman_z.update(z)

        # Map X (image cx: 0-img_w) to EE X (-0.7 to 0.7)
        mapped_x = (smoothed_x / img_w) * (0.7 - (-0.7)) + (-0.7)
        mapped_x = max(-0.7, min(mapped_x, 0.7))
        mapped_x = round(mapped_x, 3)

        # Map Y (image cy: 0-img_h) to EE Y (1.1 to 0.125)
        mapped_y = (smoothed_y / img_h) * (0.125 - 1.1) + 1.1
        mapped_y = max(0.125, min(mapped_y, 1.1))
        mapped_y = round(mapped_y, 3)

        # Map Z (z: 0.01-0.1) to EE Z (0.4 to 1.2)
        mapped_z = (smoothed_z - 0.01) / (0.1 - 0.01) * (1.2 - 0.4) + 0.4
        mapped_z = max(0.4, min(mapped_z, 1.2))
        mapped_z = round(mapped_z, 3)

        return mapped_x, mapped_y, mapped_z

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

