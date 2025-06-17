#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from franka_msgs.action import Move
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import time

class SimpleKalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.value = initial_value
        self.estimate_error = 1.0

    def update(self, measurement):
        # Prediction update
        self.estimate_error += self.process_variance
        # Measurement update
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.value += kalman_gain * (measurement - self.value)
        self.estimate_error *= (1 - kalman_gain)
        return self.value

class ArucoMarkerFollower(Node):
    def __init__(self):
        super().__init__('aruco_marker_follower')

        # Publisher to send target pose to the end-effector (Cartesian Impedance Controller)
        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)
        
        self.sim_arm = False  # Flag to indicate if the arm is simulated
        if self.sim_arm:
            # Action client for the sim gripper
            self.gripper_client = ActionClient(self, Move, '/panda_gripper_sim_node/move')
        else:
            # Action client for the real gripper
            self.gripper_client = ActionClient(self, Move, '/panda_gripper/move')
        
        self.gripper_open = True  # Track open/close state for spacebar toggle
        self.gripper_width = 0.04  # Current width (meters), 0.04 is fully open
        self.gripper_step = 0.005  # Step size for arrow key control

        # OpenCV setup
        self.cap = cv2.VideoCapture(0)  # Change to your camera ID if needed
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.camera_matirx = np.array(((9.249796138176930071e+02, 0, 3.381357122676608356e+02),(0,9.246916577452374213e+02, 2.177988860666487199e+02),(0,0,1)))
        self.distortion_coeff = np.array((-4.612486733002377388e-03,-8.481694671573700717,6.661624888698883251e-03,-4.659576407542749023e-03, 4.277086795155930332e+01))

        # Kalman filters for position (x, y, z)
        self.kalman_x = SimpleKalmanFilter(1e-5, 1e-2)
        self.kalman_y = SimpleKalmanFilter(1e-5, 1e-2)
        self.kalman_z = SimpleKalmanFilter(1e-5, 1e-2)
        # Kalman filters for orientation (9 elements of rotation matrix)
        self.kalman_orientation = [SimpleKalmanFilter(1e-5, 1e-2) for _ in range(9)]

        # Timer to continuously read camera feed
        self.timer = self.create_timer(0.05, self.tracker)
        self.get_logger().info("Aruco Marker Follower Initialized.")

    def send_gripper_command(self, width, speed=0.1):
        goal_msg = Move.Goal()
        goal_msg.width = width  # meters, e.g., 0.04 for 4cm open
        goal_msg.speed = speed  # meters/second
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal_msg)

    def tracker(self):
        ret, img = self.cap.read()
        output = self.pose_estimation(img)
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        # Spacebar toggles open/close
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
    
    def pose_estimation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Check the size of the detected marker
                marker_size = cv2.contourArea(corners[i])
                if marker_size < 3000:  # threshold for marker size
                    self.get_logger().warn(f"Rejected marker with size {marker_size}")
                    continue  # Skip this marker

                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.camera_matirx, self.distortion_coeff)
                raw_x = tvec[0][0][0]
                raw_y = tvec[0][0][1]
                raw_z = tvec[0][0][2]

                # Kalman filter for position
                filtered_x = self.kalman_x.update(raw_x)
                filtered_y = self.kalman_y.update(raw_y)
                filtered_z = self.kalman_z.update(raw_z)

                # Convert rotation vector to a rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Remap the rotation axes
                # Swap Y-axis (row 1) with X-axis (row 0)
                remapped_rotation_matrix = np.array([
                    -rotation_matrix[1],  # Y-axis becomes X-axis
                    rotation_matrix[0],  # X-axis becomes Y-axis
                    rotation_matrix[2]   # Z-axis remains Z-axis
                ])

                # Flatten the remapped rotation matrix
                rotation_flattened = remapped_rotation_matrix.flatten().tolist()

                # Kalman filter for orientation (element-wise)
                filtered_orientation = [
                    self.kalman_orientation[i].update(rotation_flattened[i])
                    for i in range(9)
                ]

                # Draw detected markers and axes
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, self.camera_matirx, self.distortion_coeff, rvec, tvec, 0.01)

                # Publish the filtered ArUco marker positions
                self.publish_target(filtered_x, filtered_y, filtered_z, filtered_orientation)

        return frame

    def publish_target(self, x, y, z, orientation):
        mapped_x, mapped_y, mapped_z = self.map_to_ee(x, y, z)
        # Round the orientation matrix values to 4 decimal places
        rounded_orientation = [round(value, 4) for value in orientation]
        msg = Float64MultiArray()
        msg.data = [mapped_z, mapped_x, mapped_y] + rounded_orientation + [1.0] # the x,y,z coordinates are in the order of z,x,y for the EE workspace
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing target: x={mapped_x}, y={mapped_y}, z={mapped_z}, orientation={rounded_orientation}")

    def map_to_ee(self, x, y, z):
        # Apply moving average filter
        smoothed_x = self.kalman_x.update(x)
        smoothed_y = self.kalman_y.update(y)
        smoothed_z = self.kalman_z.update(z)

        # Map X
        mapped_x = (smoothed_x - (-0.03)) / (0.03 - (-0.03)) * (0.7 - (-0.7)) + (-0.7)
        mapped_x = max(-0.7, min(mapped_x, 0.7))  # Clamp to EE range
        mapped_x = round(mapped_x, 3)  # Round to 3 decimal places

        # Map Y
        mapped_y = (smoothed_y - (-0.03)) / (0.03 - (-0.03)) * (0.125 - 1.1) + 1.1
        mapped_y = max(0.125, min(mapped_y, 1.1))  # Clamp to EE range
        mapped_y = round(mapped_y, 3)  # Round to 3 decimal places

        # Map Z
        mapped_z = (smoothed_z - 0.06) / (0.2 - 0.06) * (1.2 - 0.4) + 0.4
        mapped_z = max(0.4, min(mapped_z, 1.2))  # Clamp to EE range
        mapped_z = round(mapped_z, 3)  # Round to 3 decimal places

        return mapped_x, mapped_y, mapped_z

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
