#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import time


class ArucoMarkerFollower(Node):
    def __init__(self):
        super().__init__('aruco_marker_follower')

        # Publisher to send target pose to the end-effector (Cartesian Impedance Controller)
        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)

        # OpenCV setup
        self.cap = cv2.VideoCapture(0)  # Change to your camera ID if needed
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.camera_matirx = np.array(((9.249796138176930071e+02, 0, 3.381357122676608356e+02),(0,9.246916577452374213e+02, 2.177988860666487199e+02),(0,0,1)))
        self.distortion_coeff = np.array((-4.612486733002377388e-03,-8.481694671573700717,6.661624888698883251e-03,-4.659576407542749023e-03, 4.277086795155930332e+01))

        # Filters
        self.x_buffer = []
        self.y_buffer = []
        self.z_buffer = []
        self.orientation_buffer = []  # Buffer for orientation data
        self.buffer_size = 5  # Moving average window size
        self.transform_threshold = 0.003  # Maximum allowable change in xyz EE position
        self.orientation_threshold = 0.5  # Maximum allowable change in orientation
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        self.prev_orientation = None

        
        # Timer to continuously read camera feed
        self.timer = self.create_timer(0.05, self.tracker)
        self.get_logger().info("Aruco Marker Follower Initialized.")

    def tracker(self):
        ret, img = self.cap.read()
        
        output = self.pose_estimation(img)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.destroy_node()
    
    def pose_estimation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Check the size of the detected marker
                marker_size = cv2.contourArea(corners[i])
                if marker_size < 1000:  # threshold for marker size
                    self.get_logger().warn(f"Rejected marker with size {marker_size}")
                    continue  # Skip this marker

                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.camera_matirx, self.distortion_coeff)
                raw_x = tvec[0][0][0]
                raw_y = tvec[0][0][1]
                raw_z = tvec[0][0][2]

                # Apply threshold filter to raw ArUco marker positions
                filtered_x = self.threshold_filter(self.prev_x, raw_x)
                filtered_y = self.threshold_filter(self.prev_y, raw_y)
                filtered_z = self.threshold_filter(self.prev_z, raw_z)

                # Update previous values
                self.prev_x, self.prev_y, self.prev_z = filtered_x, filtered_y, filtered_z

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

                # Check for sudden changes in orientation
                if self.prev_orientation is not None:
                    orientation_change = [
                        abs(rotation_flattened[i] - self.prev_orientation[i])
                        for i in range(len(rotation_flattened))
                    ]
                    max_change = max(orientation_change)
                    if max_change > self.orientation_threshold:  # threshold for orientation change
                        self.get_logger().warn(f"Rejected marker due to sudden orientation change: {max_change}")
                        continue  # Skip this marker

                # Update previous orientation
                self.prev_orientation = rotation_flattened
                
                # Draw detected markers and axes
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, self.camera_matirx, self.distortion_coeff, rvec, tvec, 0.01)

                # Publish the filtered ArUco marker positions
                self.publish_target(filtered_x, filtered_y, filtered_z, rotation_flattened)

        return frame
    
    def moving_average(self, buffer, new_value):
        """
        Applies a moving average filter to smooth the input.
        """
        buffer.append(new_value)
        if len(buffer) > self.buffer_size:
            buffer.pop(0)  # Remove the oldest value
        if isinstance(new_value, list):  # Handle lists (e.g., orientation data)
            return [sum(values) / len(values) for values in zip(*buffer)]
        return sum(buffer) / len(buffer)

    def threshold_filter(self, prev_value, new_value):
        """
        Applies a threshold filter to prevent sudden jumps.
        """
        if prev_value is None:
            return new_value  # Initialize with the first value
        if abs(new_value - prev_value) > self.transform_threshold:
            return prev_value  # Ignore sudden jump
        return new_value
    


    #Maps the aruco marker position to the end-effector position and publishes it
    def publish_target(self, x, y, z, orientation):

        mapped_x, mapped_y, mapped_z = self.map_to_ee(x, y, z)

        # Apply moving average filter to orientation data
        smoothed_orientation = self.moving_average(self.orientation_buffer, orientation)

        # Round the orientation matrix values to 4 decimal places
        rounded_orientation = [round(value, 4) for value in smoothed_orientation]
        msg = Float64MultiArray()
        msg.data = [mapped_z, mapped_x, mapped_y] + rounded_orientation + [1.0] # the x,y,z coordinates are in the order of z,x,y for the EE workspace
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing target: x={mapped_x}, y={mapped_y}, z={mapped_z}, orientation={rounded_orientation}")


    def map_to_ee(self, x, y, z):
        # Apply moving average filter
        smoothed_x = self.moving_average(self.x_buffer, x)
        smoothed_y = self.moving_average(self.y_buffer, y)
        smoothed_z = self.moving_average(self.z_buffer, z)

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
