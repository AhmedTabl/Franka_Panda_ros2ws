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
        self.cap = cv2.VideoCapture(1)  # Change to your camera ID if needed
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.camera_matirx = np.array(((9.249796138176930071e+02, 0, 3.381357122676608356e+02),(0,9.246916577452374213e+02, 2.177988860666487199e+02),(0,0,1)))
        self.distortion_coeff = np.array((-4.612486733002377388e-03,-8.481694671573700717,6.661624888698883251e-03,-4.659576407542749023e-03, 4.277086795155930332e+01))

        # Filters
        self.x_buffer = []
        self.y_buffer = []
        self.z_buffer = []
        self.buffer_size = 5  # Moving average window size
        self.threshold = 0.005  # Maximum allowable change
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        
        # Timer to continuously read camera feed
        self.timer = self.create_timer(0.05, self.tracker)
        self.get_logger().info("Aruco Marker Follower Initialized.")


    def moving_average(self, buffer, new_value):
        """
        Applies a moving average filter to smooth the input.
        """
        buffer.append(new_value)
        if len(buffer) > self.buffer_size:
            buffer.pop(0)  # Remove the oldest value
        return sum(buffer) / len(buffer)

    def threshold_filter(self, prev_value, new_value):
        """
        Applies a threshold filter to prevent sudden jumps.
        """
        if prev_value is None:
            return new_value  # Initialize with the first value
        if abs(new_value - prev_value) > self.threshold:
            return prev_value  # Ignore sudden jump
        return new_value
    
    def pose_estimation(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

            
        if len(corners) > 0:
            for i in range(0, len(ids)):
            
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.camera_matirx, self.distortion_coeff)
                x = tvec[0][0][0]
                y = tvec[0][0][1]
                z = tvec[0][0][2]


                cv2.aruco.drawDetectedMarkers(frame, corners) 

                cv2.drawFrameAxes(frame, self.camera_matirx, self.distortion_coeff, rvec, tvec, 0.01) 

                # Convert rotation vector to a rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Flatten the rotation matrix and append it to the message
                rotation_flattened = rotation_matrix.flatten().tolist()
                self.publish_target(x,y,z,rotation_flattened)

        return frame

    
    def tracker(self):
        ret, img = self.cap.read()
        
        output = self.pose_estimation(img)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.destroy_node()

    #Maps the aruco marker position to the end-effector position and publishes it
    def publish_target(self, x, y, z, orientation):

        mapped_x, mapped_y, mapped_z = self.map_to_ee(x, y, z)

        # Round the orientation matrix values to 4 decimal places
        rounded_orientation = [round(value, 4) for value in orientation]
        msg = Float64MultiArray()
        msg.data = [mapped_z, mapped_x, mapped_y] + rounded_orientation + [1.0] # the x,y,z coordinates are in the order of z,x,y for the EE workspace
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing target: x={mapped_x}, y={mapped_y}, z={mapped_z}, orientation={rounded_orientation}")


    def map_to_ee(self, x, y, z):
        # Apply threshold filter
        filtered_x = self.threshold_filter(self.prev_x, x)
        filtered_y = self.threshold_filter(self.prev_y, y)
        filtered_z = self.threshold_filter(self.prev_z, z)

        # Update previous values
        self.prev_x, self.prev_y, self.prev_z = filtered_x, filtered_y, filtered_z

        # Apply moving average filter
        smoothed_x = self.moving_average(self.x_buffer, filtered_x)
        smoothed_y = self.moving_average(self.y_buffer, filtered_y)
        smoothed_z = self.moving_average(self.z_buffer, filtered_z)

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
