#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from pynput import mouse
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation as R

class EndEffectorFollower(Node):
    def __init__(self):
        super().__init__('end_effector_follower')

        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)

        # Initial end-effector position (fixed Z height)
        self.position = np.array([0.670, 0.001, 0.555 ], dtype=np.float64)
        initial_roll_deg = 166.0
        r = R.from_euler('x', initial_roll_deg, degrees=True)  # roll about X axis
        self.orientation = r.as_matrix()

        # Mouse listener in a separate thread
        self.target_position = self.position.copy()
        self.listener = mouse.Listener(on_move=self.on_mouse_move)
        self.listener.start()

        self.timer = self.create_timer(0.01, self.update_position)

    def on_mouse_move(self, x, y):
        # Scale the mouse coordinates to the robot workspace (adjust as needed)
        scaled_x = (x / 1920) * 0.6 - 0.3  # Adjust for your screen size
        scaled_y = (1 - y / 1080) * 0.6 - 0.3  # Y is inverted for screen to workspace

        self.target_position[0] = scaled_x
        self.target_position[1] = scaled_y

    def update_position(self):
        # Smoothly move towards the target position
        self.position += (self.target_position - self.position) * 0.05

        # Publish the updated position
        msg = Float64MultiArray()
        msg.data = self.position.tolist() + self.orientation.flatten().tolist() + [1.0]
        self.publisher_.publish(msg)

        self.get_logger().info(f"Moving to: {self.position}")

    def destroy(self):
        self.listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorFollower()
    rclpy.spin(node)

    node.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
