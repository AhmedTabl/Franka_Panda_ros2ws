#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import sys
import select
import tty
import termios
import math
from scipy.spatial.transform import Rotation as R


class CartesianKeyboardController(Node):
    def __init__(self):
        super().__init__('cartesian_keyboard_controller')

        self.publisher_ = self.create_publisher(Float64MultiArray, '/cartesian_impedance/pose_desired', 10)

        self.position = np.array([0.0, 1.571, 0.785 ], dtype=np.float64)
        initial_roll_deg = 166.0
        r = R.from_euler('x', initial_roll_deg, degrees=True)  # roll about X axis
        self.orientation = r.as_matrix()

        self.get_logger().info("Use WASD to move X/Y, RF to move Z, and TG to rotate the end-effector. Press Q to quit.")

    def get_keypress(self):
        # Non-blocking single-key reader
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

    def run(self):
        old_attrs = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            while rclpy.ok():
                key = self.get_keypress()
                if key:
                    delta = 0.01
                    angle_rad = math.radians(2)  # Small rotation step (2 degrees)
                    if key == 'w':
                        self.position[1] += delta
                    elif key == 's':
                        self.position[1] -= delta
                    elif key == 'a':
                        self.position[0] -= delta
                    elif key == 'd':
                        self.position[0] += delta
                    elif key == 'r':
                        self.position[2] += delta
                    elif key == 'f':
                        self.position[2] -= delta
                    elif key == 't':
                        # Rotate +2 degrees about X (roll)
                        r_delta = R.from_euler('x', angle_rad).as_matrix()
                        self.orientation = self.orientation @ r_delta
                    elif key == 'g':
                        # Rotate -2 degrees about X (roll)
                        r_delta = R.from_euler('x', -angle_rad).as_matrix()
                        self.orientation = self.orientation @ r_delta
                    elif key == 'q':
                        self.get_logger().info("Exiting...")
                        break

                    # Publish message
                    msg = Float64MultiArray()
                    pos_data = self.position.tolist()
                    ori_data = self.orientation.flatten().tolist()
                    msg.data = pos_data + ori_data + [1.0]
                    self.publisher_.publish(msg)

                    self.get_logger().info(f"Current Position: x={self.position[0]:.3f}, y={self.position[1]:.3f}, z={self.position[2]:.3f}")
                    r = R.from_matrix(self.orientation)
                    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
                    self.get_logger().info(f"Orientation (rpy degrees): roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")

                rclpy.spin_once(self, timeout_sec=0.01)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)

def main(args=None):
    rclpy.init(args=args)
    node = CartesianKeyboardController()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
