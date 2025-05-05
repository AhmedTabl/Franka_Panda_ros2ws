#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import sys
import termios
import tty
import time

# Initial joint configuration
joint_values = [0.0, -0.5, 0.0, -1.5, 1.5, 1.0, 0.5]

# Map key presses to joint index changes
KEY_BINDINGS = {
    'q': (0, 0.1),
    'a': (0, -0.1),
    'w': (1, 0.1),
    's': (1, -0.1),
    'e': (2, 0.1),
    'd': (2, -0.1),
    'r': (3, 0.1),
    'f': (3, -0.1),
    't': (4, 0.1),
    'g': (4, -0.1),
    'y': (5, 0.1),
    'h': (5, -0.1),
    'u': (6, 0.1),
    'j': (6, -0.1),
}

def get_key(timeout=0.1):
    """Read a single keypress with timeout (non-blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        else:
            return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

class JointTeleop(Node):
    def __init__(self):
        super().__init__('joint_teleop')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_impedance/joints_desired', 10)
        self.timer_ = self.create_timer(0.005, self.timer_callback)
        self.get_logger().info("Use Q/A, W/S, ..., U/J to move joints 1–7")
        self.get_logger().info("Press Ctrl+C to quit")

    def timer_callback(self):
        key = get_key()
        if key in KEY_BINDINGS:
            idx, delta = KEY_BINDINGS[key]
            joint_values[idx] += delta
            self.get_logger().info(f"Joint {idx+1}: {joint_values[idx]:.2f}")
        msg = Float64MultiArray()
        msg.data = joint_values
        self.publisher_.publish(msg)

if __name__ == '__main__':
    import select
    rclpy.init()
    node = JointTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
