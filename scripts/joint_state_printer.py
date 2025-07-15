#!/usr/bin/env python3
import sys
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLabel
from PyQt5.QtCore import QTimer

class JointStateGui(Node):
    def __init__(self):
        super().__init__('joint_state_gui')
        self.latest_msg = None
        self.latest_franka_state = None
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )
        self.franka_state_sub = self.create_subscription(
            FrankaState,
            '/franka_robot_state_broadcaster/robot_state',
            self.franka_state_callback,
            10
        )

    def listener_callback(self, msg):
        self.latest_msg = msg

    def franka_state_callback(self, msg):
        self.latest_franka_state = msg

class JointStateWindow(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle('Joint States Viewer')
        self.setGeometry(100, 100, 500, 400)
        layout = QVBoxLayout()
        self.label = QLabel('Latest Joint States:')
        layout.addWidget(self.label)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)
        self.ee_label = QLabel('EE Position/Orientation:')
        layout.addWidget(self.ee_label)
        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)  # Update every 100ms

    def update_display(self):
        msg = self.ros_node.latest_msg
        franka_state = self.ros_node.latest_franka_state
        if msg:
            lines = []
            # Only print arm joints (panda_joint1 to panda_joint7)
            for i, name in enumerate(msg.name):
                if name.startswith('panda_joint') and name[-1].isdigit() and 1 <= int(name[-1]) <= 7:
                    pos = round(msg.position[i], 3) if i < len(msg.position) else None
                    vel = round(msg.velocity[i], 3) if i < len(msg.velocity) else None
                    eff = round(msg.effort[i], 3) if i < len(msg.effort) else None
                    lines.append(f"{name}: Position={pos}, Velocity={vel}, Effort={eff}")
            self.text_area.setText('\n'.join(lines))
        # Show EE pose if available
        if franka_state is not None and hasattr(franka_state, 'o_t_ee') and franka_state.o_t_ee is not None and len(franka_state.o_t_ee) == 16:
            o_t_ee = franka_state.o_t_ee
            matrix = np.array(o_t_ee).reshape((4, 4), order='F')
            position = (matrix[0][3], matrix[1][3], matrix[2][3])
            # Orientation as rotation matrix
            rotation = [row[:3] for row in matrix[:3]]
            pos_str = f"x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}"
            rot_str = f"R=[{', '.join([str([round(v,3) for v in r]) for r in rotation])}]"
            self.ee_label.setText(f"EE Position: {pos_str}\nEE Orientation: {rot_str}")
        else:
            self.ee_label.setText("EE Position/Orientation: (waiting for data)")


def main():
    rclpy.init()
    ros_node = JointStateGui()
    app = QApplication(sys.argv)
    window = JointStateWindow(ros_node)
    window.show()
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(ros_node, timeout_sec=0.01))
    timer.start(10)
    sys.exit(app.exec_())
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
