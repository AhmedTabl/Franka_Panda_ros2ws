#!/usr/bin/env python3

from __future__ import annotations

import math
import re
import select
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import rclpy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import Bool, Float64MultiArray, String


# HTS/Unity: x right, y up, z forward.
# Robot frame convention used here: x forward, y left, z up.
UNITY_TO_ROBOT = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

FLOAT_PATTERN = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')

WUJI_JOINT_ORDER = (
    'right_finger1_joint1', 'right_finger1_joint2', 'right_finger1_joint3',
    'right_finger1_joint4',
    'right_finger2_joint1', 'right_finger2_joint2', 'right_finger2_joint3',
    'right_finger2_joint4',
    'right_finger3_joint1', 'right_finger3_joint2', 'right_finger3_joint3',
    'right_finger3_joint4',
    'right_finger4_joint1', 'right_finger4_joint2', 'right_finger4_joint3',
    'right_finger4_joint4',
    'right_finger5_joint1', 'right_finger5_joint2', 'right_finger5_joint3',
    'right_finger5_joint4',
)

WUJI_LIMITS = {
    'right_finger1_joint1': (0.0368, 1.6125),
    'right_finger1_joint2': (-0.1576, 0.9312),
    'right_finger1_joint3': (-0.4638, 1.5607),
    'right_finger1_joint4': (-0.4829, 1.5451),
    'right_finger2_joint1': (-0.1611, 1.5588),
    'right_finger2_joint2': (-0.4044, 0.3054),
    'right_finger2_joint3': (-0.4714, 1.5504),
    'right_finger2_joint4': (-0.4644, 1.5753),
    'right_finger3_joint1': (-0.1719, 1.5496),
    'right_finger3_joint2': (-0.4014, 0.2996),
    'right_finger3_joint3': (-0.4632, 1.5613),
    'right_finger3_joint4': (-0.4697, 1.5702),
    'right_finger4_joint1': (-0.1601, 1.5534),
    'right_finger4_joint2': (-0.4134, 0.3161),
    'right_finger4_joint3': (-0.4782, 1.5448),
    'right_finger4_joint4': (-0.4825, 1.5550),
    'right_finger5_joint1': (-0.1674, 1.5539),
    'right_finger5_joint2': (-0.4203, 0.2931),
    'right_finger5_joint3': (-0.4804, 1.5420),
    'right_finger5_joint4': (-0.4705, 1.5709),
}

WUJI_NEUTRAL_OPEN = {
    'right_finger1_joint1': 0.20,
    'right_finger1_joint2': 0.00,
    'right_finger1_joint3': -0.05,
    'right_finger1_joint4': -0.05,
    'right_finger2_joint1': 0.00,
    'right_finger2_joint2': -0.10,
    'right_finger2_joint3': -0.08,
    'right_finger2_joint4': -0.08,
    'right_finger3_joint1': 0.00,
    'right_finger3_joint2': -0.10,
    'right_finger3_joint3': -0.08,
    'right_finger3_joint4': -0.08,
    'right_finger4_joint1': 0.00,
    'right_finger4_joint2': -0.10,
    'right_finger4_joint3': -0.08,
    'right_finger4_joint4': -0.08,
    'right_finger5_joint1': 0.00,
    'right_finger5_joint2': -0.12,
    'right_finger5_joint3': -0.08,
    'right_finger5_joint4': -0.08,
}

FINGER_CHAINS = {
    'thumb': (1, 2, 3, 4),
    'index': (5, 6, 7, 8),
    'middle': (9, 10, 11, 12),
    'ring': (13, 14, 15, 16),
    'little': (17, 18, 19, 20),
}

FINGER_TO_WUJI_PREFIX = {
    'thumb': 'right_finger1',
    'index': 'right_finger2',
    'middle': 'right_finger3',
    'ring': 'right_finger4',
    'little': 'right_finger5',
}


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def convert_vec(vec: Iterable[float]) -> np.ndarray:
    return UNITY_TO_ROBOT @ np.array(vec, dtype=float)


def convert_quat(quat: Iterable[float]) -> np.ndarray:
    quat_array = normalize_quat(np.array(quat, dtype=float))
    unity_rot = Rotation.from_quat(quat_array).as_matrix()
    robot_rot = UNITY_TO_ROBOT @ unity_rot @ UNITY_TO_ROBOT.T
    return Rotation.from_matrix(robot_rot).as_quat()


def rotate_points(points: np.ndarray, quat: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    return Rotation.from_quat(normalize_quat(quat)).apply(points)


def angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm <= 1e-9:
        return 0.0
    dot = float(np.clip(np.dot(vec_a, vec_b) / norm, -1.0, 1.0))
    return math.acos(dot)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def parse_hts_line(line: str) -> Optional[Tuple[str, str, Tuple[float, ...]]]:
    if ':' not in line:
        return None
    label, payload = line.split(':', 1)
    label_lower = label.lower()
    if 'right' in label_lower:
        side = 'right'
    elif 'left' in label_lower:
        side = 'left'
    else:
        return None

    if 'wrist' in label_lower:
        kind = 'wrist'
    elif 'landmarks' in label_lower:
        kind = 'landmarks'
    else:
        return None

    values = tuple(float(match.group(0)) for match in FLOAT_PATTERN.finditer(payload))
    return side, kind, values


@dataclass
class HandState:
    side: str
    wrist_position: Optional[np.ndarray] = None
    wrist_quat: Optional[np.ndarray] = None
    landmarks_local: Optional[np.ndarray] = None
    last_wrist_time: float = field(default_factory=time.monotonic)
    last_landmarks_time: float = field(default_factory=time.monotonic)

    def update_wrist(self, values: Iterable[float]) -> bool:
        data = np.array(tuple(values), dtype=float)
        if data.size < 7:
            return False
        self.wrist_position = convert_vec(data[:3])
        self.wrist_quat = convert_quat(data[3:7])
        self.last_wrist_time = time.monotonic()
        return True

    def update_landmarks(self, values: Iterable[float]) -> bool:
        data = np.array(tuple(values), dtype=float)
        if data.size < 63:
            return False
        data = data[:63].reshape((21, 3))
        self.landmarks_local = (UNITY_TO_ROBOT @ data.T).T
        self.last_landmarks_time = time.monotonic()
        return True

    def has_recent_wrist(self, timeout: float) -> bool:
        return (
            self.wrist_position is not None
            and self.wrist_quat is not None
            and time.monotonic() - self.last_wrist_time <= timeout
        )

    def has_recent_landmarks(self, timeout: float) -> bool:
        return (
            self.landmarks_local is not None
            and time.monotonic() - self.last_landmarks_time <= timeout
        )

    def world_landmarks(self) -> Optional[np.ndarray]:
        if self.landmarks_local is None:
            return None
        if self.wrist_position is None or self.wrist_quat is None:
            return self.landmarks_local
        return rotate_points(self.landmarks_local, self.wrist_quat) + self.wrist_position


class HtsFrankaWujiTeleop(Node):
    def __init__(self) -> None:
        super().__init__('hts_franka_wuji_teleop')

        self.declare_parameter('protocol', 'tcp')
        self.declare_parameter('udp_host', '0.0.0.0')
        self.declare_parameter('udp_port', 9000)
        self.declare_parameter('tcp_host', '0.0.0.0')
        self.declare_parameter('tcp_port', 8000)
        self.declare_parameter('control_hand', 'right')
        self.declare_parameter('pause_hand', 'left')
        self.declare_parameter('control_rate', 100.0)
        self.declare_parameter('stream_timeout', 0.5)
        self.declare_parameter('position_scale', 1.0)
        self.declare_parameter('max_position_delta', 0.45)
        self.declare_parameter('position_smoothing', 0.2)
        self.declare_parameter('orientation_smoothing', 0.3)
        self.declare_parameter('control_orientation', True)
        self.declare_parameter('fist_close_threshold', 0.75)
        self.declare_parameter('fist_open_threshold', 0.45)
        self.declare_parameter('pause_debounce_time', 0.2)
        self.declare_parameter('curl_closed_flexion', 1.8)
        self.declare_parameter('arm_enabled', True)
        self.declare_parameter('wuji_enabled', True)
        self.declare_parameter('require_robot_state', True)
        self.declare_parameter('set_cartesian_stiffness', True)
        self.declare_parameter('cartesian_param_service', '/cartesian_impedance_controller/set_parameters')
        self.declare_parameter('pos_stiffness', 80.0)
        self.declare_parameter('rot_stiffness', 20.0)
        self.declare_parameter('robot_state_topic', '/franka_robot_state_broadcaster/robot_state')
        self.declare_parameter('arm_command_topic', '/cartesian_impedance/pose_desired')
        self.declare_parameter('wuji_command_topic', '/wuji_joint_position_controller/commands')
        self.declare_parameter('fallback_position', [0.39, 0.0, 0.555])
        self.declare_parameter('fallback_rpy', [2.897246558, 0.0, 0.0])

        self.protocol = str(self.get_parameter('protocol').value).lower()
        self.udp_host = self.get_parameter('udp_host').value
        self.udp_port = int(self.get_parameter('udp_port').value)
        self.tcp_host = self.get_parameter('tcp_host').value
        self.tcp_port = int(self.get_parameter('tcp_port').value)
        self.control_hand = str(self.get_parameter('control_hand').value).lower()
        self.pause_hand = str(self.get_parameter('pause_hand').value).lower()
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.stream_timeout = float(self.get_parameter('stream_timeout').value)
        self.position_scale = float(self.get_parameter('position_scale').value)
        self.max_position_delta = float(self.get_parameter('max_position_delta').value)
        self.position_smoothing = float(self.get_parameter('position_smoothing').value)
        self.orientation_smoothing = float(self.get_parameter('orientation_smoothing').value)
        self.control_orientation = bool(self.get_parameter('control_orientation').value)
        self.fist_close_threshold = float(self.get_parameter('fist_close_threshold').value)
        self.fist_open_threshold = float(self.get_parameter('fist_open_threshold').value)
        self.pause_debounce_time = float(self.get_parameter('pause_debounce_time').value)
        self.curl_closed_flexion = float(self.get_parameter('curl_closed_flexion').value)
        self.arm_enabled = bool(self.get_parameter('arm_enabled').value)
        self.wuji_enabled = bool(self.get_parameter('wuji_enabled').value)
        self.require_robot_state = bool(self.get_parameter('require_robot_state').value)
        self.set_cartesian_stiffness = bool(self.get_parameter('set_cartesian_stiffness').value)
        self.pos_stiffness = float(self.get_parameter('pos_stiffness').value)
        self.rot_stiffness = float(self.get_parameter('rot_stiffness').value)
        self.stiffness_configured = False

        self.hands: Dict[str, HandState] = {
            'right': HandState('right'),
            'left': HandState('left'),
        }
        self.lock = threading.Lock()
        self.running = True
        self.receiver_thread = threading.Thread(target=self._receiver, daemon=True)
        self.tcp_connection_threads = []

        self.arm_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter('arm_command_topic').value), 10)
        self.wuji_pub = self.create_publisher(
            Float64MultiArray, str(self.get_parameter('wuji_command_topic').value), 10)
        self.vr_wrist_pub = self.create_publisher(PoseStamped, '/vr/control_wrist_pose', 10)
        self.robot_target_pub = self.create_publisher(PoseStamped, '/vr/robot_target_pose', 10)
        self.right_landmarks_pub = self.create_publisher(Float64MultiArray, '/vr/right_landmarks', 10)
        self.left_landmarks_pub = self.create_publisher(Float64MultiArray, '/vr/left_landmarks', 10)
        self.pause_pub = self.create_publisher(Bool, '/vr_teleop/paused', 10)
        self.status_pub = self.create_publisher(String, '/vr_teleop/status', 10)
        self.diagnostics_pub = self.create_publisher(String, '/vr_teleop/diagnostics', 10)

        cartesian_param_service = str(self.get_parameter('cartesian_param_service').value)
        self.cartesian_param_client = self.create_client(SetParameters, cartesian_param_service)

        robot_state_topic = str(self.get_parameter('robot_state_topic').value)
        self.robot_state_sub = self.create_subscription(
            FrankaState, robot_state_topic, self._robot_state_callback, 10)

        fallback_position = np.array(self.get_parameter('fallback_position').value, dtype=float)
        fallback_rpy = np.array(self.get_parameter('fallback_rpy').value, dtype=float)
        self.latest_robot_position: Optional[np.ndarray] = None
        self.latest_robot_rotation: Optional[Rotation] = None
        self.fallback_position = fallback_position
        self.fallback_rotation = Rotation.from_euler('xyz', fallback_rpy)

        self.initial_vr_position: Optional[np.ndarray] = None
        self.initial_vr_rotation: Optional[Rotation] = None
        self.base_robot_position: Optional[np.ndarray] = None
        self.base_robot_rotation: Optional[Rotation] = None
        self.target_position: Optional[np.ndarray] = None
        self.target_rotation: Optional[Rotation] = None
        self.smoothed_delta = np.zeros(3, dtype=float)
        self.smoothed_relative_rotation = Rotation.identity()
        self.paused = False
        self.last_pause_state = False
        self.pause_candidate_state: Optional[bool] = None
        self.pause_candidate_since = time.monotonic()
        self.last_pause_metric = 0.0
        self.last_log_time = time.monotonic()
        self.rx_count = 0
        self.arm_command_count = 0
        self.wuji_command_count = 0
        self.last_wuji_command = self.open_wuji_command()

        self.timer = self.create_timer(1.0 / self.control_rate, self._control_timer)
        self.stiffness_timer = self.create_timer(1.0, self._configure_cartesian_stiffness)
        self.receiver_thread.start()

        if self.protocol == 'tcp':
            endpoint = f'TCP {self.tcp_host}:{self.tcp_port}'
        else:
            endpoint = f'UDP {self.udp_host}:{self.udp_port}'
        self.get_logger().info(
            f'Listening for HTS {endpoint}. Control hand: {self.control_hand}, '
            f'pause hand: {self.pause_hand}.')

    def destroy_node(self):
        self.running = False
        if self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)
        for thread in list(self.tcp_connection_threads):
            if thread.is_alive():
                thread.join(timeout=0.2)
        super().destroy_node()

    def _receiver(self) -> None:
        if self.protocol == 'tcp':
            self._tcp_receiver()
        elif self.protocol == 'udp':
            self._udp_receiver()
        else:
            self.get_logger().error(
                f"Unsupported protocol '{self.protocol}'. Use 'tcp' or 'udp'.")

    def _udp_receiver(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind((self.udp_host, self.udp_port))
            sock.setblocking(False)
        except OSError as exc:
            self.get_logger().error(
                f'Failed to bind HTS UDP socket on {self.udp_host}:{self.udp_port}: {exc}')
            return

        try:
            while self.running and rclpy.ok():
                ready, _, _ = select.select([sock], [], [], 0.1)
                if not ready:
                    continue
                try:
                    data, _addr = sock.recvfrom(65536)
                except BlockingIOError:
                    continue
                try:
                    message = data.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                for line in message.splitlines():
                    self._process_hts_line(line.strip())
        finally:
            sock.close()

    def _tcp_receiver(self) -> None:
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.tcp_host, self.tcp_port))
            server_sock.listen(5)
            server_sock.settimeout(0.2)
        except OSError as exc:
            self.get_logger().error(
                f'Failed to bind HTS TCP server on {self.tcp_host}:{self.tcp_port}: {exc}')
            return

        self.get_logger().info(f'HTS TCP server ready on {self.tcp_host}:{self.tcp_port}')
        try:
            while self.running and rclpy.ok():
                try:
                    conn, addr = server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                thread = threading.Thread(
                    target=self._handle_tcp_connection, args=(conn, addr), daemon=True)
                self.tcp_connection_threads.append(thread)
                thread.start()
        finally:
            server_sock.close()

    def _handle_tcp_connection(self, conn: socket.socket, addr) -> None:
        self.get_logger().info(f'HTS TCP connection from {addr}')
        buffer = ''
        with conn:
            conn.settimeout(0.2)
            while self.running and rclpy.ok():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                try:
                    buffer += data.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_hts_line(line.strip())
        self.get_logger().info(f'HTS TCP connection closed from {addr}')

    def _process_hts_line(self, line: str) -> None:
        if not line:
            return
        parsed = parse_hts_line(line)
        if not parsed:
            return
        side, kind, values = parsed
        with self.lock:
            self.rx_count += 1
            hand = self.hands.get(side)
            if hand is None:
                return
            if kind == 'wrist':
                hand.update_wrist(values)
            elif kind == 'landmarks':
                hand.update_landmarks(values)

    def _robot_state_callback(self, msg: FrankaState) -> None:
        transform = np.array(msg.o_t_ee, dtype=float).reshape((4, 4), order='F')
        self.latest_robot_position = transform[:3, 3].copy()
        self.latest_robot_rotation = Rotation.from_matrix(transform[:3, :3])

    def _configure_cartesian_stiffness(self) -> None:
        if not self.set_cartesian_stiffness or self.stiffness_configured:
            return
        if not self.cartesian_param_client.service_is_ready():
            if not self.cartesian_param_client.wait_for_service(timeout_sec=0.01):
                return

        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name='pos_stiff',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=self.pos_stiffness,
                ),
            ),
            Parameter(
                name='rot_stiff',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=self.rot_stiffness,
                ),
            ),
        ]
        future = self.cartesian_param_client.call_async(request)
        future.add_done_callback(self._stiffness_response_callback)
        self.stiffness_configured = True

    def _stiffness_response_callback(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.stiffness_configured = False
            self.get_logger().warn(f'Cartesian stiffness request failed: {exc}')
            return
        if all(result.successful for result in response.results):
            self.get_logger().info(
                f'Set Cartesian stiffness pos={self.pos_stiffness}, rot={self.rot_stiffness}')
        else:
            self.stiffness_configured = False
            reasons = ', '.join(result.reason for result in response.results if result.reason)
            self.get_logger().warn(f'Cartesian stiffness request rejected: {reasons}')

    def _control_timer(self) -> None:
        with self.lock:
            control_hand = self._copy_hand(self.control_hand)
            pause_hand = self._copy_hand(self.pause_hand)

        self._publish_debug_landmarks(control_hand)
        self._publish_debug_landmarks(pause_hand)

        self._update_pause_state(pause_hand, control_hand)
        self.pause_pub.publish(Bool(data=self.paused))

        if self.arm_enabled and not self.paused:
            self._update_and_publish_arm(control_hand)

        if self.wuji_enabled:
            self._update_and_publish_wuji(control_hand)

        self._log_rate()

    def _copy_hand(self, side: str) -> HandState:
        hand = self.hands[side]
        copied = HandState(side)
        copied.wrist_position = None if hand.wrist_position is None else hand.wrist_position.copy()
        copied.wrist_quat = None if hand.wrist_quat is None else hand.wrist_quat.copy()
        copied.landmarks_local = None if hand.landmarks_local is None else hand.landmarks_local.copy()
        copied.last_wrist_time = hand.last_wrist_time
        copied.last_landmarks_time = hand.last_landmarks_time
        return copied

    def _robot_base_ready(self) -> bool:
        return self.latest_robot_position is not None and self.latest_robot_rotation is not None

    def _get_robot_base_pose(self) -> Optional[Tuple[np.ndarray, Rotation]]:
        if self._robot_base_ready():
            return self.latest_robot_position.copy(), self.latest_robot_rotation
        if self.require_robot_state:
            return None
        return self.fallback_position.copy(), self.fallback_rotation

    def _initialize_relative_control(self, hand: HandState) -> bool:
        if not hand.has_recent_wrist(self.stream_timeout):
            return False

        base_pose = self._get_robot_base_pose()
        if base_pose is None:
            return False

        self.initial_vr_position = hand.wrist_position.copy()
        self.initial_vr_rotation = Rotation.from_quat(hand.wrist_quat)
        self.base_robot_position, self.base_robot_rotation = base_pose
        self.target_position = self.base_robot_position.copy()
        self.target_rotation = self.base_robot_rotation
        self.smoothed_delta = np.zeros(3, dtype=float)
        self.smoothed_relative_rotation = Rotation.identity()
        self._publish_status('relative control initialized')
        return True

    def _rebase_after_pause(self, hand: HandState) -> None:
        if not hand.has_recent_wrist(self.stream_timeout):
            return
        if self.target_position is None or self.target_rotation is None:
            self._initialize_relative_control(hand)
            return
        self.initial_vr_position = hand.wrist_position.copy()
        self.initial_vr_rotation = Rotation.from_quat(hand.wrist_quat)
        self.base_robot_position = self.target_position.copy()
        self.base_robot_rotation = self.target_rotation
        self.smoothed_delta = np.zeros(3, dtype=float)
        self.smoothed_relative_rotation = Rotation.identity()

    def _update_and_publish_arm(self, hand: HandState) -> None:
        if not hand.has_recent_wrist(self.stream_timeout):
            return
        if (
            self.initial_vr_position is None
            or self.initial_vr_rotation is None
            or self.base_robot_position is None
            or self.base_robot_rotation is None
        ):
            if not self._initialize_relative_control(hand):
                return

        vr_delta = (hand.wrist_position - self.initial_vr_position) * self.position_scale
        delta_norm = np.linalg.norm(vr_delta)
        if delta_norm > self.max_position_delta:
            vr_delta = vr_delta / delta_norm * self.max_position_delta

        pos_alpha = clamp(self.position_smoothing, 0.0, 0.99)
        self.smoothed_delta = pos_alpha * self.smoothed_delta + (1.0 - pos_alpha) * vr_delta
        self.target_position = self.base_robot_position + self.smoothed_delta

        current_rot = Rotation.from_quat(hand.wrist_quat)
        relative_rotation = current_rot * self.initial_vr_rotation.inv()
        ori_alpha = clamp(self.orientation_smoothing, 0.0, 0.99)
        self.smoothed_relative_rotation = self._slerp_rotation(
            self.smoothed_relative_rotation, relative_rotation, 1.0 - ori_alpha)

        if self.control_orientation:
            self.target_rotation = self.smoothed_relative_rotation * self.base_robot_rotation
        else:
            self.target_rotation = self.base_robot_rotation

        msg = Float64MultiArray()
        msg.data = (
            self.target_position.tolist()
            + self.target_rotation.as_matrix().flatten().tolist()
            + [1.0]
        )
        self.arm_pub.publish(msg)
        self.arm_command_count += 1
        self._publish_pose(self.vr_wrist_pub, hand.wrist_position, Rotation.from_quat(hand.wrist_quat))
        self._publish_pose(self.robot_target_pub, self.target_position, self.target_rotation)

    def _slerp_rotation(self, start: Rotation, end: Rotation, ratio: float) -> Rotation:
        ratio = clamp(ratio, 0.0, 1.0)
        rotations = Rotation.from_quat([start.as_quat(), end.as_quat()])
        return Slerp([0.0, 1.0], rotations)([ratio])[0]

    def _publish_pose(self, publisher, position: np.ndarray, rotation: Rotation) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'panda_link0'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        quat = rotation.as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        publisher.publish(msg)

    def _update_and_publish_wuji(self, hand: HandState) -> None:
        if not hand.has_recent_landmarks(self.stream_timeout):
            return
        curls = self._finger_curls(hand.landmarks_local)
        command = self._curls_to_wuji_command(curls)
        self.last_wuji_command = command
        self.wuji_pub.publish(Float64MultiArray(data=command))
        self.wuji_command_count += 1

    def _finger_curls(self, landmarks: Optional[np.ndarray]) -> Dict[str, float]:
        if landmarks is None or landmarks.shape[0] < 21:
            return {name: 0.0 for name in FINGER_CHAINS}

        curls = {}
        for name, chain in FINGER_CHAINS.items():
            points = [landmarks[index] for index in chain]
            flexion = 0.0
            for idx in range(1, len(points) - 1):
                vec_a = points[idx - 1] - points[idx]
                vec_b = points[idx + 1] - points[idx]
                flexion += max(0.0, math.pi - angle_between(vec_a, vec_b))
            curls[name] = clamp(flexion / max(self.curl_closed_flexion, 1e-3), 0.0, 1.0)
        return curls

    def _fist_metric(self, hand: HandState) -> Optional[float]:
        if not hand.has_recent_landmarks(self.stream_timeout):
            return None
        landmarks = hand.landmarks_local
        if landmarks is None or landmarks.shape[0] < 21:
            return None

        curls = self._finger_curls(hand.landmarks_local)
        angle_curl = float(np.mean([curls['index'], curls['middle'], curls['ring'], curls['little']]))

        distance_curls = []
        for finger_name in ('index', 'middle', 'ring', 'little'):
            chain = FINGER_CHAINS[finger_name]
            points = [landmarks[index] for index in chain]
            path_length = sum(
                np.linalg.norm(points[idx + 1] - points[idx]) for idx in range(len(points) - 1)
            )
            tip_distance = np.linalg.norm(points[-1] - points[0])
            if path_length <= 1e-6:
                continue
            distance_curls.append(clamp(1.0 - tip_distance / path_length, 0.0, 1.0))

        distance_curl = float(np.mean(distance_curls)) if distance_curls else angle_curl
        return clamp(0.7 * angle_curl + 0.3 * distance_curl, 0.0, 1.0)

    def _update_pause_state(self, pause_hand: HandState, control_hand: HandState) -> None:
        metric = self._fist_metric(pause_hand)
        if metric is None:
            return
        self.last_pause_metric = metric

        if self.paused:
            desired_paused = metric > self.fist_open_threshold
        else:
            desired_paused = metric >= self.fist_close_threshold

        now = time.monotonic()
        if desired_paused == self.paused:
            self.pause_candidate_state = None
            self.pause_candidate_since = now
            return

        if self.pause_candidate_state != desired_paused:
            self.pause_candidate_state = desired_paused
            self.pause_candidate_since = now
            return

        if now - self.pause_candidate_since < self.pause_debounce_time:
            return

        self.paused = desired_paused
        self.pause_candidate_state = None
        self.pause_candidate_since = now
        if self.paused:
            self._publish_status(f'paused: left fist metric {metric:.2f}')
        else:
            self._rebase_after_pause(control_hand)
            self._publish_status(f'resumed: left fist metric {metric:.2f}')

    def _curls_to_wuji_command(self, curls: Dict[str, float]) -> list:
        command_by_joint = {}
        for finger_name, prefix in FINGER_TO_WUJI_PREFIX.items():
            curl_value = curls.get(finger_name, 0.0)
            if finger_name == 'thumb':
                weights = (0.75, 0.45, 0.85, 0.85)
            else:
                weights = (1.0, 0.25, 0.9, 0.9)

            for joint_index, weight in enumerate(weights, start=1):
                joint_name = f'{prefix}_joint{joint_index}'
                lower, upper = WUJI_LIMITS[joint_name]
                neutral = WUJI_NEUTRAL_OPEN[joint_name]
                value = neutral + curl_value * weight * (upper - neutral)
                command_by_joint[joint_name] = clamp(value, lower, upper)

        return [command_by_joint[joint_name] for joint_name in WUJI_JOINT_ORDER]

    def open_wuji_command(self) -> list:
        return [WUJI_NEUTRAL_OPEN[joint_name] for joint_name in WUJI_JOINT_ORDER]

    def _publish_debug_landmarks(self, hand: HandState) -> None:
        if not hand.has_recent_landmarks(self.stream_timeout):
            return
        points = hand.world_landmarks()
        if points is None:
            return
        msg = Float64MultiArray()
        msg.data = points.reshape(-1).tolist()
        if hand.side == 'right':
            self.right_landmarks_pub.publish(msg)
        elif hand.side == 'left':
            self.left_landmarks_pub.publish(msg)

    def _publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _log_rate(self) -> None:
        now = time.monotonic()
        if now - self.last_log_time < 2.0:
            return
        with self.lock:
            rx_count = self.rx_count
            self.rx_count = 0
        arm_count = self.arm_command_count
        wuji_count = self.wuji_command_count
        self.arm_command_count = 0
        self.wuji_command_count = 0
        rate = rx_count / max(now - self.last_log_time, 1e-3)
        arm_rate = arm_count / max(now - self.last_log_time, 1e-3)
        wuji_rate = wuji_count / max(now - self.last_log_time, 1e-3)
        self.last_log_time = now
        status = 'paused' if self.paused else 'active'
        diagnostic = (
            f'protocol={self.protocol} hts_lines={rate:.1f}/s arm_cmd={arm_rate:.1f}/s '
            f'wuji_cmd={wuji_rate:.1f}/s pause_metric={self.last_pause_metric:.2f} '
            f'teleop={status}'
        )
        self.diagnostics_pub.publish(String(data=diagnostic))
        self.get_logger().info(diagnostic)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HtsFrankaWujiTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
