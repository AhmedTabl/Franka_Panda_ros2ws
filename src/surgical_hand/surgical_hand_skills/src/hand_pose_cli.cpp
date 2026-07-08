// hand_pose_cli: send a named hand pose to the hand joint position controller.
//
// Usage:
//   ros2 run surgical_hand_skills hand_pose_cli list
//   ros2 run surgical_hand_skills hand_pose_cli <pose_name>
//
// Backend-agnostic: works identically against the mock, MuJoCo, and (later)
// real backends because it only talks to the ros2_control controller topic.
// Safety gating for real hardware lives in the hardware layer, not here.
// Pose loading/validation lives in pose_library (shared with primitive_cli).

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "surgical_hand_skills/pose_library.hpp"

namespace shs = surgical_hand_skills;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);

  if (args.size() < 2) {
    std::fprintf(stderr, "usage: hand_pose_cli <pose_name>|list\n");
    rclcpp::shutdown();
    return 2;
  }
  const std::string pose_name = args[1];

  try {
    const shs::PoseLibrary library(
        ament_index_cpp::get_package_share_directory("surgical_hand_description") +
            "/config/hand_joints.yaml",
        ament_index_cpp::get_package_share_directory("surgical_hand_skills") +
            "/config/named_poses.yaml");

    if (pose_name == "list") {
      std::printf("Available poses:\n");
      for (const auto& name : library.poseNames()) {
        std::printf("  %s\n", name.c_str());
      }
      rclcpp::shutdown();
      return 0;
    }
    if (!library.hasPose(pose_name)) {
      std::fprintf(stderr, "error: unknown pose \"%s\" (try: hand_pose_cli list)\n",
                   pose_name.c_str());
      rclcpp::shutdown();
      return 2;
    }

    std::vector<std::string> clamped;
    const std::vector<double> command = library.buildCommand(pose_name, {}, &clamped);
    for (const auto& alias : clamped) {
      std::fprintf(stderr, "warning: %s clamped to its joint limits\n", alias.c_str());
    }

    auto node = rclcpp::Node::make_shared("hand_pose_cli");
    const std::string topic = node->declare_parameter<std::string>(
        "topic", "/hand_joint_position_controller/commands");
    auto publisher = node->create_publisher<std_msgs::msg::Float64MultiArray>(topic, 10);

    // Wait for the controller so the command is not lost to discovery.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (publisher->get_subscription_count() == 0 &&
           std::chrono::steady_clock::now() < deadline && rclcpp::ok()) {
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    if (publisher->get_subscription_count() == 0) {
      std::fprintf(stderr,
                   "error: no subscriber on %s after 5 s. Is the hand stack running?\n",
                   topic.c_str());
      rclcpp::shutdown();
      return 1;
    }

    // Publish the (identical) command repeatedly for a short window: a
    // single volatile message can be lost against a busy multithreaded
    // executor (observed with the MuJoCo backend), and the forward
    // controller holds the last received command anyway.
    std_msgs::msg::Float64MultiArray msg;
    msg.data = command;
    for (int i = 0; i < 20 && rclcpp::ok(); ++i) {
      publisher->publish(msg);
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }

    std::printf("sent pose \"%s\" to %s:\n", pose_name.c_str(), topic.c_str());
    const auto& joints = library.joints();
    for (size_t i = 0; i < joints.size(); ++i) {
      std::printf("  %-12s % .4f\n", joints[i].alias.c_str(), command[i]);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
