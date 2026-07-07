// hand_pose_cli: send a named hand pose to the hand joint position controller.
//
// Usage:
//   ros2 run surgical_hand_skills hand_pose_cli list
//   ros2 run surgical_hand_skills hand_pose_cli <pose_name>
//
// Reads:
//   surgical_hand_description/config/hand_joints.yaml  (joint order, limits,
//                                                       aliases, neutrals)
//   surgical_hand_skills/config/named_poses.yaml       (pose definitions)
//
// Publishes one std_msgs/Float64MultiArray on
// /hand_joint_position_controller/commands, ordered exactly as the joint
// list in hand_joints.yaml (the same order the controller is configured
// with at launch time).
//
// This tool is backend-agnostic: it works identically against the mock,
// MuJoCo, and (later) real backends because it only talks to the
// ros2_control controller topic. Safety gating for real hardware lives in
// the hardware layer, not here.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <yaml-cpp/yaml.h>

namespace {

struct HandJoint {
  std::string name;
  std::string alias;
  double lower{0.0};
  double upper{0.0};
  double initial_position{0.0};
};

std::vector<HandJoint> loadHandJoints(const std::string& config_path) {
  const YAML::Node config = YAML::LoadFile(config_path);
  std::vector<HandJoint> joints;
  for (const auto& entry : config["joints"]) {
    HandJoint joint;
    joint.name = entry["name"].as<std::string>();
    joint.alias = entry["alias"].as<std::string>();
    joint.lower = entry["lower"].as<double>();
    joint.upper = entry["upper"].as<double>();
    joint.initial_position = entry["initial_position"].as<double>();
    joints.push_back(joint);
  }
  return joints;
}

YAML::Node loadPoses(const std::string& poses_path) {
  return YAML::LoadFile(poses_path)["poses"];
}

int listPoses(const YAML::Node& poses) {
  std::printf("Available poses:\n");
  for (const auto& pose : poses) {
    std::printf("  %s\n", pose.first.as<std::string>().c_str());
  }
  return 0;
}

// Build the command vector in hand_joints.yaml order. Joints the pose does
// not mention hold their initial (neutral) position. Values are clamped to
// the joint limits, with a warning, so a bad pose file cannot command an
// out-of-range position.
std::vector<double> buildCommand(const std::vector<HandJoint>& joints,
                                 const YAML::Node& pose) {
  std::vector<double> command;
  command.reserve(joints.size());
  for (const auto& joint : joints) {
    double value = joint.initial_position;
    if (pose[joint.alias]) {
      value = pose[joint.alias].as<double>();
    }
    const double clamped = std::clamp(value, joint.lower, joint.upper);
    if (clamped != value) {
      std::fprintf(stderr,
                   "warning: %s (%s) value %.4f outside [%.4f, %.4f]; clamped to %.4f\n",
                   joint.alias.c_str(), joint.name.c_str(), value, joint.lower,
                   joint.upper, clamped);
    }
    command.push_back(clamped);
  }
  return command;
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);

  if (args.size() < 2) {
    std::fprintf(stderr, "usage: hand_pose_cli <pose_name>|list\n");
    rclcpp::shutdown();
    return 2;
  }
  const std::string pose_name = args[1];

  const std::string joints_path =
      ament_index_cpp::get_package_share_directory("surgical_hand_description") +
      "/config/hand_joints.yaml";
  const std::string poses_path =
      ament_index_cpp::get_package_share_directory("surgical_hand_skills") +
      "/config/named_poses.yaml";

  const YAML::Node poses = loadPoses(poses_path);
  if (pose_name == "list") {
    const int result = listPoses(poses);
    rclcpp::shutdown();
    return result;
  }
  if (!poses[pose_name]) {
    std::fprintf(stderr, "error: unknown pose \"%s\" (try: hand_pose_cli list)\n",
                 pose_name.c_str());
    rclcpp::shutdown();
    return 2;
  }

  const std::vector<HandJoint> joints = loadHandJoints(joints_path);
  const std::vector<double> command = buildCommand(joints, poses[pose_name]);

  auto node = rclcpp::Node::make_shared("hand_pose_cli");
  const std::string topic = node->declare_parameter<std::string>(
      "topic", "/hand_joint_position_controller/commands");
  auto publisher = node->create_publisher<std_msgs::msg::Float64MultiArray>(topic, 10);

  // Wait for the controller to be subscribed so the one-shot message is not
  // lost to discovery timing.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (publisher->get_subscription_count() == 0 &&
         std::chrono::steady_clock::now() < deadline && rclcpp::ok()) {
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }
  if (publisher->get_subscription_count() == 0) {
    std::fprintf(stderr,
                 "error: no subscriber on %s after 5 s. Is the hand stack "
                 "running (surgical_hand_bringup)?\n",
                 topic.c_str());
    rclcpp::shutdown();
    return 1;
  }

  std_msgs::msg::Float64MultiArray msg;
  msg.data = command;
  publisher->publish(msg);

  std::printf("sent pose \"%s\" to %s:\n", pose_name.c_str(), topic.c_str());
  for (size_t i = 0; i < joints.size(); ++i) {
    std::printf("  %-12s % .4f\n", joints[i].alias.c_str(), command[i]);
  }

  // Give the middleware time to flush before exiting.
  rclcpp::sleep_for(std::chrono::milliseconds(300));
  rclcpp::shutdown();
  return 0;
}
