// Execute a surgical manipulation primitive as a scripted pose sequence.
//
//   ros2 run surgical_hand_skills primitive_cli list
//   ros2 run surgical_hand_skills primitive_cli needle_driver_acquisition
//
// Works against any backend (mock / MuJoCo / later real hardware) because
// it only publishes to the hand controller topic. During each step the
// command is re-published at 20 Hz (single volatile messages can be lost;
// the forward controller holds the last value).
//
// FEEDBACK (skeleton stage): if the tension estimator and/or the tactile
// publisher are running, their latest values are logged at every step
// transition, and guard violations (max_tension_n, require_contact) are
// reported as warnings. Guards do NOT abort the sequence yet — enforcement
// is deliberately deferred until the tension estimator is bench-calibrated
// and real tactile hardware exists.

#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <surgical_hand_msgs/msg/tactile_state.hpp>
#include <surgical_hand_msgs/msg/tendon_tension.hpp>

#include "surgical_hand_skills/pose_library.hpp"
#include "surgical_hand_skills/primitive_engine.hpp"

namespace shs = surgical_hand_skills;
using surgical_hand_msgs::msg::TactileState;
using surgical_hand_msgs::msg::TendonTension;

namespace {

class PrimitiveRunner : public rclcpp::Node {
 public:
  PrimitiveRunner() : rclcpp::Node("primitive_runner") {
    topic_ = declare_parameter<std::string>("topic",
                                            "/hand_joint_position_controller/commands");
    tension_topic_ = declare_parameter<std::string>("tension_topic",
                                                    "/tension_estimator/tension");
    tactile_topics_ = declare_parameter<std::vector<std::string>>(
        "tactile_topics", {"/tactile_publisher/thumb", "/tactile_publisher/index",
                           "/tactile_publisher/middle"});

    publisher_ = create_publisher<std_msgs::msg::Float64MultiArray>(topic_, 10);
    tension_sub_ = create_subscription<TendonTension>(
        tension_topic_, 10, [this](const TendonTension& msg) { last_tension_ = msg; });
    for (const auto& topic : tactile_topics_) {
      tactile_subs_.push_back(create_subscription<TactileState>(
          topic, 10, [this](const TactileState& msg) { last_tactile_[msg.sensor_name] = msg; }));
    }
  }

  bool waitForController(std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (publisher_->get_subscription_count() == 0 &&
           std::chrono::steady_clock::now() < deadline && rclcpp::ok()) {
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    return publisher_->get_subscription_count() > 0;
  }

  void runStep(const shs::PoseLibrary& library, const shs::PrimitiveStep& step) {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = library.buildCommand(step.pose, step.overrides);

    RCLCPP_INFO(get_logger(), "step '%s': pose '%s'%s for %.1f s", step.name.c_str(),
                step.pose.c_str(), step.overrides.empty() ? "" : " (with overrides)",
                step.duration_s);

    const auto step_end = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(step.duration_s));
    while (std::chrono::steady_clock::now() < step_end && rclcpp::ok()) {
      publisher_->publish(msg);
      rclcpp::spin_some(get_node_base_interface());  // service feedback subscriptions
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }
    logFeedbackAndGuards(step);
  }

 private:
  void logFeedbackAndGuards(const shs::PrimitiveStep& step) {
    if (last_tension_) {
      RCLCPP_INFO(get_logger(), "  tension: %.2f N (status %u, confidence %.2f)",
                  last_tension_->tension_n, last_tension_->status, last_tension_->confidence);
      if (step.max_tension_n && last_tension_->tension_n > *step.max_tension_n) {
        RCLCPP_WARN(get_logger(),
                    "  GUARD (not enforced yet): tension %.2f N exceeds step cap %.2f N",
                    last_tension_->tension_n, *step.max_tension_n);
      }
    } else if (step.max_tension_n) {
      RCLCPP_INFO(get_logger(), "  tension guard %.2f N declared, but no estimator heard on %s",
                  *step.max_tension_n, tension_topic_.c_str());
    }

    bool any_contact = false;
    for (const auto& [sensor, state] : last_tactile_) {
      any_contact |= state.contact;
      RCLCPP_INFO(get_logger(), "  tactile[%s]: contact=%s normal=%.2f N slip=%.2f%s",
                  sensor.c_str(), state.contact ? "yes" : "no", state.normal_force_n,
                  state.slip_probability, state.simulated ? " (simulated)" : "");
    }
    if (step.require_contact) {
      if (last_tactile_.empty()) {
        RCLCPP_INFO(get_logger(), "  contact guard declared, but no tactile source heard");
      } else if (!any_contact) {
        RCLCPP_WARN(get_logger(), "  GUARD (not enforced yet): step expects contact, none seen");
      }
    }
  }

  std::string topic_;
  std::string tension_topic_;
  std::vector<std::string> tactile_topics_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
  rclcpp::Subscription<TendonTension>::SharedPtr tension_sub_;
  std::vector<rclcpp::Subscription<TactileState>::SharedPtr> tactile_subs_;
  std::optional<TendonTension> last_tension_;
  std::map<std::string, TactileState> last_tactile_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);

  const std::string skills_share =
      ament_index_cpp::get_package_share_directory("surgical_hand_skills");
  const std::string description_share =
      ament_index_cpp::get_package_share_directory("surgical_hand_description");

  try {
    const shs::PoseLibrary library(description_share + "/config/hand_joints.yaml",
                                   skills_share + "/config/named_poses.yaml");
    const auto primitives =
        shs::loadPrimitives(skills_share + "/config/primitives.yaml", library);

    if (args.size() < 2 || args[1] == "list") {
      std::printf("Available primitives:\n");
      for (const auto& primitive : primitives) {
        std::printf("  %-26s %zu steps  %s\n", primitive.name.c_str(),
                    primitive.steps.size(), primitive.description.c_str());
      }
      rclcpp::shutdown();
      return args.size() < 2 ? 2 : 0;
    }

    const auto it = std::find_if(primitives.begin(), primitives.end(),
                                 [&](const shs::Primitive& p) { return p.name == args[1]; });
    if (it == primitives.end()) {
      std::fprintf(stderr, "error: unknown primitive '%s' (try: primitive_cli list)\n",
                   args[1].c_str());
      rclcpp::shutdown();
      return 2;
    }

    auto runner = std::make_shared<PrimitiveRunner>();
    if (!runner->waitForController(std::chrono::seconds(5))) {
      std::fprintf(stderr, "error: no subscriber on the hand controller topic. "
                           "Is a hand backend running?\n");
      rclcpp::shutdown();
      return 1;
    }

    RCLCPP_INFO(runner->get_logger(), "running primitive '%s' (%zu steps): %s",
                it->name.c_str(), it->steps.size(), it->description.c_str());
    for (const auto& step : it->steps) {
      if (!rclcpp::ok()) {
        break;
      }
      runner->runStep(library, step);
    }
    RCLCPP_INFO(runner->get_logger(), "primitive '%s' finished", it->name.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
