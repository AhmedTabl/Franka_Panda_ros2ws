// Read-only XC330 state publisher.
//
// Publishes at rate_hz:
//   ~/joint_state    sensor_msgs/JointState (position rad, velocity rad/s;
//                    effort intentionally left empty — publish torque there
//                    only once the tension estimator is characterized)
//   ~/motor_current  std_msgs/Float64 [A] — remap onto
//                    /tension_estimator/motor_current to chain the tension
//                    estimator onto the real motor:
//
//   ros2 run surgical_hand_xc330 xc330_state_publisher --ros-args \
//     -p port:=/dev/ttyUSB0 -p baud:=57600 -p id:=1 \
//     -r ~/motor_current:=/tension_estimator/motor_current
//
// This node NEVER writes to the motor. There is no default port on purpose.

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

#include "surgical_hand_xc330/xc330_client.hpp"

namespace surgical_hand_xc330 {

class Xc330StatePublisher : public rclcpp::Node {
 public:
  Xc330StatePublisher() : rclcpp::Node("xc330_state_publisher") {
    const auto port = declare_parameter<std::string>("port", "");
    const auto baud = static_cast<int>(declare_parameter<int64_t>("baud", 57600));
    id_ = static_cast<uint8_t>(declare_parameter<int64_t>("id", 1));
    joint_name_ = declare_parameter<std::string>("joint_name", "tendon_motor");
    const double rate_hz = declare_parameter<double>("rate_hz", 20.0);
    temperature_warn_c_ = static_cast<int>(declare_parameter<int64_t>("temperature_warn_c", 60));

    if (port.empty()) {
      throw std::invalid_argument(
          "the 'port' parameter is required (e.g. -p port:=/dev/ttyUSB0); "
          "no default device path by design");
    }

    client_ = std::make_unique<Xc330Client>(port, baud);
    const auto info = client_->ping(id_);
    if (!info) {
      throw std::runtime_error("no motor at id " + std::to_string(id_) + " on " + port +
                               " (baud " + std::to_string(baud) + "); try xc330_cli scan");
    }
    RCLCPP_INFO(get_logger(), "found motor id %u (model %u) on %s @ %d baud; read-only publisher",
                info->id, info->model_number, port.c_str(), baud);

    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("~/joint_state", 10);
    current_pub_ = create_publisher<std_msgs::msg::Float64>("~/motor_current", 10);
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz),
                               [this] { onTimer(); });
  }

 private:
  void onTimer() {
    MotorState state;
    try {
      state = client_->readState(id_);
    } catch (const std::runtime_error& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "read failed: %s", e.what());
      return;
    }

    sensor_msgs::msg::JointState joint;
    joint.header.stamp = now();
    joint.name = {joint_name_};
    joint.position = {state.positionRadians()};
    joint.velocity = {state.velocityRadPerSec()};
    joint_pub_->publish(joint);

    std_msgs::msg::Float64 current;
    current.data = state.currentAmps();
    current_pub_->publish(current);

    if (state.temperature_c >= temperature_warn_c_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "motor temperature %u C >= %d C warning threshold",
                           state.temperature_c, temperature_warn_c_);
    }
  }

  std::unique_ptr<Xc330Client> client_;
  uint8_t id_{1};
  std::string joint_name_;
  int temperature_warn_c_{60};
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr current_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace surgical_hand_xc330

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<surgical_hand_xc330::Xc330StatePublisher>());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
