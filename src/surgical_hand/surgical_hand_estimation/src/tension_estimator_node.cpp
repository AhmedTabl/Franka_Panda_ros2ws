// ROS 2 wrapper around TensionEstimator (one node = one tendon/motor).
//
// Subscribes:  ~/motor_current  (std_msgs/Float64, amps)
// Publishes:   ~/tension        (surgical_hand_msgs/TendonTension)
//
// For the current one-motor phase, remap ~/motor_current onto whatever
// topic the XC330 reader publishes. Multi-tendon hands run one instance
// per tendon (composition/launch handles the fan-out); an array-based
// variant can be added when the 12-DOF hand exists.
//
// All model parameters are declared as ROS parameters so they can be set
// from YAML (see config/tension_estimator.yaml) or tuned at runtime with
// `ros2 param set` (changes require restart; live retune is not worth the
// complexity until the estimator is characterized on real hardware).

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <surgical_hand_msgs/msg/tendon_tension.hpp>

#include "surgical_hand_estimation/tension_estimator.hpp"

namespace surgical_hand_estimation {

class TensionEstimatorNode : public rclcpp::Node {
 public:
  TensionEstimatorNode() : rclcpp::Node("tension_estimator") {
    TensionEstimatorConfig config;
    config.kt_nm_per_a = declare_parameter("kt_nm_per_a", config.kt_nm_per_a);
    config.current_offset_a = declare_parameter("current_offset_a", config.current_offset_a);
    config.current_deadband_a = declare_parameter("current_deadband_a", config.current_deadband_a);
    config.spool_radius_m = declare_parameter("spool_radius_m", config.spool_radius_m);
    config.efficiency = declare_parameter("efficiency", config.efficiency);
    config.direction = static_cast<int>(declare_parameter("direction", config.direction));
    config.lowpass_cutoff_hz = declare_parameter("lowpass_cutoff_hz", config.lowpass_cutoff_hz);
    config.saturation_current_a =
        declare_parameter("saturation_current_a", config.saturation_current_a);
    config.safety_threshold_n = declare_parameter("safety_threshold_n", config.safety_threshold_n);

    // Calibration table as two parallel arrays (ROS params cannot hold
    // pair lists). Empty = analytic model.
    const auto cal_currents =
        declare_parameter("calibration_currents_a", std::vector<double>{});
    const auto cal_tensions =
        declare_parameter("calibration_tensions_n", std::vector<double>{});
    if (cal_currents.size() != cal_tensions.size()) {
      throw std::invalid_argument(
          "calibration_currents_a and calibration_tensions_n must have the same length");
    }
    for (size_t i = 0; i < cal_currents.size(); ++i) {
      config.calibration_table.emplace_back(cal_currents[i], cal_tensions[i]);
    }

    estimator_ = std::make_unique<TensionEstimator>(config);  // throws on bad config

    publisher_ = create_publisher<surgical_hand_msgs::msg::TendonTension>("~/tension", 10);
    subscription_ = create_subscription<std_msgs::msg::Float64>(
        "~/motor_current", 10,
        [this](const std_msgs::msg::Float64& msg) { onCurrent(msg); });

    RCLCPP_INFO(get_logger(),
                "tension estimator ready (%s model, kt=%.3f N.m/A, spool=%.4f m, "
                "efficiency=%.2f, safety threshold=%.2f N). This is an observer, "
                "not a tension sensor.",
                config.calibration_table.empty() ? "analytic" : "calibrated",
                config.kt_nm_per_a, config.spool_radius_m, config.efficiency,
                config.safety_threshold_n);
  }

 private:
  void onCurrent(const std_msgs::msg::Float64& msg) {
    // The input carries no timestamp, so dt comes from the steady clock;
    // the estimator clamps it to a sane range internally.
    const auto now = std::chrono::steady_clock::now();
    double dt_s = 0.01;
    if (last_sample_time_) {
      dt_s = std::chrono::duration<double>(now - *last_sample_time_).count();
    }
    last_sample_time_ = now;

    const TensionEstimate estimate = estimator_->update(msg.data, dt_s);

    surgical_hand_msgs::msg::TendonTension out;
    out.header.stamp = get_clock()->now();
    out.status = static_cast<uint8_t>(estimate.status);
    out.tension_n = estimate.tension_n;
    out.raw_tension_n = estimate.raw_tension_n;
    out.motor_torque_nm = estimate.motor_torque_nm;
    out.motor_current_a = estimate.motor_current_a;
    out.confidence = estimate.confidence;
    out.over_safety_threshold = estimate.over_safety_threshold;
    publisher_->publish(out);

    if (estimate.over_safety_threshold) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "estimated tendon tension %.2f N exceeds safety threshold %.2f N",
                           estimate.tension_n, estimator_->config().safety_threshold_n);
    }
  }

  std::unique_ptr<TensionEstimator> estimator_;
  rclcpp::Publisher<surgical_hand_msgs::msg::TendonTension>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr subscription_;
  std::optional<std::chrono::steady_clock::time_point> last_sample_time_;
};

}  // namespace surgical_hand_estimation

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<surgical_hand_estimation::TensionEstimatorNode>());
  rclcpp::shutdown();
  return 0;
}
