// Tactile publisher node.
//
// Publishes surgical_hand_msgs/TactileState per fingertip sensor on
// ~/<sensor_name>. Today every sensor is backed by SimulatedTactileSource
// (simulated=true in every message); when real eFlesh hardware arrives,
// an EfleshTactileSource replaces the simulated one per sensor without
// changing topics, message layout, or consumers.
//
//   ros2 run surgical_hand_tactile tactile_publisher
//   ros2 topic echo /tactile_publisher/index

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <surgical_hand_msgs/msg/tactile_state.hpp>

#include "surgical_hand_tactile/tactile_source.hpp"

namespace surgical_hand_tactile {

class TactilePublisherNode : public rclcpp::Node {
 public:
  TactilePublisherNode() : rclcpp::Node("tactile_publisher") {
    const auto sensor_names = declare_parameter<std::vector<std::string>>(
        "sensor_names", {"thumb", "index", "middle"});
    const double rate_hz = declare_parameter<double>("rate_hz", 50.0);

    SimulatedTactileConfig config;
    config.contact_period_s = declare_parameter<double>("contact_period_s", config.contact_period_s);
    config.contact_duration_s =
        declare_parameter<double>("contact_duration_s", config.contact_duration_s);
    config.peak_normal_force_n =
        declare_parameter<double>("peak_normal_force_n", config.peak_normal_force_n);
    config.peak_shear_force_n =
        declare_parameter<double>("peak_shear_force_n", config.peak_shear_force_n);
    config.noise_std_ut = declare_parameter<double>("noise_std_ut", config.noise_std_ut);

    for (size_t i = 0; i < sensor_names.size(); ++i) {
      Sensor sensor;
      sensor.name = sensor_names[i];
      // Stagger the seeds and phases so fingers don't touch in lockstep.
      SimulatedTactileConfig sensor_config = config;
      sensor_config.seed = config.seed + static_cast<uint32_t>(i);
      sensor.phase_offset_s = config.contact_period_s * static_cast<double>(i) /
                              static_cast<double>(sensor_names.size());
      sensor.source = std::make_unique<SimulatedTactileSource>(sensor_config);
      sensor.publisher =
          create_publisher<surgical_hand_msgs::msg::TactileState>("~/" + sensor.name, 10);
      sensors_.push_back(std::move(sensor));
    }

    start_ = now();
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz),
                               [this] { onTimer(); });
    RCLCPP_INFO(get_logger(),
                "publishing SIMULATED tactile data for %zu sensors at %.0f Hz "
                "(real eFlesh integration pending)",
                sensors_.size(), rate_hz);
  }

 private:
  struct Sensor {
    std::string name;
    double phase_offset_s{0.0};
    std::unique_ptr<TactileSource> source;
    rclcpp::Publisher<surgical_hand_msgs::msg::TactileState>::SharedPtr publisher;
  };

  void onTimer() {
    const double t = (now() - start_).seconds();
    for (auto& sensor : sensors_) {
      const TactileSample sample = sensor.source->sample(t + sensor.phase_offset_s);
      surgical_hand_msgs::msg::TactileState msg;
      msg.header.stamp = now();
      msg.sensor_name = sensor.name;
      msg.simulated = sample.simulated;
      msg.raw_magnetometer_ut = sample.raw_magnetometer_ut;
      msg.processed_valid = sample.processed_valid;
      msg.contact = sample.contact;
      msg.normal_force_n = sample.normal_force_n;
      msg.shear_force_x_n = sample.shear_force_x_n;
      msg.shear_force_y_n = sample.shear_force_y_n;
      msg.slip_probability = sample.slip_probability;
      msg.contact_location_m = {sample.contact_location_m[0], sample.contact_location_m[1],
                                sample.contact_location_m[2]};
      sensor.publisher->publish(msg);
    }
  }

  std::vector<Sensor> sensors_;
  rclcpp::Time start_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace surgical_hand_tactile

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<surgical_hand_tactile::TactilePublisherNode>());
  rclcpp::shutdown();
  return 0;
}
