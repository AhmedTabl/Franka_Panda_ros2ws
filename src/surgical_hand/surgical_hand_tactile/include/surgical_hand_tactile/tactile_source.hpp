// Tactile source interface for the surgical hand fingertips.
//
// WHAT IS STUBBED: only SimulatedTactileSource exists today. When real
// eFlesh hardware (arXiv:2506.09994) arrives, add an EfleshTactileSource
// implementing the same interface: read raw magnetometer values from the
// sensor board, run the calibration/learned model, fill the same
// TactileSample. Everything above this interface (node, topics, message)
// stays unchanged.
//
// ROS-free by design so sources are unit-testable and reusable inside a
// future ros2_control sensor component.

#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace surgical_hand_tactile {

struct TactileSample {
  bool simulated{true};
  std::vector<double> raw_magnetometer_ut{};  // xyz per magnetometer
  bool processed_valid{false};
  bool contact{false};
  double normal_force_n{0.0};
  double shear_force_x_n{0.0};
  double shear_force_y_n{0.0};
  double slip_probability{0.0};
  std::array<double, 3> contact_location_m{{0.0, 0.0, 0.0}};
};

class TactileSource {
 public:
  virtual ~TactileSource() = default;
  // Produce the tactile state at time t_s (seconds, monotonically
  // increasing from the source's start).
  virtual TactileSample sample(double t_s) = 0;
};

struct SimulatedTactileConfig {
  // Periodic synthetic contact episodes: contact for contact_duration_s at
  // the start of every contact_period_s window.
  double contact_period_s{4.0};
  double contact_duration_s{1.0};
  double peak_normal_force_n{2.0};   // half-sine peak during an episode
  double peak_shear_force_n{0.4};
  // Raw field synthesis: baseline + deflection proportional to force + noise.
  int magnetometer_count{5};         // eFlesh reference design
  double field_baseline_ut{50.0};
  double field_gain_ut_per_n{30.0};
  double noise_std_ut{0.5};          // 0 disables noise (deterministic)
  uint32_t seed{42};
};

// Deterministic-when-noiseless synthetic tactile generator: smooth
// half-sine contact episodes, small shear wobble, slip probability ramping
// up toward episode release, and magnetometer values that respond to the
// synthetic force. Throws std::invalid_argument on nonsensical config.
class SimulatedTactileSource : public TactileSource {
 public:
  explicit SimulatedTactileSource(const SimulatedTactileConfig& config);
  TactileSample sample(double t_s) override;

 private:
  SimulatedTactileConfig config_;
  std::mt19937 rng_;
  std::normal_distribution<double> noise_;
};

}  // namespace surgical_hand_tactile
