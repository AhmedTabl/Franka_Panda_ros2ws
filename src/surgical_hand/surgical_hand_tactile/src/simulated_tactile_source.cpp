#include "surgical_hand_tactile/tactile_source.hpp"

#include <cmath>
#include <stdexcept>

namespace surgical_hand_tactile {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

SimulatedTactileSource::SimulatedTactileSource(const SimulatedTactileConfig& config)
    : config_(config), rng_(config.seed), noise_(0.0, 1.0) {
  if (config_.contact_period_s <= 0.0 || config_.contact_duration_s <= 0.0 ||
      config_.contact_duration_s > config_.contact_period_s) {
    throw std::invalid_argument(
        "need 0 < contact_duration_s <= contact_period_s in SimulatedTactileConfig");
  }
  if (config_.magnetometer_count <= 0) {
    throw std::invalid_argument("magnetometer_count must be > 0");
  }
  if (config_.noise_std_ut < 0.0) {
    throw std::invalid_argument("noise_std_ut must be >= 0");
  }
}

TactileSample SimulatedTactileSource::sample(double t_s) {
  TactileSample out;
  out.simulated = true;
  out.processed_valid = true;

  const double phase_s = std::fmod(std::max(t_s, 0.0), config_.contact_period_s);
  out.contact = phase_s < config_.contact_duration_s;

  if (out.contact) {
    // Half-sine force bump over the episode; peaks at the midpoint.
    const double u = phase_s / config_.contact_duration_s;  // 0..1
    out.normal_force_n = config_.peak_normal_force_n * std::sin(kPi * u);
    // Small shear wobble, phase-shifted between x and y.
    out.shear_force_x_n = config_.peak_shear_force_n * std::sin(2.0 * kPi * u);
    out.shear_force_y_n = config_.peak_shear_force_n * 0.5 * std::cos(2.0 * kPi * u);
    // Slip becomes likely toward release (last 30% of the episode).
    out.slip_probability = u < 0.7 ? 0.05 : 0.05 + 0.9 * (u - 0.7) / 0.3;
    // Contact centroid drifts slightly across the pad during the episode.
    out.contact_location_m = {0.002 * std::cos(2.0 * kPi * u),
                              0.002 * std::sin(2.0 * kPi * u), 0.0};
  }

  // Synthetic raw field: per-axis baseline offset shifted by the contact
  // force, plus optional Gaussian noise. The exact model is arbitrary —
  // its only job is exercising downstream consumers of the raw field.
  out.raw_magnetometer_ut.reserve(static_cast<size_t>(config_.magnetometer_count) * 3);
  for (int m = 0; m < config_.magnetometer_count; ++m) {
    for (int axis = 0; axis < 3; ++axis) {
      const double baseline = config_.field_baseline_ut * (1.0 + 0.1 * m - 0.05 * axis);
      const double deflection =
          config_.field_gain_ut_per_n *
          (axis == 2 ? out.normal_force_n : (axis == 0 ? out.shear_force_x_n : out.shear_force_y_n)) /
          (1.0 + m);  // magnets farther from the contact see less shift
      const double noise = config_.noise_std_ut > 0.0 ? config_.noise_std_ut * noise_(rng_) : 0.0;
      out.raw_magnetometer_ut.push_back(baseline + deflection + noise);
    }
  }
  return out;
}

}  // namespace surgical_hand_tactile
