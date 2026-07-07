#include "surgical_hand_estimation/tension_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace surgical_hand_estimation {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kMinDt = 1e-4;
constexpr double kMaxDt = 0.1;
// Confidence heuristic values. Never 1.0: this is an observer.
constexpr double kConfidenceFloor = 0.1;
constexpr double kConfidenceSlack = 0.2;
constexpr double kConfidenceSaturated = 0.3;
constexpr double kConfidenceCeiling = 0.9;
}  // namespace

TensionEstimator::TensionEstimator(const TensionEstimatorConfig& config) : config_(config) {
  if (config_.kt_nm_per_a <= 0.0) {
    throw std::invalid_argument("kt_nm_per_a must be > 0");
  }
  if (config_.spool_radius_m <= 0.0) {
    throw std::invalid_argument("spool_radius_m must be > 0");
  }
  if (config_.efficiency <= 0.0 || config_.efficiency > 1.0) {
    throw std::invalid_argument("efficiency must be in (0, 1]");
  }
  if (config_.direction != 1 && config_.direction != -1) {
    throw std::invalid_argument("direction must be +1 or -1");
  }
  if (config_.current_deadband_a < 0.0) {
    throw std::invalid_argument("current_deadband_a must be >= 0");
  }
  if (config_.saturation_current_a <= config_.current_deadband_a) {
    throw std::invalid_argument("saturation_current_a must be > current_deadband_a");
  }
  for (size_t i = 1; i < config_.calibration_table.size(); ++i) {
    if (config_.calibration_table[i].first <= config_.calibration_table[i - 1].first) {
      throw std::invalid_argument("calibration_table currents must be strictly increasing");
    }
  }
}

double TensionEstimator::applyCalibration(double i_eff) const {
  const auto& table = config_.calibration_table;
  const double magnitude = std::abs(i_eff);
  const double sign = (i_eff < 0.0) ? -1.0 : 1.0;

  double tension_magnitude = 0.0;
  if (magnitude <= table.front().first) {
    tension_magnitude = table.front().second;
  } else if (magnitude >= table.back().first) {
    tension_magnitude = table.back().second;
  } else {
    for (size_t i = 1; i < table.size(); ++i) {
      if (magnitude <= table[i].first) {
        const double x0 = table[i - 1].first;
        const double y0 = table[i - 1].second;
        const double x1 = table[i].first;
        const double y1 = table[i].second;
        tension_magnitude = y0 + (y1 - y0) * (magnitude - x0) / (x1 - x0);
        break;
      }
    }
  }
  return config_.direction * sign * tension_magnitude;
}

TensionEstimate TensionEstimator::update(double current_a, double dt_s) {
  TensionEstimate estimate;
  estimate.motor_current_a = current_a;

  // 1. Offset and deadband.
  double i_eff = current_a - config_.current_offset_a;
  const bool in_deadband = std::abs(i_eff) <= config_.current_deadband_a;
  if (in_deadband) {
    i_eff = 0.0;
  }
  const bool saturated = std::abs(i_eff) >= config_.saturation_current_a;

  // 2. Torque and raw tension (analytic model or calibration table).
  estimate.motor_torque_nm = config_.kt_nm_per_a * i_eff;
  if (!config_.calibration_table.empty()) {
    estimate.raw_tension_n = in_deadband ? 0.0 : applyCalibration(i_eff);
  } else {
    estimate.raw_tension_n =
        config_.direction * config_.efficiency * estimate.motor_torque_nm / config_.spool_radius_m;
  }

  // 3. A tendon can only pull: negative model output means the motor is
  //    unwinding or the cable is slack.
  const bool slack = estimate.raw_tension_n < 0.0;
  const double clamped_tension = std::max(estimate.raw_tension_n, 0.0);

  // 4. Optional first-order low-pass on the clamped tension.
  const double dt = std::clamp(dt_s, kMinDt, kMaxDt);
  if (config_.lowpass_cutoff_hz > 0.0 && has_filter_state_) {
    const double time_constant = 1.0 / (2.0 * kPi * config_.lowpass_cutoff_hz);
    const double alpha = dt / (dt + time_constant);
    filtered_tension_n_ += alpha * (clamped_tension - filtered_tension_n_);
  } else {
    filtered_tension_n_ = clamped_tension;  // filter disabled or first sample
  }
  has_filter_state_ = true;
  estimate.tension_n = filtered_tension_n_;

  // 5. Status (estimation quality) and confidence heuristic.
  if (saturated) {
    estimate.status = TensionStatus::kSaturated;
    estimate.confidence = kConfidenceSaturated;
  } else if (in_deadband) {
    estimate.status = TensionStatus::kBelowDeadband;
    estimate.confidence = kConfidenceFloor;
  } else if (slack) {
    estimate.status = TensionStatus::kSlack;
    estimate.confidence = kConfidenceSlack;
  } else {
    estimate.status = TensionStatus::kOk;
    // Ramp confidence from the floor at the deadband edge up to the
    // ceiling at 2x deadband; constant ceiling beyond (deadband may be 0).
    const double band = std::max(config_.current_deadband_a, 1e-9);
    const double excess = std::abs(i_eff) - band;
    const double ramp = std::clamp(excess / band, 0.0, 1.0);
    estimate.confidence = kConfidenceFloor + ramp * (kConfidenceCeiling - kConfidenceFloor);
  }

  // 6. Safety flag, always evaluated on the filtered estimate regardless
  //    of status: a saturated-but-dangerous reading must still trip.
  estimate.over_safety_threshold =
      config_.safety_threshold_n > 0.0 && estimate.tension_n > config_.safety_threshold_n;

  return estimate;
}

void TensionEstimator::reset() {
  filtered_tension_n_ = 0.0;
  has_filter_state_ = false;
}

}  // namespace surgical_hand_estimation
