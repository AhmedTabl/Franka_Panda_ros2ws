// Current-based tendon tension estimator (observer).
//
// Estimates the tension in a single tendon/cable from the driving motor's
// current draw. Base model (see the workspace surgical hand docs and
// Sang et al. 2017, doi:10.1002/rcs.1824, for the current-based force
// estimation approach this simplifies):
//
//   i_eff    = current - current_offset          (no-load/idle compensation)
//   i_eff    = 0 if |i_eff| <= current_deadband  (friction/noise floor)
//   tau      = kt * i_eff                        (torque constant)
//   tension  = direction * efficiency * tau / spool_radius
//   output   = first-order low-pass of tension (optional)
//
// If a calibration table is provided (empirically measured
// (effective current -> tension) pairs), it replaces the analytic
// kt/efficiency/spool step with piecewise-linear interpolation; offset,
// deadband, direction, filtering, and status logic still apply.
//
// THIS IS NOT A SENSOR. Gearbox friction (the XC330-M288 has a 288.4:1
// gear ratio with high internal friction), spool/capstan friction, cable
// routing losses, slack, hysteresis, acceleration transients, and motor
// heating all bias the estimate. The `status` and `confidence` outputs
// exist so downstream code never has to guess how trustworthy a sample is.
//
// The class is deliberately free of ROS dependencies so it can be unit
// tested in isolation and reused inside the future real-hardware
// ros2_control plugin.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace surgical_hand_estimation {

struct TensionEstimatorConfig {
  // Motor torque constant [N*m/A], torque at the OUTPUT shaft per amp of
  // input current. Default derived from the XC330-M288-T datasheet stall
  // point (0.93 N*m at 1.8 A, 6 V) — a rough starting value that must be
  // replaced by bench characterization in the one-motor hardware slice.
  double kt_nm_per_a{0.52};

  // Current the motor draws when the tendon is doing no useful work [A].
  double current_offset_a{0.04};

  // Effective currents with |i_eff| at or below this are treated as zero
  // torque [A] (friction/measurement noise floor).
  double current_deadband_a{0.02};

  // Radius of the spool/capstan the tendon winds on [m].
  double spool_radius_m{0.005};

  // Fraction of motor torque that survives gearbox + spool + routing
  // losses, in (0, 1]. Placeholder until characterized.
  double efficiency{0.6};

  // +1 or -1: makes "positive current" mean "pulling the tendon".
  int direction{1};

  // First-order low-pass cutoff [Hz]; <= 0 disables filtering.
  double lowpass_cutoff_hz{10.0};

  // |i_eff| at or above this is the saturation/stall region where the
  // linear model breaks down [A]. XC330-M288-T stall is 1.8 A.
  double saturation_current_a{1.6};

  // Estimates above this trip the over_safety_threshold flag [N].
  // 4 N default: suture-strand design capacity from the hand spec doc.
  double safety_threshold_n{4.0};

  // Optional empirical calibration: (effective_current_a, tension_n)
  // pairs, strictly increasing in current, interpolated piecewise-linearly
  // and clamped at both ends. Applied to |i_eff|; sign is restored from
  // direction * sign(i_eff). Empty = use the analytic model.
  std::vector<std::pair<double, double>> calibration_table{};
};

enum class TensionStatus {
  kOk = 0,
  kBelowDeadband = 1,  // inside the deadband; tension indistinguishable from friction
  kSlack = 2,          // model produced negative tension (unwinding/slack)
  kSaturated = 3,      // near stall; linear model unreliable
};

struct TensionEstimate {
  double tension_n{0.0};        // filtered, clamped >= 0
  double raw_tension_n{0.0};    // unfiltered, signed model output
  double motor_torque_nm{0.0};  // after offset/deadband
  double motor_current_a{0.0};  // input sample
  TensionStatus status{TensionStatus::kBelowDeadband};
  double confidence{0.0};             // heuristic 0..1, never 1.0
  bool over_safety_threshold{false};  // checked on the filtered estimate
};

class TensionEstimator {
 public:
  // Throws std::invalid_argument on an invalid config (non-positive kt or
  // spool radius, efficiency outside (0, 1], direction not +/-1, negative
  // deadband, or a calibration table not strictly increasing in current).
  explicit TensionEstimator(const TensionEstimatorConfig& config);

  // Process one current sample [A] taken dt_s seconds after the previous
  // one. dt_s is clamped internally to a sane range; it only affects the
  // low-pass filter.
  TensionEstimate update(double current_a, double dt_s);

  // Forget filter state (e.g. after re-tensioning or a communication gap).
  void reset();

  const TensionEstimatorConfig& config() const { return config_; }

 private:
  double applyCalibration(double i_eff) const;

  TensionEstimatorConfig config_;
  double filtered_tension_n_{0.0};
  bool has_filter_state_{false};
};

}  // namespace surgical_hand_estimation
