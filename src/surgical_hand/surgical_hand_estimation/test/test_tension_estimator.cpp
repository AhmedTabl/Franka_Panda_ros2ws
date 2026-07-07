// Unit tests for TensionEstimator with hand-computed numerical cases.

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "surgical_hand_estimation/tension_estimator.hpp"

namespace she = surgical_hand_estimation;

namespace {

// A config with "clean" numbers so expected values are easy to compute by
// hand. Filtering off unless a test enables it.
she::TensionEstimatorConfig cleanConfig() {
  she::TensionEstimatorConfig config;
  config.kt_nm_per_a = 0.5;
  config.current_offset_a = 0.0;
  config.current_deadband_a = 0.0;
  config.spool_radius_m = 0.01;
  config.efficiency = 1.0;
  config.direction = 1;
  config.lowpass_cutoff_hz = 0.0;  // off
  config.saturation_current_a = 1.6;
  config.safety_threshold_n = 4.0;
  return config;
}

constexpr double kDt = 0.01;

}  // namespace

TEST(TensionEstimator, BasicConversion) {
  // tau = 0.5 * 0.2 = 0.1 N*m; tension = 1.0 * 0.1 / 0.01 = 10 N.
  she::TensionEstimator estimator(cleanConfig());
  const auto estimate = estimator.update(0.2, kDt);
  EXPECT_NEAR(estimate.motor_torque_nm, 0.1, 1e-12);
  EXPECT_NEAR(estimate.tension_n, 10.0, 1e-12);
  EXPECT_NEAR(estimate.raw_tension_n, 10.0, 1e-12);
  EXPECT_EQ(estimate.status, she::TensionStatus::kOk);
  EXPECT_TRUE(estimate.over_safety_threshold);  // 10 N > 4 N threshold
}

TEST(TensionEstimator, OffsetIsSubtracted) {
  auto config = cleanConfig();
  config.current_offset_a = 0.05;
  she::TensionEstimator estimator(config);
  // i_eff = 0.25 - 0.05 = 0.2 -> same 10 N as the basic case.
  EXPECT_NEAR(estimator.update(0.25, kDt).tension_n, 10.0, 1e-12);
}

TEST(TensionEstimator, DeadbandZeroesSmallCurrents) {
  auto config = cleanConfig();
  config.current_offset_a = 0.05;
  config.current_deadband_a = 0.02;
  she::TensionEstimator estimator(config);

  // Values chosen clearly inside the band: exact boundary values (e.g.
  // 0.03 with offset 0.05) fall on the |i_eff| == deadband edge where
  // floating-point rounding legitimately decides either way.
  for (const double current : {0.05, 0.065, 0.035, 0.069}) {
    const auto estimate = estimator.update(current, kDt);
    EXPECT_EQ(estimate.tension_n, 0.0) << "current " << current;
    EXPECT_EQ(estimate.motor_torque_nm, 0.0) << "current " << current;
    EXPECT_EQ(estimate.status, she::TensionStatus::kBelowDeadband) << "current " << current;
    EXPECT_LE(estimate.confidence, 0.11) << "current " << current;
  }
  // Just outside the deadband: nonzero again.
  EXPECT_GT(estimator.update(0.08, kDt).tension_n, 0.0);
}

TEST(TensionEstimator, DirectionFlipsPullSign) {
  auto config = cleanConfig();
  config.direction = -1;
  she::TensionEstimator estimator(config);

  // With direction=-1, NEGATIVE current pulls: tension positive.
  const auto pulling = estimator.update(-0.2, kDt);
  EXPECT_NEAR(pulling.tension_n, 10.0, 1e-12);
  EXPECT_EQ(pulling.status, she::TensionStatus::kOk);

  estimator.reset();
  // Positive current now unwinds -> slack, clamped to zero.
  const auto slack = estimator.update(0.2, kDt);
  EXPECT_EQ(slack.tension_n, 0.0);
  EXPECT_NEAR(slack.raw_tension_n, -10.0, 1e-12);
  EXPECT_EQ(slack.status, she::TensionStatus::kSlack);
}

TEST(TensionEstimator, EfficiencyScalesOutput) {
  auto config = cleanConfig();
  config.efficiency = 0.6;
  she::TensionEstimator estimator(config);
  EXPECT_NEAR(estimator.update(0.2, kDt).tension_n, 6.0, 1e-12);
}

TEST(TensionEstimator, LowPassStepResponse) {
  auto config = cleanConfig();
  config.lowpass_cutoff_hz = 10.0;
  she::TensionEstimator estimator(config);

  // First sample seeds the filter directly (no stale state).
  EXPECT_NEAR(estimator.update(0.2, kDt).tension_n, 10.0, 1e-12);

  // Step down to 0 A: second sample must move one alpha-step toward 0.
  // alpha = dt / (dt + 1/(2*pi*fc)) = 0.01 / (0.01 + 0.0159155) = 0.385869...
  const double alpha = kDt / (kDt + 1.0 / (2.0 * M_PI * 10.0));
  const double expected = 10.0 + alpha * (0.0 - 10.0);
  EXPECT_NEAR(estimator.update(0.0, kDt).tension_n, expected, 1e-9);

  // Repeated samples converge monotonically toward 0.
  double previous = expected;
  for (int i = 0; i < 50; ++i) {
    const double value = estimator.update(0.0, kDt).tension_n;
    EXPECT_LT(value, previous);
    previous = value;
  }
  EXPECT_LT(previous, 0.01);
}

TEST(TensionEstimator, ResetClearsFilterState) {
  auto config = cleanConfig();
  config.lowpass_cutoff_hz = 10.0;
  she::TensionEstimator estimator(config);
  estimator.update(0.2, kDt);  // filter at 10 N
  estimator.reset();
  // After reset the next sample seeds the filter directly again.
  EXPECT_NEAR(estimator.update(0.1, kDt).tension_n, 5.0, 1e-12);
}

TEST(TensionEstimator, CalibrationTableInterpolatesAndClamps) {
  auto config = cleanConfig();
  config.calibration_table = {{0.0, 0.0}, {1.0, 20.0}};
  she::TensionEstimator estimator(config);

  EXPECT_NEAR(estimator.update(0.5, kDt).tension_n, 10.0, 1e-12);   // midpoint
  EXPECT_NEAR(estimator.update(0.25, kDt).tension_n, 5.0, 1e-12);   // quarter
  EXPECT_NEAR(estimator.update(1.2, kDt).tension_n, 20.0, 1e-12);   // clamped high
  // Note 1.2 A also exceeds saturation_current_a (1.6? no: 1.2 < 1.6) -> OK.
  EXPECT_NEAR(estimator.update(-0.5, kDt).raw_tension_n, -10.0, 1e-12);  // sign restored
}

TEST(TensionEstimator, SaturationFlagsNearStall) {
  she::TensionEstimator estimator(cleanConfig());
  const auto estimate = estimator.update(1.7, kDt);
  EXPECT_EQ(estimate.status, she::TensionStatus::kSaturated);
  EXPECT_LE(estimate.confidence, 0.3);
  // Safety flag still evaluated: 1.7 A -> 85 N >> 4 N.
  EXPECT_TRUE(estimate.over_safety_threshold);
}

TEST(TensionEstimator, SafetyThresholdBoundary) {
  she::TensionEstimator estimator(cleanConfig());
  // 0.079 A -> 3.95 N < 4 N; 0.081 A -> 4.05 N > 4 N.
  EXPECT_FALSE(estimator.update(0.079, kDt).over_safety_threshold);
  estimator.reset();
  EXPECT_TRUE(estimator.update(0.081, kDt).over_safety_threshold);
}

TEST(TensionEstimator, ConfidenceRampsAboveDeadband) {
  auto config = cleanConfig();
  config.current_deadband_a = 0.02;
  she::TensionEstimator estimator(config);

  const auto near_edge = estimator.update(0.021, kDt);  // just past deadband
  const auto far = estimator.update(0.2, kDt);          // deep in trusted region
  EXPECT_EQ(near_edge.status, she::TensionStatus::kOk);
  EXPECT_LT(near_edge.confidence, 0.2);
  EXPECT_NEAR(far.confidence, 0.9, 1e-12);
  EXPECT_LT(far.confidence, 1.0);  // observer: never fully confident
}

TEST(TensionEstimator, InvalidConfigsThrow) {
  {
    auto config = cleanConfig();
    config.kt_nm_per_a = 0.0;
    EXPECT_THROW(she::TensionEstimator{config}, std::invalid_argument);
  }
  {
    auto config = cleanConfig();
    config.spool_radius_m = -0.01;
    EXPECT_THROW(she::TensionEstimator{config}, std::invalid_argument);
  }
  {
    auto config = cleanConfig();
    config.efficiency = 1.5;
    EXPECT_THROW(she::TensionEstimator{config}, std::invalid_argument);
  }
  {
    auto config = cleanConfig();
    config.direction = 0;
    EXPECT_THROW(she::TensionEstimator{config}, std::invalid_argument);
  }
  {
    auto config = cleanConfig();
    config.calibration_table = {{0.5, 5.0}, {0.5, 10.0}};  // not strictly increasing
    EXPECT_THROW(she::TensionEstimator{config}, std::invalid_argument);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
