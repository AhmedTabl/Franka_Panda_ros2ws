// Unit tests for the simulated tactile source.

#include <gtest/gtest.h>

#include "surgical_hand_tactile/tactile_source.hpp"

namespace sht = surgical_hand_tactile;

namespace {
sht::SimulatedTactileConfig noiseless() {
  sht::SimulatedTactileConfig config;
  config.noise_std_ut = 0.0;  // deterministic
  return config;
}
}  // namespace

TEST(SimulatedTactile, ContactWindowing) {
  sht::SimulatedTactileSource source(noiseless());
  // Default: 1 s contact at the start of every 4 s period.
  EXPECT_TRUE(source.sample(0.5).contact);
  EXPECT_FALSE(source.sample(2.0).contact);
  EXPECT_TRUE(source.sample(4.5).contact);
  EXPECT_FALSE(source.sample(7.9).contact);
}

TEST(SimulatedTactile, ForcesZeroOutsideContact) {
  sht::SimulatedTactileSource source(noiseless());
  const auto sample = source.sample(2.0);
  EXPECT_EQ(sample.normal_force_n, 0.0);
  EXPECT_EQ(sample.shear_force_x_n, 0.0);
  EXPECT_EQ(sample.shear_force_y_n, 0.0);
  EXPECT_EQ(sample.slip_probability, 0.0);
}

TEST(SimulatedTactile, PeakForceAtEpisodeMidpoint) {
  auto config = noiseless();
  config.peak_normal_force_n = 2.0;
  sht::SimulatedTactileSource source(config);
  EXPECT_NEAR(source.sample(0.5).normal_force_n, 2.0, 1e-9);  // mid of 1 s episode
  EXPECT_LT(source.sample(0.1).normal_force_n, 2.0);
  EXPECT_GT(source.sample(0.1).normal_force_n, 0.0);
}

TEST(SimulatedTactile, SlipProbabilityBoundedAndRampsAtRelease) {
  sht::SimulatedTactileSource source(noiseless());
  for (double t = 0.0; t < 8.0; t += 0.01) {
    const double p = source.sample(t).slip_probability;
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
  }
  EXPECT_GT(source.sample(0.95).slip_probability, source.sample(0.5).slip_probability);
}

TEST(SimulatedTactile, RawFieldShapeAndForceResponse) {
  sht::SimulatedTactileSource source(noiseless());
  const auto idle = source.sample(2.0);       // no contact
  const auto touching = source.sample(0.5);   // peak contact
  ASSERT_EQ(idle.raw_magnetometer_ut.size(), 15u);  // 5 magnetometers x 3 axes
  // z-axis of the closest magnetometer shifts with normal force.
  EXPECT_GT(touching.raw_magnetometer_ut[2], idle.raw_magnetometer_ut[2]);
}

TEST(SimulatedTactile, DeterministicWithoutNoise) {
  sht::SimulatedTactileSource a(noiseless());
  sht::SimulatedTactileSource b(noiseless());
  for (double t : {0.1, 0.5, 2.0, 4.2}) {
    EXPECT_EQ(a.sample(t).raw_magnetometer_ut, b.sample(t).raw_magnetometer_ut) << "t=" << t;
  }
}

TEST(SimulatedTactile, MarkedSimulatedAndValid) {
  sht::SimulatedTactileSource source(noiseless());
  const auto sample = source.sample(0.0);
  EXPECT_TRUE(sample.simulated);
  EXPECT_TRUE(sample.processed_valid);
}

TEST(SimulatedTactile, InvalidConfigThrows) {
  auto config = noiseless();
  config.contact_duration_s = 5.0;  // > period (4.0)
  EXPECT_THROW(sht::SimulatedTactileSource{config}, std::invalid_argument);
  config = noiseless();
  config.magnetometer_count = 0;
  EXPECT_THROW(sht::SimulatedTactileSource{config}, std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
