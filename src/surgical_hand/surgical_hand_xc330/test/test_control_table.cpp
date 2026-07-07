// Unit tests for XC330 control-table constants and unit conversions.

#include <gtest/gtest.h>

#include "surgical_hand_xc330/xc330_control_table.hpp"

namespace shx = surgical_hand_xc330;

TEST(ControlTable, StateBlockCoversCurrentThroughTemperature) {
  EXPECT_EQ(shx::kStateBlockAddr, 126);
  EXPECT_EQ(shx::kStateBlockLength, 21);  // 126..146 inclusive
  // Field offsets inside the block used by readState():
  EXPECT_EQ(shx::kAddrPresentVelocity - shx::kStateBlockAddr, 2);
  EXPECT_EQ(shx::kAddrPresentPosition - shx::kStateBlockAddr, 6);
  EXPECT_EQ(shx::kAddrPresentInputVoltage - shx::kStateBlockAddr, 18);
  EXPECT_EQ(shx::kAddrPresentTemperature - shx::kStateBlockAddr, 20);
}

TEST(Conversions, PositionTicksToRadians) {
  EXPECT_DOUBLE_EQ(shx::ticksToRadians(2048), 0.0);              // center
  EXPECT_NEAR(shx::ticksToRadians(4096), shx::kPi, 1e-12);       // +half rev
  EXPECT_NEAR(shx::ticksToRadians(0), -shx::kPi, 1e-12);         // -half rev
  EXPECT_NEAR(shx::ticksToRadians(3072), shx::kPi / 2.0, 1e-12); // +90 deg
}

TEST(Conversions, RadiansToTicksRoundTrip) {
  EXPECT_EQ(shx::radiansToTicks(0.0), 2048);
  EXPECT_EQ(shx::radiansToTicks(shx::kPi / 2.0), 3072);
  EXPECT_EQ(shx::radiansToTicks(-shx::kPi / 2.0), 1024);
  // Round trip within one tick (0.088 deg) resolution.
  for (const double rad : {-2.0, -0.5, 0.0, 0.3, 1.7}) {
    EXPECT_NEAR(shx::ticksToRadians(shx::radiansToTicks(rad)), rad,
                2.0 * shx::kPi / shx::kTicksPerRevolution);
  }
}

TEST(Conversions, VelocityRawToRadPerSec) {
  // 100 raw * 0.229 rpm = 22.9 rpm = 22.9 * 2pi / 60 rad/s.
  EXPECT_NEAR(shx::velocityRawToRadPerSec(100), 22.9 * 2.0 * shx::kPi / 60.0, 1e-12);
  EXPECT_DOUBLE_EQ(shx::velocityRawToRadPerSec(0), 0.0);
  EXPECT_LT(shx::velocityRawToRadPerSec(-100), 0.0);
}

TEST(Conversions, CurrentAndVoltage) {
  EXPECT_DOUBLE_EQ(shx::currentRawToAmps(1000), 1.0);   // 1 mA / LSB
  EXPECT_DOUBLE_EQ(shx::currentRawToAmps(-50), -0.05);  // signed
  EXPECT_DOUBLE_EQ(shx::voltageRawToVolts(50), 5.0);    // 0.1 V / LSB
}

TEST(ControlTable, OperatingModeNames) {
  EXPECT_STREQ(shx::operatingModeName(5), "current-position");
  EXPECT_STREQ(shx::operatingModeName(3), "position");
  EXPECT_STREQ(shx::operatingModeName(99), "unknown");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
