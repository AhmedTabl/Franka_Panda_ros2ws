// DYNAMIXEL XC330-M288-T control table subset and unit conversions.
// Source: ROBOTIS e-manual (https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/)
// and the hand design doc's XC330 notes.

#pragma once

#include <cmath>
#include <cstdint>

namespace surgical_hand_xc330 {

// ---- addresses (name, size in bytes) ---------------------------------------
// EEPROM (writes require torque off and are gated behind --write-eeprom)
constexpr uint16_t kAddrModelNumber = 0;      // 2, XC330-M288-T = 1240
constexpr uint16_t kAddrId = 7;               // 1
constexpr uint16_t kAddrBaudRate = 8;         // 1
constexpr uint16_t kAddrOperatingMode = 11;   // 1
constexpr uint16_t kAddrCurrentLimit = 38;    // 2, unit 1 mA
// RAM
constexpr uint16_t kAddrTorqueEnable = 64;         // 1
constexpr uint16_t kAddrHardwareErrorStatus = 70;  // 1
constexpr uint16_t kAddrGoalCurrent = 102;         // 2, unit 1 mA
constexpr uint16_t kAddrGoalPosition = 116;        // 4, unit 1 tick
constexpr uint16_t kAddrPresentCurrent = 126;      // 2, unit 1 mA (signed)
constexpr uint16_t kAddrPresentVelocity = 128;     // 4, unit 0.229 rpm (signed)
constexpr uint16_t kAddrPresentPosition = 132;     // 4, unit 1 tick (signed)
constexpr uint16_t kAddrPresentInputVoltage = 144; // 2, unit 0.1 V
constexpr uint16_t kAddrPresentTemperature = 146;  // 1, unit 1 C

// One contiguous block from Present Current through Present Temperature so
// the whole fast state can be fetched in a single read transaction.
constexpr uint16_t kStateBlockAddr = kAddrPresentCurrent;                          // 126
constexpr uint16_t kStateBlockLength = kAddrPresentTemperature + 1 - kStateBlockAddr;  // 21

constexpr uint16_t kModelNumberXc330M288 = 1240;

// ---- operating modes (address 11) -------------------------------------------
enum class OperatingMode : uint8_t {
  kCurrent = 0,
  kVelocity = 1,
  kPosition = 3,
  kExtendedPosition = 4,
  kCurrentBasedPosition = 5,  // the mode this project uses on hardware
  kPwm = 16,
};

inline const char* operatingModeName(uint8_t mode) {
  switch (static_cast<OperatingMode>(mode)) {
    case OperatingMode::kCurrent: return "current";
    case OperatingMode::kVelocity: return "velocity";
    case OperatingMode::kPosition: return "position";
    case OperatingMode::kExtendedPosition: return "extended-position";
    case OperatingMode::kCurrentBasedPosition: return "current-position";
    case OperatingMode::kPwm: return "pwm";
  }
  return "unknown";
}

// ---- unit conversions ---------------------------------------------------------
constexpr double kTicksPerRevolution = 4096.0;
constexpr int32_t kCenterTicks = 2048;             // horn zero
constexpr double kVelocityUnitRpm = 0.229;          // per LSB
constexpr double kCurrentUnitAmps = 0.001;          // 1 mA per LSB
constexpr double kVoltageUnitVolts = 0.1;
constexpr double kPi = 3.14159265358979323846;

inline double ticksToRadians(int32_t ticks) {
  return (ticks - kCenterTicks) * (2.0 * kPi / kTicksPerRevolution);
}

inline int32_t radiansToTicks(double radians) {
  return static_cast<int32_t>(std::lround(radians * kTicksPerRevolution / (2.0 * kPi))) +
         kCenterTicks;
}

inline double velocityRawToRadPerSec(int32_t raw) {
  return raw * kVelocityUnitRpm * 2.0 * kPi / 60.0;
}

inline double currentRawToAmps(int16_t raw) { return raw * kCurrentUnitAmps; }

inline double voltageRawToVolts(uint16_t raw) { return raw * kVoltageUnitVolts; }

}  // namespace surgical_hand_xc330
