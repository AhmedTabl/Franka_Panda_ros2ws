// Thin, throwing wrapper around the DYNAMIXEL SDK for the XC330.
//
// Like DueClient in surgical_hand_serial, this is a TRANSPORT: it does not
// gate writes itself. Safety gating lives in the tools (xc330_cli requires
// --enable-torque / --write-eeprom) and in the motor's own protections.
// Keeping the client gate-free keeps the gate logic auditable in one place
// per tool.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "surgical_hand_xc330/xc330_control_table.hpp"

// Forward declarations so dependents don't need SDK headers.
namespace dynamixel {
class PortHandler;
class PacketHandler;
}  // namespace dynamixel

namespace surgical_hand_xc330 {

struct PingInfo {
  uint8_t id{0};
  uint16_t model_number{0};
  uint8_t firmware_version{0};
};

// Fast state: one 21-byte block read (Present Current..Present Temperature).
struct MotorState {
  int16_t current_raw{0};
  int32_t velocity_raw{0};
  int32_t position_ticks{0};
  uint16_t voltage_raw{0};
  uint8_t temperature_c{0};

  double currentAmps() const { return currentRawToAmps(current_raw); }
  double velocityRadPerSec() const { return velocityRawToRadPerSec(velocity_raw); }
  double positionRadians() const { return ticksToRadians(position_ticks); }
  double voltageVolts() const { return voltageRawToVolts(voltage_raw); }
};

// Slow/config state: individual register reads.
struct MotorConfig {
  bool torque_enabled{false};
  uint8_t operating_mode{0};
  uint16_t current_limit_ma{0};
  uint8_t hardware_error{0};
};

class Xc330Client {
 public:
  // Opens the U2D2 serial device and sets the bus baud rate. Throws
  // std::runtime_error on failure. No default device path by design.
  Xc330Client(const std::string& device, int baud);
  ~Xc330Client();
  Xc330Client(const Xc330Client&) = delete;
  Xc330Client& operator=(const Xc330Client&) = delete;

  // Change the host-side baud rate (same open port), for scanning.
  void setBaud(int baud);

  // Read-only.
  std::optional<PingInfo> ping(uint8_t id);
  std::vector<PingInfo> broadcastPing();
  MotorState readState(uint8_t id);
  MotorConfig readConfig(uint8_t id);

  // Writes (RAM).
  void writeTorqueEnable(uint8_t id, bool enable);
  void writeGoalPosition(uint8_t id, int32_t ticks);
  void writeGoalCurrent(uint8_t id, int16_t milliamps);

  // Writes (EEPROM; motor rejects them while torque is enabled).
  void writeOperatingMode(uint8_t id, uint8_t mode);
  void writeCurrentLimit(uint8_t id, uint16_t milliamps);

 private:
  uint32_t readBytes(uint8_t id, uint16_t address, uint16_t length);
  void writeBytes(uint8_t id, uint16_t address, uint32_t value, uint16_t length);

  dynamixel::PortHandler* port_{nullptr};
  dynamixel::PacketHandler* packet_{nullptr};
};

}  // namespace surgical_hand_xc330
