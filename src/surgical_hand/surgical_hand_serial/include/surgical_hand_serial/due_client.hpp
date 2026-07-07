// Request/response client for the Arduino Due over a SerialPort.
//
// This layer is a transport: it does NOT gate writes itself. The safety
// gates live (a) in the CLI/application, which must collect explicit
// operator flags before calling unlockWrites()/write methods, and (b) in
// the firmware, which rejects writes until unlocked and re-locks on
// heartbeat timeout. Keeping the client gate-free makes the gating logic
// auditable in exactly two places instead of three.

#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "surgical_hand_serial/packet.hpp"
#include "surgical_hand_serial/protocol.hpp"
#include "surgical_hand_serial/serial_port.hpp"

namespace surgical_hand_serial {

struct PingResult {
  uint8_t protocol_version{0};
  uint8_t firmware_version{0};
};

struct DueStatus {
  bool writes_unlocked{false};
  bool torque_enabled{false};
  bool heartbeat_ok{false};
  uint32_t uptime_ms{0};
  uint32_t heartbeat_age_ms{0};
};

struct MotorState {
  uint8_t motor_id{0};
  int32_t position_ticks{0};
  int32_t velocity_ticks{0};
  int16_t current_ma{0};
  double voltage_v{0.0};
  int temperature_c{0};
};

class TimeoutError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Thrown when the firmware answers with a non-OK StatusCode.
class ProtocolError : public std::runtime_error {
 public:
  ProtocolError(StatusCode code, const std::string& context)
      : std::runtime_error(context + ": firmware returned " + statusCodeName(code)),
        code_(code) {}
  StatusCode code() const { return code_; }

 private:
  StatusCode code_;
};

class DueClient {
 public:
  DueClient(SerialPort& port, std::chrono::milliseconds timeout = std::chrono::milliseconds(200))
      : port_(port), timeout_(timeout) {}

  // Read-only.
  PingResult ping();
  DueStatus getStatus();
  MotorState readMotorState(uint8_t motor_id);
  void sendHeartbeat();

  // Writes (firmware rejects these until unlockWrites(); see header note).
  void unlockWrites();
  void enableTorque(uint8_t motor_id, bool enable);
  void setGoalPosition(uint8_t motor_id, int32_t position_ticks);
  void setGoalCurrent(uint8_t motor_id, int16_t current_ma);

 private:
  // Send a request and wait for its matching response (seq + command).
  // Returns the response payload AFTER the leading status byte, throwing
  // ProtocolError on non-OK status and TimeoutError on deadline.
  std::vector<uint8_t> transact(CommandId command, const std::vector<uint8_t>& payload,
                                size_t expected_response_payload);

  SerialPort& port_;
  std::chrono::milliseconds timeout_;
  PacketDecoder decoder_{};
  uint8_t next_seq_{0};
};

}  // namespace surgical_hand_serial
