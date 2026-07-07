// Host <-> Arduino Due wire protocol for the surgical hand.
//
// Frame layout (little-endian multi-byte values):
//
//   [0xAA][0x55][len][seq][cmd][payload ...][crc_lo][crc_hi]
//
//   len  = 2 + payload size (counts seq + cmd + payload)
//   seq  = request sequence number, echoed in the response
//   cmd  = CommandId; responses set the kResponseFlag bit (cmd | 0x80)
//   crc  = CRC16-CCITT-FALSE over [len, seq, cmd, payload...]
//
// Every response payload starts with a StatusCode byte.
//
// SAFETY MODEL (defense in depth, mirrors the workspace safety rules):
//   - The firmware boots with writes LOCKED and torque off. All write
//     commands return kWritesLocked until kUnlockWrites arrives with the
//     magic value 0x5AFE.
//   - The host CLI only sends kUnlockWrites when the operator passes an
//     explicit safety flag (--enable-torque).
//   - The host must send kHeartbeat at least every kHeartbeatTimeoutMs;
//     on timeout the firmware disables torque and re-locks writes.
//
// The Arduino firmware keeps its own copy of these constants
// (firmware/arduino_due/...): KEEP THEM IN SYNC when editing.

#pragma once

#include <cstdint>

namespace surgical_hand_serial {

constexpr uint8_t kSync0 = 0xAA;
constexpr uint8_t kSync1 = 0x55;
constexpr uint8_t kProtocolVersion = 1;
constexpr uint8_t kResponseFlag = 0x80;
constexpr uint8_t kMaxPayloadSize = 64;
constexpr uint16_t kUnlockMagic = 0x5AFE;
constexpr uint32_t kHeartbeatTimeoutMs = 500;

enum class CommandId : uint8_t {
  // Read-only (always available)
  kPing = 0x01,            // -> [status, protocol_version, firmware_version]
  kGetStatus = 0x02,       // -> [status, flags, uptime_ms u32, heartbeat_age_ms u32]
  kReadMotorState = 0x03,  // [motor_id] -> [status, motor_id, position i32,
                           //     velocity i32, current_ma i16, voltage_dV u8, temp_C u8]
  kHeartbeat = 0x04,       // -> [status]

  // Writes (locked until kUnlockWrites; see safety model above)
  kEnableTorque = 0x10,     // [motor_id, enable u8] -> [status]
  kSetGoalPosition = 0x11,  // [motor_id, position_ticks i32] -> [status]
  kSetGoalCurrent = 0x12,   // [motor_id, current_ma i16] -> [status]
  kUnlockWrites = 0x1F,     // [magic u16 = 0x5AFE] -> [status]
};

enum class StatusCode : uint8_t {
  kOk = 0,
  kBadCrc = 1,
  kUnknownCommand = 2,
  kInvalidPayload = 3,
  kWritesLocked = 4,    // write command received while locked
  kTorqueDisabled = 5,  // motion command while torque off
  kMotorTimeout = 6,    // DYNAMIXEL bus did not answer
  kNotImplemented = 7,  // firmware skeleton stub (motor path lands in slice 5)
};

// GetStatus `flags` bits.
constexpr uint8_t kFlagWritesUnlocked = 1 << 0;
constexpr uint8_t kFlagTorqueEnabled = 1 << 1;
constexpr uint8_t kFlagHeartbeatOk = 1 << 2;

inline const char* statusCodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kOk: return "OK";
    case StatusCode::kBadCrc: return "BAD_CRC";
    case StatusCode::kUnknownCommand: return "UNKNOWN_COMMAND";
    case StatusCode::kInvalidPayload: return "INVALID_PAYLOAD";
    case StatusCode::kWritesLocked: return "WRITES_LOCKED";
    case StatusCode::kTorqueDisabled: return "TORQUE_DISABLED";
    case StatusCode::kMotorTimeout: return "MOTOR_TIMEOUT";
    case StatusCode::kNotImplemented: return "NOT_IMPLEMENTED";
  }
  return "UNKNOWN_STATUS";
}

}  // namespace surgical_hand_serial
