#include "surgical_hand_serial/due_client.hpp"

namespace surgical_hand_serial {

std::vector<uint8_t> DueClient::transact(CommandId command, const std::vector<uint8_t>& payload,
                                         size_t expected_response_payload) {
  Packet request;
  request.seq = next_seq_++;
  request.command = static_cast<uint8_t>(command);
  request.payload = payload;

  const std::vector<uint8_t> frame = encodePacket(request);
  port_.write(frame.data(), frame.size());

  const uint8_t expected_command = static_cast<uint8_t>(command) | kResponseFlag;
  const auto deadline = std::chrono::steady_clock::now() + timeout_;

  uint8_t buffer[256];
  while (std::chrono::steady_clock::now() < deadline) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    const size_t bytes_read =
        port_.read(buffer, sizeof(buffer), std::max(remaining, std::chrono::milliseconds(1)));
    for (size_t i = 0; i < bytes_read; ++i) {
      auto packet = decoder_.feed(buffer[i]);
      if (!packet) {
        continue;
      }
      if (packet->command != expected_command || packet->seq != request.seq) {
        continue;  // stale/mismatched response; keep scanning until deadline
      }
      if (packet->payload.empty()) {
        throw ProtocolError(StatusCode::kInvalidPayload, "response missing status byte");
      }
      const auto status = static_cast<StatusCode>(packet->payload[0]);
      if (status != StatusCode::kOk) {
        throw ProtocolError(status, "command 0x" + std::to_string(static_cast<int>(command)));
      }
      std::vector<uint8_t> rest(packet->payload.begin() + 1, packet->payload.end());
      if (rest.size() < expected_response_payload) {
        throw ProtocolError(StatusCode::kInvalidPayload, "response payload too short");
      }
      return rest;
    }
  }
  throw TimeoutError("no response from Due within timeout (command 0x" +
                     std::to_string(static_cast<int>(command)) + ")");
}

PingResult DueClient::ping() {
  const auto payload = transact(CommandId::kPing, {}, 2);
  PingResult result;
  result.protocol_version = payload[0];
  result.firmware_version = payload[1];
  return result;
}

DueStatus DueClient::getStatus() {
  const auto payload = transact(CommandId::kGetStatus, {}, 9);
  DueStatus status;
  status.writes_unlocked = (payload[0] & kFlagWritesUnlocked) != 0;
  status.torque_enabled = (payload[0] & kFlagTorqueEnabled) != 0;
  status.heartbeat_ok = (payload[0] & kFlagHeartbeatOk) != 0;
  status.uptime_ms = readU32(payload, 1);
  status.heartbeat_age_ms = readU32(payload, 5);
  return status;
}

MotorState DueClient::readMotorState(uint8_t motor_id) {
  const auto payload = transact(CommandId::kReadMotorState, {motor_id}, 13);
  MotorState state;
  state.motor_id = payload[0];
  state.position_ticks = readI32(payload, 1);
  state.velocity_ticks = readI32(payload, 5);
  state.current_ma = readI16(payload, 9);
  state.voltage_v = payload[11] * 0.1;
  state.temperature_c = payload[12];
  return state;
}

void DueClient::sendHeartbeat() {
  transact(CommandId::kHeartbeat, {}, 0);
}

void DueClient::unlockWrites() {
  std::vector<uint8_t> payload;
  appendU16(payload, kUnlockMagic);
  transact(CommandId::kUnlockWrites, payload, 0);
}

void DueClient::enableTorque(uint8_t motor_id, bool enable) {
  transact(CommandId::kEnableTorque, {motor_id, static_cast<uint8_t>(enable ? 1 : 0)}, 0);
}

void DueClient::setGoalPosition(uint8_t motor_id, int32_t position_ticks) {
  std::vector<uint8_t> payload{motor_id};
  appendI32(payload, position_ticks);
  transact(CommandId::kSetGoalPosition, payload, 0);
}

void DueClient::setGoalCurrent(uint8_t motor_id, int16_t current_ma) {
  std::vector<uint8_t> payload{motor_id};
  appendI16(payload, current_ma);
  transact(CommandId::kSetGoalCurrent, payload, 0);
}

}  // namespace surgical_hand_serial
