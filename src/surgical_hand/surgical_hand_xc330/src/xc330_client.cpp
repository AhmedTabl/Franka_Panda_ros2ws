#include "surgical_hand_xc330/xc330_client.hpp"

#include <stdexcept>

#include "dynamixel_sdk/dynamixel_sdk.h"

namespace surgical_hand_xc330 {

namespace {
constexpr double kProtocolVersion = 2.0;

[[noreturn]] void throwComm(dynamixel::PacketHandler* packet, int comm_result,
                            uint8_t dxl_error, const std::string& context) {
  std::string message = context + ": ";
  if (comm_result != COMM_SUCCESS) {
    message += packet->getTxRxResult(comm_result);
  } else {
    message += std::string("motor error: ") + packet->getRxPacketError(dxl_error);
  }
  throw std::runtime_error(message);
}
}  // namespace

Xc330Client::Xc330Client(const std::string& device, int baud) {
  port_ = dynamixel::PortHandler::getPortHandler(device.c_str());
  packet_ = dynamixel::PacketHandler::getPacketHandler(kProtocolVersion);
  if (!port_->openPort()) {
    throw std::runtime_error("failed to open DYNAMIXEL device " + device);
  }
  if (!port_->setBaudRate(baud)) {
    port_->closePort();
    throw std::runtime_error("failed to set baud rate " + std::to_string(baud) + " on " + device);
  }
}

Xc330Client::~Xc330Client() {
  if (port_ != nullptr) {
    port_->closePort();
  }
}

void Xc330Client::setBaud(int baud) {
  if (!port_->setBaudRate(baud)) {
    throw std::runtime_error("failed to set baud rate " + std::to_string(baud));
  }
}

std::optional<PingInfo> Xc330Client::ping(uint8_t id) {
  uint16_t model = 0;
  uint8_t dxl_error = 0;
  const int comm = packet_->ping(port_, id, &model, &dxl_error);
  if (comm != COMM_SUCCESS) {
    return std::nullopt;  // no answer at this id/baud — normal during scans
  }
  PingInfo info;
  info.id = id;
  info.model_number = model;
  // Firmware version register (6) is 1 byte; best effort.
  try {
    info.firmware_version = static_cast<uint8_t>(readBytes(id, 6, 1));
  } catch (const std::runtime_error&) {
    info.firmware_version = 0;
  }
  return info;
}

std::vector<PingInfo> Xc330Client::broadcastPing() {
  std::vector<uint8_t> ids;
  const int comm = packet_->broadcastPing(port_, ids);
  std::vector<PingInfo> found;
  if (comm != COMM_SUCCESS) {
    return found;
  }
  for (const uint8_t id : ids) {
    if (auto info = ping(id)) {
      found.push_back(*info);
    }
  }
  return found;
}

MotorState Xc330Client::readState(uint8_t id) {
  // Single contiguous block read: current(2) velocity(4) position(4)
  // [reserved/trajectory 136..143] voltage(2) temperature(1).
  uint8_t buffer[kStateBlockLength] = {0};
  uint8_t dxl_error = 0;
  const int comm =
      packet_->readTxRx(port_, id, kStateBlockAddr, kStateBlockLength, buffer, &dxl_error);
  if (comm != COMM_SUCCESS || dxl_error != 0) {
    throwComm(packet_, comm, dxl_error, "readState(id=" + std::to_string(id) + ")");
  }

  auto u16 = [&buffer](int offset) {
    return static_cast<uint16_t>(buffer[offset] | (buffer[offset + 1] << 8));
  };
  auto u32 = [&buffer](int offset) {
    return static_cast<uint32_t>(buffer[offset]) | (static_cast<uint32_t>(buffer[offset + 1]) << 8) |
           (static_cast<uint32_t>(buffer[offset + 2]) << 16) |
           (static_cast<uint32_t>(buffer[offset + 3]) << 24);
  };

  MotorState state;
  state.current_raw = static_cast<int16_t>(u16(kAddrPresentCurrent - kStateBlockAddr));
  state.velocity_raw = static_cast<int32_t>(u32(kAddrPresentVelocity - kStateBlockAddr));
  state.position_ticks = static_cast<int32_t>(u32(kAddrPresentPosition - kStateBlockAddr));
  state.voltage_raw = u16(kAddrPresentInputVoltage - kStateBlockAddr);
  state.temperature_c = buffer[kAddrPresentTemperature - kStateBlockAddr];
  return state;
}

MotorConfig Xc330Client::readConfig(uint8_t id) {
  MotorConfig config;
  config.torque_enabled = readBytes(id, kAddrTorqueEnable, 1) != 0;
  config.operating_mode = static_cast<uint8_t>(readBytes(id, kAddrOperatingMode, 1));
  config.current_limit_ma = static_cast<uint16_t>(readBytes(id, kAddrCurrentLimit, 2));
  config.hardware_error = static_cast<uint8_t>(readBytes(id, kAddrHardwareErrorStatus, 1));
  return config;
}

void Xc330Client::writeTorqueEnable(uint8_t id, bool enable) {
  writeBytes(id, kAddrTorqueEnable, enable ? 1 : 0, 1);
}

void Xc330Client::writeGoalPosition(uint8_t id, int32_t ticks) {
  writeBytes(id, kAddrGoalPosition, static_cast<uint32_t>(ticks), 4);
}

void Xc330Client::writeGoalCurrent(uint8_t id, int16_t milliamps) {
  writeBytes(id, kAddrGoalCurrent, static_cast<uint16_t>(milliamps), 2);
}

void Xc330Client::writeOperatingMode(uint8_t id, uint8_t mode) {
  writeBytes(id, kAddrOperatingMode, mode, 1);
}

void Xc330Client::writeCurrentLimit(uint8_t id, uint16_t milliamps) {
  writeBytes(id, kAddrCurrentLimit, milliamps, 2);
}

uint32_t Xc330Client::readBytes(uint8_t id, uint16_t address, uint16_t length) {
  uint8_t dxl_error = 0;
  int comm = COMM_TX_FAIL;
  uint32_t value = 0;
  if (length == 1) {
    uint8_t v = 0;
    comm = packet_->read1ByteTxRx(port_, id, address, &v, &dxl_error);
    value = v;
  } else if (length == 2) {
    uint16_t v = 0;
    comm = packet_->read2ByteTxRx(port_, id, address, &v, &dxl_error);
    value = v;
  } else {
    comm = packet_->read4ByteTxRx(port_, id, address, &value, &dxl_error);
  }
  if (comm != COMM_SUCCESS || dxl_error != 0) {
    throwComm(packet_, comm, dxl_error,
              "read(addr=" + std::to_string(address) + ", id=" + std::to_string(id) + ")");
  }
  return value;
}

void Xc330Client::writeBytes(uint8_t id, uint16_t address, uint32_t value, uint16_t length) {
  uint8_t dxl_error = 0;
  int comm = COMM_TX_FAIL;
  if (length == 1) {
    comm = packet_->write1ByteTxRx(port_, id, address, static_cast<uint8_t>(value), &dxl_error);
  } else if (length == 2) {
    comm = packet_->write2ByteTxRx(port_, id, address, static_cast<uint16_t>(value), &dxl_error);
  } else {
    comm = packet_->write4ByteTxRx(port_, id, address, value, &dxl_error);
  }
  if (comm != COMM_SUCCESS || dxl_error != 0) {
    throwComm(packet_, comm, dxl_error,
              "write(addr=" + std::to_string(address) + ", id=" + std::to_string(id) + ")");
  }
}

}  // namespace surgical_hand_xc330
