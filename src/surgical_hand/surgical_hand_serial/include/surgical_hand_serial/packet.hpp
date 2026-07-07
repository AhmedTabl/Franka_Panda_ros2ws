// Packet encoder/decoder for the host <-> Due protocol (see protocol.hpp).
// ROS-free and allocation-light so it stays reusable and easy to test.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "surgical_hand_serial/protocol.hpp"

namespace surgical_hand_serial {

struct Packet {
  uint8_t seq{0};
  uint8_t command{0};  // CommandId value, response flag included for replies
  std::vector<uint8_t> payload{};
};

// Serialize a packet into a wire frame. Throws std::length_error if the
// payload exceeds kMaxPayloadSize.
std::vector<uint8_t> encodePacket(const Packet& packet);

// Incremental decoder: feed bytes as they arrive (any chunking, including
// one byte at a time); a completed, CRC-valid packet is returned when its
// last byte is consumed.
//
// Resync policy: bytes outside a valid frame, frames with an out-of-range
// length, and frames failing CRC are dropped and counted in
// discardedBytes(). After a bad frame the decoder hunts for the next
// 0xAA 0x55 sync pair. (Bytes of a corrupted frame are not re-scanned for
// embedded sync pairs — acceptable for a point-to-point USB link where
// corruption is rare; the counter makes link quality observable.)
class PacketDecoder {
 public:
  std::optional<Packet> feed(uint8_t byte);
  uint64_t discardedBytes() const { return discarded_bytes_; }
  void reset();

 private:
  enum class State { kSync0, kSync1, kLength, kBody };

  State state_{State::kSync0};
  std::vector<uint8_t> body_{};  // [len, seq, cmd, payload...] then 2 CRC bytes
  size_t expected_body_size_{0};
  uint64_t discarded_bytes_{0};
};

// Little-endian payload helpers (match the firmware's byte order).
void appendU16(std::vector<uint8_t>& buffer, uint16_t value);
void appendU32(std::vector<uint8_t>& buffer, uint32_t value);
void appendI16(std::vector<uint8_t>& buffer, int16_t value);
void appendI32(std::vector<uint8_t>& buffer, int32_t value);
uint16_t readU16(const std::vector<uint8_t>& buffer, size_t offset);
uint32_t readU32(const std::vector<uint8_t>& buffer, size_t offset);
int16_t readI16(const std::vector<uint8_t>& buffer, size_t offset);
int32_t readI32(const std::vector<uint8_t>& buffer, size_t offset);

}  // namespace surgical_hand_serial
