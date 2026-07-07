#include "surgical_hand_serial/packet.hpp"

#include <stdexcept>

#include "surgical_hand_serial/crc16.hpp"

namespace surgical_hand_serial {

std::vector<uint8_t> encodePacket(const Packet& packet) {
  if (packet.payload.size() > kMaxPayloadSize) {
    throw std::length_error("packet payload exceeds kMaxPayloadSize");
  }
  const uint8_t length = static_cast<uint8_t>(2 + packet.payload.size());

  std::vector<uint8_t> frame;
  frame.reserve(5 + packet.payload.size() + 2);
  frame.push_back(kSync0);
  frame.push_back(kSync1);
  frame.push_back(length);
  frame.push_back(packet.seq);
  frame.push_back(packet.command);
  frame.insert(frame.end(), packet.payload.begin(), packet.payload.end());

  // CRC over [len, seq, cmd, payload...] = frame bytes after the sync pair.
  const uint16_t crc = crc16Ccitt(frame.data() + 2, frame.size() - 2);
  frame.push_back(static_cast<uint8_t>(crc & 0xFF));
  frame.push_back(static_cast<uint8_t>(crc >> 8));
  return frame;
}

std::optional<Packet> PacketDecoder::feed(uint8_t byte) {
  switch (state_) {
    case State::kSync0:
      if (byte == kSync0) {
        state_ = State::kSync1;
      } else {
        ++discarded_bytes_;
      }
      return std::nullopt;

    case State::kSync1:
      if (byte == kSync1) {
        state_ = State::kLength;
      } else {
        // The 0xAA we saw was not a frame start (unless this byte itself
        // restarts a sync pair).
        discarded_bytes_ += (byte == kSync0) ? 1 : 2;
        state_ = (byte == kSync0) ? State::kSync1 : State::kSync0;
      }
      return std::nullopt;

    case State::kLength: {
      const size_t length = byte;
      if (length < 2 || length > 2u + kMaxPayloadSize) {
        discarded_bytes_ += 3;  // sync pair + bogus length byte
        state_ = State::kSync0;
        return std::nullopt;
      }
      body_.clear();
      body_.push_back(byte);
      expected_body_size_ = 1 + length + 2;  // len byte + body + crc
      state_ = State::kBody;
      return std::nullopt;
    }

    case State::kBody: {
      body_.push_back(byte);
      if (body_.size() < expected_body_size_) {
        return std::nullopt;
      }
      state_ = State::kSync0;

      const size_t crc_offset = body_.size() - 2;
      const uint16_t received_crc =
          static_cast<uint16_t>(body_[crc_offset]) |
          static_cast<uint16_t>(body_[crc_offset + 1]) << 8;
      const uint16_t computed_crc = crc16Ccitt(body_.data(), crc_offset);
      if (received_crc != computed_crc) {
        discarded_bytes_ += 2 + body_.size();  // whole frame incl. sync
        return std::nullopt;
      }

      Packet packet;
      packet.seq = body_[1];
      packet.command = body_[2];
      packet.payload.assign(body_.begin() + 3, body_.begin() + crc_offset);
      return packet;
    }
  }
  return std::nullopt;
}

void PacketDecoder::reset() {
  state_ = State::kSync0;
  body_.clear();
  expected_body_size_ = 0;
}

void appendU16(std::vector<uint8_t>& buffer, uint16_t value) {
  buffer.push_back(static_cast<uint8_t>(value & 0xFF));
  buffer.push_back(static_cast<uint8_t>(value >> 8));
}

void appendU32(std::vector<uint8_t>& buffer, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    buffer.push_back(static_cast<uint8_t>((value >> shift) & 0xFF));
  }
}

void appendI16(std::vector<uint8_t>& buffer, int16_t value) {
  appendU16(buffer, static_cast<uint16_t>(value));
}

void appendI32(std::vector<uint8_t>& buffer, int32_t value) {
  appendU32(buffer, static_cast<uint32_t>(value));
}

uint16_t readU16(const std::vector<uint8_t>& buffer, size_t offset) {
  if (offset + 2 > buffer.size()) {
    throw std::out_of_range("readU16 past end of buffer");
  }
  return static_cast<uint16_t>(buffer[offset]) |
         static_cast<uint16_t>(buffer[offset + 1]) << 8;
}

uint32_t readU32(const std::vector<uint8_t>& buffer, size_t offset) {
  if (offset + 4 > buffer.size()) {
    throw std::out_of_range("readU32 past end of buffer");
  }
  uint32_t value = 0;
  for (int i = 3; i >= 0; --i) {
    value = (value << 8) | buffer[offset + static_cast<size_t>(i)];
  }
  return value;
}

int16_t readI16(const std::vector<uint8_t>& buffer, size_t offset) {
  return static_cast<int16_t>(readU16(buffer, offset));
}

int32_t readI32(const std::vector<uint8_t>& buffer, size_t offset) {
  return static_cast<int32_t>(readU32(buffer, offset));
}

}  // namespace surgical_hand_serial
