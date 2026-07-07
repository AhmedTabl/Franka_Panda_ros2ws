// CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no xor-out).
// Table-free so the exact same code can be pasted into the Arduino firmware.
// Known vector: "123456789" -> 0x29B1.

#pragma once

#include <cstddef>
#include <cstdint>

namespace surgical_hand_serial {

inline uint16_t crc16Ccitt(const uint8_t* data, size_t length, uint16_t crc = 0xFFFF) {
  for (size_t i = 0; i < length; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                           : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

}  // namespace surgical_hand_serial
