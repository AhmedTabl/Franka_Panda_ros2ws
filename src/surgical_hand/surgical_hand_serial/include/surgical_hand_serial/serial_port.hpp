// Minimal POSIX serial port wrapper (blocking reads with poll() timeout).
// No assumptions about device paths: the caller always names the device.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace surgical_hand_serial {

class SerialPort {
 public:
  // Open and configure a serial device (raw mode, 8N1, no flow control).
  // Throws std::runtime_error on failure. Common bauds only (9600..1000000).
  static SerialPort open(const std::string& device, int baud);

  // Adopt an already-open file descriptor without reconfiguring it
  // (used by the pty loopback tests).
  static SerialPort fromFd(int fd);

  SerialPort(SerialPort&& other) noexcept;
  SerialPort& operator=(SerialPort&& other) noexcept;
  SerialPort(const SerialPort&) = delete;
  SerialPort& operator=(const SerialPort&) = delete;
  ~SerialPort();

  // Read up to max_length bytes; returns 0 on timeout. Throws on I/O error
  // or if the device disappears.
  size_t read(uint8_t* buffer, size_t max_length, std::chrono::milliseconds timeout);

  // Write the whole buffer. Throws on I/O error.
  void write(const uint8_t* data, size_t length);

  int fd() const { return fd_; }

 private:
  explicit SerialPort(int fd) : fd_(fd) {}
  int fd_{-1};
};

}  // namespace surgical_hand_serial
