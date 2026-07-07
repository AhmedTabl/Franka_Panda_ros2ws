#include "surgical_hand_serial/serial_port.hpp"

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace surgical_hand_serial {

namespace {

speed_t baudConstant(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 921600: return B921600;
    case 1000000: return B1000000;
    default:
      throw std::runtime_error("unsupported baud rate: " + std::to_string(baud));
  }
}

[[noreturn]] void throwErrno(const std::string& what) {
  throw std::runtime_error(what + ": " + std::strerror(errno));
}

}  // namespace

SerialPort SerialPort::open(const std::string& device, int baud) {
  const int fd = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
  if (fd < 0) {
    throwErrno("failed to open serial device " + device);
  }

  termios tty{};
  if (tcgetattr(fd, &tty) != 0) {
    ::close(fd);
    throwErrno("tcgetattr failed on " + device);
  }
  cfmakeraw(&tty);
  tty.c_cflag |= CLOCAL | CREAD;  // ignore modem lines, enable receiver
  tty.c_cflag &= ~CRTSCTS;        // no hardware flow control
  tty.c_cc[VMIN] = 0;             // poll() provides the timeout
  tty.c_cc[VTIME] = 0;
  cfsetispeed(&tty, baudConstant(baud));
  cfsetospeed(&tty, baudConstant(baud));
  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    ::close(fd);
    throwErrno("tcsetattr failed on " + device);
  }
  tcflush(fd, TCIOFLUSH);  // drop any stale bytes from before we opened
  return SerialPort(fd);
}

SerialPort SerialPort::fromFd(int fd) {
  if (fd < 0) {
    throw std::runtime_error("SerialPort::fromFd given invalid fd");
  }
  return SerialPort(fd);
}

SerialPort::SerialPort(SerialPort&& other) noexcept : fd_(other.fd_) {
  other.fd_ = -1;
}

SerialPort& SerialPort::operator=(SerialPort&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    fd_ = other.fd_;
    other.fd_ = -1;
  }
  return *this;
}

SerialPort::~SerialPort() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

size_t SerialPort::read(uint8_t* buffer, size_t max_length, std::chrono::milliseconds timeout) {
  pollfd pfd{};
  pfd.fd = fd_;
  pfd.events = POLLIN;
  const int poll_result = ::poll(&pfd, 1, static_cast<int>(timeout.count()));
  if (poll_result < 0) {
    throwErrno("poll failed");
  }
  if (poll_result == 0) {
    return 0;  // timeout
  }
  if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0 && (pfd.revents & POLLIN) == 0) {
    throw std::runtime_error("serial device error/hangup");
  }
  const ssize_t bytes_read = ::read(fd_, buffer, max_length);
  if (bytes_read < 0) {
    throwErrno("read failed");
  }
  return static_cast<size_t>(bytes_read);
}

void SerialPort::write(const uint8_t* data, size_t length) {
  size_t written = 0;
  while (written < length) {
    const ssize_t result = ::write(fd_, data + written, length - written);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throwErrno("write failed");
    }
    written += static_cast<size_t>(result);
  }
}

}  // namespace surgical_hand_serial
