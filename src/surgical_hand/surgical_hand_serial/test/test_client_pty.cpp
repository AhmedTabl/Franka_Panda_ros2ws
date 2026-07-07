// Loopback test: DueClient talking to a fake firmware over a pty pair.
// Verifies the full host-side path (SerialPort -> encode -> decode ->
// request/response matching -> payload parsing) with no hardware attached.
// The fake responder mirrors the real firmware's safety behavior: writes
// are rejected with WRITES_LOCKED until the unlock command arrives.

#include <pty.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "surgical_hand_serial/due_client.hpp"
#include "surgical_hand_serial/packet.hpp"
#include "surgical_hand_serial/serial_port.hpp"

namespace shs = surgical_hand_serial;

namespace {

// Minimal fake firmware living on the master end of a pty.
class FakeFirmware {
 public:
  explicit FakeFirmware(int fd) : fd_(fd), thread_([this] { run(); }) {}

  ~FakeFirmware() {
    running_ = false;
    thread_.join();
  }

  std::atomic<bool> writes_unlocked{false};
  std::atomic<bool> torque_enabled{false};
  std::atomic<int32_t> last_goal_position{0};

 private:
  void run() {
    shs::PacketDecoder decoder;
    uint8_t buffer[128];
    while (running_) {
      const ssize_t bytes_read = ::read(fd_, buffer, sizeof(buffer));
      if (bytes_read <= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      for (ssize_t i = 0; i < bytes_read; ++i) {
        if (auto request = decoder.feed(buffer[i])) {
          respond(*request);
        }
      }
    }
  }

  void respond(const shs::Packet& request) {
    shs::Packet response;
    response.seq = request.seq;
    response.command = request.command | shs::kResponseFlag;
    auto& payload = response.payload;
    const auto command = static_cast<shs::CommandId>(request.command);

    switch (command) {
      case shs::CommandId::kPing:
        payload = {ok(), shs::kProtocolVersion, 1};
        break;
      case shs::CommandId::kGetStatus: {
        uint8_t flags = 0;
        if (writes_unlocked) flags |= shs::kFlagWritesUnlocked;
        if (torque_enabled) flags |= shs::kFlagTorqueEnabled;
        payload = {ok(), flags};
        shs::appendU32(payload, 12345);  // uptime_ms
        shs::appendU32(payload, 50);     // heartbeat_age_ms
        break;
      }
      case shs::CommandId::kReadMotorState:
        payload = {ok(), request.payload.at(0)};
        shs::appendI32(payload, 2048);  // position
        shs::appendI32(payload, -5);    // velocity
        shs::appendI16(payload, 123);   // current mA
        payload.push_back(50);          // 5.0 V
        payload.push_back(34);          // 34 C
        break;
      case shs::CommandId::kHeartbeat:
        payload = {ok()};
        break;
      case shs::CommandId::kUnlockWrites:
        if (request.payload.size() == 2 &&
            shs::readU16(request.payload, 0) == shs::kUnlockMagic) {
          writes_unlocked = true;
          payload = {ok()};
        } else {
          payload = {static_cast<uint8_t>(shs::StatusCode::kInvalidPayload)};
        }
        break;
      case shs::CommandId::kEnableTorque:
        if (!writes_unlocked) {
          payload = {static_cast<uint8_t>(shs::StatusCode::kWritesLocked)};
        } else {
          torque_enabled = request.payload.at(1) != 0;
          payload = {ok()};
        }
        break;
      case shs::CommandId::kSetGoalPosition:
        if (!writes_unlocked) {
          payload = {static_cast<uint8_t>(shs::StatusCode::kWritesLocked)};
        } else if (!torque_enabled) {
          payload = {static_cast<uint8_t>(shs::StatusCode::kTorqueDisabled)};
        } else {
          last_goal_position = shs::readI32(request.payload, 1);
          payload = {ok()};
        }
        break;
      default:
        payload = {static_cast<uint8_t>(shs::StatusCode::kUnknownCommand)};
        break;
    }

    const auto frame = shs::encodePacket(response);
    ASSERT_EQ(::write(fd_, frame.data(), frame.size()),
              static_cast<ssize_t>(frame.size()));
  }

  static uint8_t ok() { return static_cast<uint8_t>(shs::StatusCode::kOk); }

  int fd_;
  std::atomic<bool> running_{true};
  std::thread thread_;
};

struct PtyPair {
  int master{-1};
  int slave{-1};
  PtyPair() {
    // Raw mode is essential: a default pty is a cooked terminal (canonical
    // line buffering, echo, \n -> \r\n output translation) and mangles the
    // binary protocol.
    termios raw{};
    cfmakeraw(&raw);
    if (openpty(&master, &slave, nullptr, &raw, nullptr) != 0) {
      throw std::runtime_error("openpty failed");
    }
  }
  ~PtyPair() {
    // slave fd ownership moves into SerialPort; master closed by firmware
    if (master >= 0) ::close(master);
  }
};

}  // namespace

TEST(DueClientPty, ReadOnlyRoundTrips) {
  PtyPair pty;
  FakeFirmware firmware(pty.master);
  shs::SerialPort port = shs::SerialPort::fromFd(pty.slave);
  shs::DueClient client(port, std::chrono::milliseconds(500));

  const auto ping = client.ping();
  EXPECT_EQ(ping.protocol_version, shs::kProtocolVersion);
  EXPECT_EQ(ping.firmware_version, 1);

  const auto status = client.getStatus();
  EXPECT_FALSE(status.writes_unlocked);
  EXPECT_FALSE(status.torque_enabled);
  EXPECT_EQ(status.uptime_ms, 12345u);
  EXPECT_EQ(status.heartbeat_age_ms, 50u);

  const auto motor = client.readMotorState(3);
  EXPECT_EQ(motor.motor_id, 3);
  EXPECT_EQ(motor.position_ticks, 2048);
  EXPECT_EQ(motor.velocity_ticks, -5);
  EXPECT_EQ(motor.current_ma, 123);
  EXPECT_NEAR(motor.voltage_v, 5.0, 1e-9);
  EXPECT_EQ(motor.temperature_c, 34);

  client.sendHeartbeat();  // must not throw
}

TEST(DueClientPty, WritesLockedUntilUnlock) {
  PtyPair pty;
  FakeFirmware firmware(pty.master);
  shs::SerialPort port = shs::SerialPort::fromFd(pty.slave);
  shs::DueClient client(port, std::chrono::milliseconds(500));

  // Locked firmware refuses torque and motion.
  try {
    client.enableTorque(1, true);
    FAIL() << "expected ProtocolError";
  } catch (const shs::ProtocolError& e) {
    EXPECT_EQ(e.code(), shs::StatusCode::kWritesLocked);
  }

  // Unlock, but torque still off -> motion refused as TORQUE_DISABLED.
  client.unlockWrites();
  try {
    client.setGoalPosition(1, 1000);
    FAIL() << "expected ProtocolError";
  } catch (const shs::ProtocolError& e) {
    EXPECT_EQ(e.code(), shs::StatusCode::kTorqueDisabled);
  }

  // Full explicit chain works.
  client.enableTorque(1, true);
  client.setGoalPosition(1, 1000);
  EXPECT_EQ(firmware.last_goal_position.load(), 1000);
}

TEST(DueClientPty, TimeoutWhenNoResponder) {
  PtyPair pty;  // no FakeFirmware attached to the master end
  shs::SerialPort port = shs::SerialPort::fromFd(pty.slave);
  shs::DueClient client(port, std::chrono::milliseconds(100));
  EXPECT_THROW(client.ping(), shs::TimeoutError);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
