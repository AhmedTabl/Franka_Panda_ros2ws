// Bench CLI for the Arduino Due serial link.
//
// Read-only commands (always available):
//   hand_serial_cli --port /dev/ttyACM0 ping
//   hand_serial_cli --port /dev/ttyACM0 status
//   hand_serial_cli --port /dev/ttyACM0 read-motor 1
//   hand_serial_cli --port /dev/ttyACM0 heartbeat
//
// Write commands (each prints exactly what it is about to do, then requires
// the explicit safety flag; the firmware additionally rejects writes until
// the unlock command that only these gated paths send):
//   hand_serial_cli --port ... enable-torque 1        --enable-torque
//   hand_serial_cli --port ... goal-position 1 2048   --enable-torque
//   hand_serial_cli --port ... goal-current 1 100     --enable-torque
//   hand_serial_cli --port ... disable-torque 1       (no flag needed: safe direction)
//
// Options: --port <dev> (required), --baud <n> (default 115200; USB CDC
// ignores it), --timeout-ms <n> (default 200).
//
// There is deliberately no default port: never assume device paths.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "surgical_hand_serial/due_client.hpp"
#include "surgical_hand_serial/serial_port.hpp"

namespace shs = surgical_hand_serial;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: hand_serial_cli --port <device> [--baud N] [--timeout-ms N] <command>\n"
               "read-only: ping | status | read-motor <id> | heartbeat\n"
               "writes:    enable-torque <id> | goal-position <id> <ticks> |\n"
               "           goal-current <id> <mA>   (all require --enable-torque)\n"
               "           disable-torque <id>      (no flag: safe direction)\n");
  return 2;
}

bool confirmWrite(const std::string& description, bool flag_given) {
  std::printf("ABOUT TO WRITE TO HARDWARE: %s\n", description.c_str());
  if (!flag_given) {
    std::fprintf(stderr,
                 "refused: pass --enable-torque to allow this write command.\n"
                 "(defaults are read-only by design; see the surgical hand README)\n");
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string port_path;
  int baud = 115200;
  int timeout_ms = 200;
  bool enable_torque_flag = false;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--port" && i + 1 < argc) {
      port_path = argv[++i];
    } else if (arg == "--baud" && i + 1 < argc) {
      baud = std::atoi(argv[++i]);
    } else if (arg == "--timeout-ms" && i + 1 < argc) {
      timeout_ms = std::atoi(argv[++i]);
    } else if (arg == "--enable-torque") {
      enable_torque_flag = true;
    } else if (arg.rfind("--", 0) == 0) {
      std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
      return usage();
    } else {
      positional.push_back(arg);
    }
  }
  if (port_path.empty() || positional.empty()) {
    return usage();
  }
  const std::string command = positional[0];

  // Gate write commands BEFORE opening the port: a refused write must never
  // touch the device at all.
  const bool is_gated_write =
      command == "enable-torque" || command == "goal-position" || command == "goal-current";
  if (is_gated_write) {
    std::string description = command;
    for (size_t i = 1; i < positional.size(); ++i) {
      description += " " + positional[i];
    }
    if (!confirmWrite(description, enable_torque_flag)) {
      return 3;
    }
  }

  try {
    shs::SerialPort port = shs::SerialPort::open(port_path, baud);
    shs::DueClient client(port, std::chrono::milliseconds(timeout_ms));

    if (command == "ping") {
      const auto result = client.ping();
      std::printf("ping OK: protocol v%u, firmware v%u\n", result.protocol_version,
                  result.firmware_version);
    } else if (command == "status") {
      const auto status = client.getStatus();
      std::printf("uptime: %u ms\nwrites unlocked: %s\ntorque enabled: %s\n"
                  "heartbeat ok: %s (age %u ms)\n",
                  status.uptime_ms, status.writes_unlocked ? "yes" : "no",
                  status.torque_enabled ? "yes" : "no",
                  status.heartbeat_ok ? "yes" : "no", status.heartbeat_age_ms);
    } else if (command == "read-motor" && positional.size() >= 2) {
      const auto state = client.readMotorState(static_cast<uint8_t>(std::stoi(positional[1])));
      std::printf("motor %u: position %d ticks, velocity %d, current %d mA, "
                  "voltage %.1f V, temperature %d C\n",
                  state.motor_id, state.position_ticks, state.velocity_ticks,
                  state.current_ma, state.voltage_v, state.temperature_c);
    } else if (command == "heartbeat") {
      client.sendHeartbeat();
      std::printf("heartbeat acknowledged\n");
    } else if (command == "disable-torque" && positional.size() >= 2) {
      // Moving toward the safe state never needs a flag, but still needs
      // the firmware unlock (a locked firmware has torque off anyway).
      const uint8_t id = static_cast<uint8_t>(std::stoi(positional[1]));
      std::printf("disabling torque on motor %u\n", id);
      client.unlockWrites();
      client.enableTorque(id, false);
      std::printf("torque disabled\n");
    } else if (command == "enable-torque" && positional.size() >= 2) {
      const uint8_t id = static_cast<uint8_t>(std::stoi(positional[1]));
      client.unlockWrites();
      client.enableTorque(id, true);
      std::printf("torque enabled on motor %u\n", id);
    } else if (command == "goal-position" && positional.size() >= 3) {
      const uint8_t id = static_cast<uint8_t>(std::stoi(positional[1]));
      const int32_t ticks = std::stoi(positional[2]);
      client.unlockWrites();
      client.setGoalPosition(id, ticks);
      std::printf("goal position sent\n");
    } else if (command == "goal-current" && positional.size() >= 3) {
      const uint8_t id = static_cast<uint8_t>(std::stoi(positional[1]));
      const int16_t ma = static_cast<int16_t>(std::stoi(positional[2]));
      client.unlockWrites();
      client.setGoalCurrent(id, ma);
      std::printf("goal current sent\n");
    } else {
      return usage();
    }
  } catch (const shs::TimeoutError& e) {
    std::fprintf(stderr, "timeout: %s\nIs the Due connected on %s and flashed with the "
                         "surgical hand firmware?\n",
                 e.what(), port_path.c_str());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
