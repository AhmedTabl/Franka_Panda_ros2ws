// Bench CLI for the XC330-M288-T via a U2D2 (direct USB, no Arduino).
//
// Read-only commands (always available, never write to the motor):
//   xc330_cli --port /dev/ttyUSB0 scan                     # all common bauds
//   xc330_cli --port /dev/ttyUSB0 --baud 57600 ping 1
//   xc330_cli --port /dev/ttyUSB0 --baud 57600 read 1
//   xc330_cli --port /dev/ttyUSB0 --baud 57600 monitor 1 --hz 20 --csv out.csv
//
// RAM writes (require --enable-torque; each prints what it will do):
//   xc330_cli ... torque-on 1                --enable-torque
//   xc330_cli ... goal-position 1 --deg 30   --enable-torque
//   xc330_cli ... goal-current 1 --ma 100    --enable-torque
//   xc330_cli ... torque-off 1               (no flag: safe direction)
//
// EEPROM writes (require --write-eeprom AND torque off; read back to verify):
//   xc330_cli ... set-mode 1 current-position   --write-eeprom
//   xc330_cli ... set-current-limit 1 --ma 500  --write-eeprom
//
// No default port, ID, or baud is assumed: scan finds the motor.

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "surgical_hand_xc330/xc330_client.hpp"
#include "surgical_hand_xc330/xc330_control_table.hpp"

namespace shx = surgical_hand_xc330;

namespace {

const int kScanBauds[] = {57600, 1000000, 115200, 2000000, 9600, 3000000, 4000000};

int usage() {
  std::fprintf(stderr,
      "usage: xc330_cli --port <device> [--baud N] <command>\n"
      "read-only: scan | ping <id> | read <id> |\n"
      "           monitor <id> [--hz N] [--duration S] [--csv FILE]\n"
      "RAM writes (--enable-torque): torque-on <id> | goal-position <id> --deg D|--ticks T |\n"
      "                              goal-current <id> --ma M | torque-off <id> (no flag)\n"
      "EEPROM writes (--write-eeprom, torque must be off):\n"
      "           set-mode <id> <current|velocity|position|extended-position|current-position|pwm>\n"
      "           set-current-limit <id> --ma M\n");
  return 2;
}

bool refuseWithoutFlag(const std::string& description, bool flag_given, const char* flag_name) {
  std::printf("ABOUT TO WRITE TO MOTOR: %s\n", description.c_str());
  if (!flag_given) {
    std::fprintf(stderr, "refused: pass %s to allow this. Defaults are read-only.\n", flag_name);
    return true;
  }
  return false;
}

void printState(const shx::MotorState& s) {
  std::printf("position:    %d ticks  (%.4f rad, %.2f deg)\n", s.position_ticks,
              s.positionRadians(), s.positionRadians() * 180.0 / shx::kPi);
  std::printf("velocity:    %d raw    (%.4f rad/s)\n", s.velocity_raw, s.velocityRadPerSec());
  std::printf("current:     %d mA    (%.3f A)\n", s.current_raw, s.currentAmps());
  std::printf("voltage:     %.1f V\n", s.voltageVolts());
  std::printf("temperature: %d C\n", s.temperature_c);
}

uint8_t modeFromName(const std::string& name) {
  if (name == "current") return 0;
  if (name == "velocity") return 1;
  if (name == "position") return 3;
  if (name == "extended-position") return 4;
  if (name == "current-position") return 5;
  if (name == "pwm") return 16;
  throw std::runtime_error("unknown operating mode: " + name);
}

int scan(const std::string& port_path) {
  std::printf("scanning %s (read-only broadcast ping at %zu baud rates)...\n", port_path.c_str(),
              sizeof(kScanBauds) / sizeof(kScanBauds[0]));
  bool found_any = false;
  shx::Xc330Client client(port_path, kScanBauds[0]);
  for (const int baud : kScanBauds) {
    try {
      client.setBaud(baud);
    } catch (const std::runtime_error&) {
      continue;  // host adapter cannot do this rate
    }
    const auto found = client.broadcastPing();
    for (const auto& info : found) {
      found_any = true;
      std::printf("  baud %-8d id %-3u model %u%s firmware v%u\n", baud, info.id,
                  info.model_number,
                  info.model_number == shx::kModelNumberXc330M288 ? " (XC330-M288)" : "",
                  info.firmware_version);
    }
  }
  if (!found_any) {
    std::printf("no motors found. Check power (external 5 V supply on), the JST cable, "
                "and that no other program holds the port.\n");
    return 1;
  }
  return 0;
}

int monitor(shx::Xc330Client& client, uint8_t id, double hz, double duration_s,
            const std::string& csv_path) {
  std::FILE* csv = nullptr;
  if (!csv_path.empty()) {
    csv = std::fopen(csv_path.c_str(), "w");
    if (csv == nullptr) {
      std::fprintf(stderr, "cannot open %s for writing\n", csv_path.c_str());
      return 1;
    }
  }
  std::FILE* out = csv != nullptr ? csv : stdout;
  std::fprintf(out,
               "t_s,position_ticks,position_rad,velocity_raw,velocity_rad_s,"
               "current_ma,current_a,voltage_v,temperature_c\n");

  const auto period = std::chrono::duration<double>(1.0 / hz);
  const auto start = std::chrono::steady_clock::now();
  auto next_sample = start;
  while (true) {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (duration_s > 0.0 && t >= duration_s) {
      break;
    }
    const auto state = client.readState(id);
    std::fprintf(out, "%.4f,%d,%.6f,%d,%.6f,%d,%.4f,%.1f,%u\n", t, state.position_ticks,
                 state.positionRadians(), state.velocity_raw, state.velocityRadPerSec(),
                 state.current_raw, state.currentAmps(), state.voltageVolts(),
                 state.temperature_c);
    std::fflush(out);
    next_sample += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    std::this_thread::sleep_until(next_sample);
  }
  if (csv != nullptr) {
    std::fclose(csv);
    std::printf("wrote %s\n", csv_path.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string port_path;
  int baud = 57600;  // XC330 factory default; scan covers the rest
  bool enable_torque_flag = false;
  bool write_eeprom_flag = false;
  double hz = 20.0;
  double duration_s = 0.0;  // 0 = until Ctrl-C
  std::string csv_path;
  double deg = NAN;
  double ticks = NAN;
  double ma = NAN;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (arg == "--port") port_path = next();
    else if (arg == "--baud") baud = std::atoi(next());
    else if (arg == "--hz") hz = std::atof(next());
    else if (arg == "--duration") duration_s = std::atof(next());
    else if (arg == "--csv") csv_path = next();
    else if (arg == "--deg") deg = std::atof(next());
    else if (arg == "--ticks") ticks = std::atof(next());
    else if (arg == "--ma") ma = std::atof(next());
    else if (arg == "--enable-torque") enable_torque_flag = true;
    else if (arg == "--write-eeprom") write_eeprom_flag = true;
    else if (arg.rfind("--", 0) == 0) { std::fprintf(stderr, "unknown option %s\n", arg.c_str()); return usage(); }
    else positional.push_back(arg);
  }
  if (port_path.empty() || positional.empty()) {
    return usage();
  }
  const std::string command = positional[0];

  // Gate write commands BEFORE opening the port (same rule as
  // hand_serial_cli): a refused write must never touch the device.
  {
    std::string description = command;
    for (size_t i = 1; i < positional.size(); ++i) description += " " + positional[i];
    if (!std::isnan(deg)) description += " --deg " + std::to_string(deg);
    if (!std::isnan(ticks)) description += " --ticks " + std::to_string(static_cast<int>(ticks));
    if (!std::isnan(ma)) description += " --ma " + std::to_string(static_cast<int>(ma));

    const bool needs_torque_flag =
        command == "torque-on" || command == "goal-position" || command == "goal-current";
    const bool needs_eeprom_flag = command == "set-mode" || command == "set-current-limit";
    if (needs_torque_flag &&
        refuseWithoutFlag(description, enable_torque_flag, "--enable-torque")) {
      return 3;
    }
    if (needs_eeprom_flag &&
        refuseWithoutFlag(description + "  [EEPROM]", write_eeprom_flag, "--write-eeprom")) {
      return 3;
    }
  }

  try {
    if (command == "scan") {
      return scan(port_path);
    }
    if (positional.size() < 2) {
      return usage();
    }
    const uint8_t id = static_cast<uint8_t>(std::stoi(positional[1]));
    shx::Xc330Client client(port_path, baud);

    if (command == "ping") {
      const auto info = client.ping(id);
      if (!info) {
        std::fprintf(stderr, "no answer from id %u at %d baud (try: scan)\n", id, baud);
        return 1;
      }
      std::printf("id %u: model %u%s, firmware v%u\n", info->id, info->model_number,
                  info->model_number == shx::kModelNumberXc330M288 ? " (XC330-M288)" : "",
                  info->firmware_version);

    } else if (command == "read") {
      const auto config = client.readConfig(id);
      std::printf("torque:      %s\n", config.torque_enabled ? "ENABLED" : "off");
      std::printf("mode:        %u (%s)\n", config.operating_mode,
                  shx::operatingModeName(config.operating_mode));
      std::printf("curr. limit: %u mA\n", config.current_limit_ma);
      std::printf("hw error:    0x%02X%s\n", config.hardware_error,
                  config.hardware_error != 0 ? "  <-- FAULT, see e-manual addr 70" : "");
      printState(client.readState(id));

    } else if (command == "monitor") {
      return monitor(client, id, hz, duration_s, csv_path);

    } else if (command == "torque-off") {
      std::printf("disabling torque on id %u (safe direction, no flag needed)\n", id);
      client.writeTorqueEnable(id, false);
      std::printf("torque off\n");

    } else if (command == "torque-on") {
      const auto config = client.readConfig(id);
      std::printf("(mode is %s, current limit %u mA)\n",
                  shx::operatingModeName(config.operating_mode), config.current_limit_ma);
      client.writeTorqueEnable(id, true);
      std::printf("torque ENABLED on id %u — motor will hold/track goals now\n", id);

    } else if (command == "goal-position") {
      int32_t goal_ticks = 0;
      if (!std::isnan(deg)) goal_ticks = shx::radiansToTicks(deg * shx::kPi / 180.0);
      else if (!std::isnan(ticks)) goal_ticks = static_cast<int32_t>(ticks);
      else { std::fprintf(stderr, "goal-position needs --deg or --ticks\n"); return 2; }
      const auto before = client.readState(id);
      std::printf("current position: %d ticks -> commanding %d ticks\n",
                  before.position_ticks, goal_ticks);
      client.writeGoalPosition(id, goal_ticks);
      std::printf("goal position sent (%d ticks)\n", goal_ticks);

    } else if (command == "goal-current") {
      if (std::isnan(ma)) { std::fprintf(stderr, "goal-current needs --ma\n"); return 2; }
      client.writeGoalCurrent(id, static_cast<int16_t>(ma));
      std::printf("goal current sent (%d mA)\n", static_cast<int>(ma));

    } else if (command == "set-mode" || command == "set-current-limit") {
      // EEPROM writes: gated above; torque must also already be off (we
      // refuse rather than silently disabling torque ourselves).
      if (command == "set-mode" && positional.size() < 3) return usage();
      if (command == "set-current-limit" && std::isnan(ma)) {
        std::fprintf(stderr, "set-current-limit needs --ma\n");
        return 2;
      }
      const auto config = client.readConfig(id);
      if (config.torque_enabled) {
        std::fprintf(stderr, "refused: torque is enabled; EEPROM writes need torque off.\n"
                             "run: xc330_cli --port ... torque-off %u\n", id);
        return 3;
      }
      if (command == "set-mode") {
        const uint8_t mode = modeFromName(positional[2]);
        std::printf("mode before: %s\n", shx::operatingModeName(config.operating_mode));
        client.writeOperatingMode(id, mode);
        const auto after = client.readConfig(id);
        std::printf("mode after:  %s %s\n", shx::operatingModeName(after.operating_mode),
                    after.operating_mode == mode ? "(verified)" : "(MISMATCH!)");
        if (after.operating_mode != mode) return 1;
      } else {
        std::printf("limit before: %u mA\n", config.current_limit_ma);
        client.writeCurrentLimit(id, static_cast<uint16_t>(ma));
        const auto after = client.readConfig(id);
        std::printf("limit after:  %u mA %s\n", after.current_limit_ma,
                    after.current_limit_ma == static_cast<uint16_t>(ma) ? "(verified)"
                                                                        : "(MISMATCH!)");
        if (after.current_limit_ma != static_cast<uint16_t>(ma)) return 1;
      }

    } else {
      return usage();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
