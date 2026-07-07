// Surgical hand Arduino Due firmware skeleton.
//
// STATUS: SKELETON. Not yet compiled or flashed. The DYNAMIXEL bus path is
// stubbed (returns NOT_IMPLEMENTED); it lands in the one-motor slice.
// PING / GET_STATUS / HEARTBEAT / UNLOCK_WRITES are fully implemented so a
// flashed Due can be exercised end-to-end with hand_serial_cli before any
// motor is wired.
//
// Protocol: see surgical_hand_serial/include/surgical_hand_serial/protocol.hpp
// (this file keeps a synchronized copy of the constants — KEEP IN SYNC).
//
// SAFETY BEHAVIOR:
//   - Boots with writes LOCKED and torque OFF.
//   - Write commands return WRITES_LOCKED until UNLOCK_WRITES(0x5AFE).
//   - If no HEARTBEAT arrives for HEARTBEAT_TIMEOUT_MS while torque is
//     enabled, torque is disabled and writes re-lock (fail-safe).
//
// Wiring assumptions and unknowns: see ../README.md.

#include <stdint.h>

// ---- protocol constants (sync with protocol.hpp) --------------------------
static const uint8_t SYNC0 = 0xAA;
static const uint8_t SYNC1 = 0x55;
static const uint8_t PROTOCOL_VERSION = 1;
static const uint8_t FIRMWARE_VERSION = 1;
static const uint8_t RESPONSE_FLAG = 0x80;
static const uint8_t MAX_PAYLOAD = 64;
static const uint16_t UNLOCK_MAGIC = 0x5AFE;
static const uint32_t HEARTBEAT_TIMEOUT_MS = 500;

enum Command {
  CMD_PING = 0x01,
  CMD_GET_STATUS = 0x02,
  CMD_READ_MOTOR_STATE = 0x03,
  CMD_HEARTBEAT = 0x04,
  CMD_ENABLE_TORQUE = 0x10,
  CMD_SET_GOAL_POSITION = 0x11,
  CMD_SET_GOAL_CURRENT = 0x12,
  CMD_UNLOCK_WRITES = 0x1F,
};

enum Status {
  ST_OK = 0,
  ST_BAD_CRC = 1,
  ST_UNKNOWN_COMMAND = 2,
  ST_INVALID_PAYLOAD = 3,
  ST_WRITES_LOCKED = 4,
  ST_TORQUE_DISABLED = 5,
  ST_MOTOR_TIMEOUT = 6,
  ST_NOT_IMPLEMENTED = 7,
};

static const uint8_t FLAG_WRITES_UNLOCKED = 1 << 0;
static const uint8_t FLAG_TORQUE_ENABLED = 1 << 1;
static const uint8_t FLAG_HEARTBEAT_OK = 1 << 2;

// ---- state -----------------------------------------------------------------
static bool writes_unlocked = false;
static bool torque_enabled = false;
static uint32_t last_heartbeat_ms = 0;
static bool heartbeat_seen = false;

// Host link: the Due's PROGRAMMING port ("Serial", the port you flash
// through) — one cable for flashing and comms. Switch to SerialUSB (native
// USB port, faster, baud ignored) once throughput matters.
#define HOST Serial

// ---- crc16-ccitt (identical to host crc16.hpp) ------------------------------
static uint16_t crc16_ccitt(const uint8_t* data, uint16_t length) {
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < length; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

// ---- decoder (mirror of host PacketDecoder) ---------------------------------
enum DecodeState { DS_SYNC0, DS_SYNC1, DS_LENGTH, DS_BODY };
static DecodeState decode_state = DS_SYNC0;
static uint8_t body[3 + 64 + 2];  // len + seq + cmd + payload + crc
static uint16_t body_fill = 0;
static uint16_t body_expected = 0;

// Returns true when `body` holds a CRC-valid frame [len, seq, cmd, payload...].
static bool feed_byte(uint8_t byte) {
  switch (decode_state) {
    case DS_SYNC0:
      if (byte == SYNC0) decode_state = DS_SYNC1;
      return false;
    case DS_SYNC1:
      decode_state = (byte == SYNC1) ? DS_LENGTH : (byte == SYNC0 ? DS_SYNC1 : DS_SYNC0);
      return false;
    case DS_LENGTH:
      if (byte < 2 || byte > 2 + MAX_PAYLOAD) {
        decode_state = DS_SYNC0;
        return false;
      }
      body[0] = byte;
      body_fill = 1;
      body_expected = 1 + byte + 2;
      decode_state = DS_BODY;
      return false;
    case DS_BODY:
      body[body_fill++] = byte;
      if (body_fill < body_expected) return false;
      decode_state = DS_SYNC0;
      {
        const uint16_t crc_offset = body_fill - 2;
        const uint16_t received = (uint16_t)body[crc_offset] | ((uint16_t)body[crc_offset + 1] << 8);
        return received == crc16_ccitt(body, crc_offset);
      }
  }
  return false;
}

// ---- response helpers -------------------------------------------------------
static void send_response(uint8_t seq, uint8_t cmd, const uint8_t* payload, uint8_t payload_len) {
  uint8_t frame[5 + 64 + 2];
  const uint8_t len = 2 + payload_len;
  frame[0] = SYNC0;
  frame[1] = SYNC1;
  frame[2] = len;
  frame[3] = seq;
  frame[4] = cmd | RESPONSE_FLAG;
  for (uint8_t i = 0; i < payload_len; ++i) frame[5 + i] = payload[i];
  const uint16_t crc = crc16_ccitt(frame + 2, 3 + payload_len);
  frame[5 + payload_len] = crc & 0xFF;
  frame[6 + payload_len] = crc >> 8;
  HOST.write(frame, 7 + payload_len);
}

static void send_status_only(uint8_t seq, uint8_t cmd, uint8_t status) {
  send_response(seq, cmd, &status, 1);
}

static void put_u32(uint8_t* buffer, uint32_t value) {
  buffer[0] = value & 0xFF;
  buffer[1] = (value >> 8) & 0xFF;
  buffer[2] = (value >> 16) & 0xFF;
  buffer[3] = (value >> 24) & 0xFF;
}

// ---- safety -----------------------------------------------------------------
static void enforce_heartbeat() {
  if (!torque_enabled) return;
  const uint32_t age = millis() - last_heartbeat_ms;
  if (!heartbeat_seen || age > HEARTBEAT_TIMEOUT_MS) {
    // Fail-safe: kill torque and re-lock. TODO(slice 5): also send the
    // DYNAMIXEL torque-off write on the motor bus here.
    torque_enabled = false;
    writes_unlocked = false;
  }
}

// ---- command handlers ---------------------------------------------------------
static void handle_frame() {
  const uint8_t len = body[0];
  const uint8_t seq = body[1];
  const uint8_t cmd = body[2];
  const uint8_t* payload = &body[3];
  const uint8_t payload_len = len - 2;

  switch (cmd) {
    case CMD_PING: {
      const uint8_t out[] = {ST_OK, PROTOCOL_VERSION, FIRMWARE_VERSION};
      send_response(seq, cmd, out, sizeof(out));
      break;
    }
    case CMD_GET_STATUS: {
      uint8_t out[10];
      out[0] = ST_OK;
      out[1] = (writes_unlocked ? FLAG_WRITES_UNLOCKED : 0) |
               (torque_enabled ? FLAG_TORQUE_ENABLED : 0) |
               ((heartbeat_seen && millis() - last_heartbeat_ms <= HEARTBEAT_TIMEOUT_MS)
                    ? FLAG_HEARTBEAT_OK : 0);
      put_u32(out + 2, millis());
      put_u32(out + 6, heartbeat_seen ? millis() - last_heartbeat_ms : 0xFFFFFFFF);
      send_response(seq, cmd, out, sizeof(out));
      break;
    }
    case CMD_HEARTBEAT:
      last_heartbeat_ms = millis();
      heartbeat_seen = true;
      send_status_only(seq, cmd, ST_OK);
      break;
    case CMD_UNLOCK_WRITES:
      if (payload_len == 2 &&
          ((uint16_t)payload[0] | ((uint16_t)payload[1] << 8)) == UNLOCK_MAGIC) {
        writes_unlocked = true;
        // Unlocking arms the heartbeat requirement from now.
        last_heartbeat_ms = millis();
        heartbeat_seen = true;
        send_status_only(seq, cmd, ST_OK);
      } else {
        send_status_only(seq, cmd, ST_INVALID_PAYLOAD);
      }
      break;

    // ---- DYNAMIXEL bus commands: stubs until the one-motor slice ----------
    case CMD_READ_MOTOR_STATE:
      // TODO(slice 5): DYNAMIXEL Protocol 2.0 read of Present Position(132),
      // Velocity(128), Current(126), Voltage(144), Temperature(146).
      send_status_only(seq, cmd, ST_NOT_IMPLEMENTED);
      break;
    case CMD_ENABLE_TORQUE:
    case CMD_SET_GOAL_POSITION:
    case CMD_SET_GOAL_CURRENT:
      if (!writes_unlocked) {
        send_status_only(seq, cmd, ST_WRITES_LOCKED);
      } else {
        // TODO(slice 5): Torque Enable(64), Goal Position(116),
        // Goal Current(102) writes, gated on torque state.
        send_status_only(seq, cmd, ST_NOT_IMPLEMENTED);
      }
      break;

    default:
      send_status_only(seq, cmd, ST_UNKNOWN_COMMAND);
      break;
  }
}

// ---- arduino entry points -----------------------------------------------------
void setup() {
  HOST.begin(115200);  // CDC: rate ignored, call still required
  // TODO(slice 5): Serial1.begin(57600) + direction pin for the DYNAMIXEL
  // half-duplex TTL bus (see ../README.md for the level-shifter question).
}

void loop() {
  enforce_heartbeat();
  while (HOST.available() > 0) {
    if (feed_byte((uint8_t)HOST.read())) {
      handle_frame();
    }
  }
}
