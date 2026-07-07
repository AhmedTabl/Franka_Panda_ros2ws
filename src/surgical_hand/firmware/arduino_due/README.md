# Arduino Due Firmware (skeleton)

Firmware for the Arduino Due acting as the bridge between the host PC and
the DYNAMIXEL XC330-M288-T (and later the tendon/tactile electronics).

**STATUS: skeleton — written but NOT yet compiled or flashed.** The
host↔Due protocol (ping/status/heartbeat/write-lock) is implemented; all
DYNAMIXEL bus commands return `NOT_IMPLEMENTED` until the one-motor slice.
The host-side counterpart (protocol, client, CLI, tests) lives in
`surgical_hand_serial` and is fully unit/loopback tested; this file keeps a
synchronized copy of the protocol constants — keep them in sync.

## Safety behavior (implemented in this skeleton)

- Boots with writes locked and torque off.
- Write commands are refused (`WRITES_LOCKED`) until `UNLOCK_WRITES` with
  magic `0x5AFE` — which the host CLI only sends when the operator passes
  `--enable-torque`.
- Heartbeat watchdog: if torque is enabled and no heartbeat arrives within
  500 ms, torque is disabled and writes re-lock. (The actual motor torque-off
  bus write is a slice-5 TODO, marked in the code.)

## Build / flash (when ready — not done yet)

```bash
arduino-cli core install arduino:sam
arduino-cli compile --fqbn arduino:sam:arduino_due_x_dbg surgical_hand_due
arduino-cli upload  --fqbn arduino:sam:arduino_due_x_dbg -p /dev/ttyACM0 surgical_hand_due
# then, from the workspace:
ros2 run surgical_hand_serial hand_serial_cli --port /dev/ttyACM0 ping
```

The firmware currently talks on the **programming port** (`Serial`), so the
same cable flashes and communicates.

## Wiring assumptions and unknowns

Assumed:

- Host ↔ Due over USB (programming port), CDC serial, framed protocol from
  `surgical_hand_serial/protocol.hpp`.
- Due ↔ XC330 over half-duplex TTL DYNAMIXEL Protocol 2.0 on `Serial1`
  (TX1/RX1) with a GPIO direction pin — planned for slice 5.
- XC330 powered from a separate 5 V supply able to source the 1.8 A stall
  current, common ground with the Due. Never power the motor from the Due's
  5 V pin.

Unresolved (must be settled before wiring the motor):

1. **Logic levels / half-duplex interface.** The Due is a 3.3 V board with
   NOT-5V-tolerant pins; the DYNAMIXEL TTL bus idles at its logic high
   (typically ~5 V with a 5 V motor supply). A tri-state buffer/level
   shifter circuit (e.g. 74LVC2G241-style per ROBOTIS' half-duplex reference
   design) or a ready-made interface (DYNAMIXEL Shield, or a diode+resistor
   half-duplex circuit rated for 3.3 V) is required. **Do not wire TX/RX
   directly.**
2. Whether the Due stays in the loop for the motor at all vs. a U2D2/USB
   adapter direct to the PC for slice 5 bench characterization (the Due
   remains needed later for tactile/auxiliary sensors either way; the
   protocol above is transport-agnostic on purpose).
3. Direction-pin choice and timing for bus turnaround.
4. Connector/pinout of the XC330 (JST EH 3-pin: GND, VDD, DATA) harness.
