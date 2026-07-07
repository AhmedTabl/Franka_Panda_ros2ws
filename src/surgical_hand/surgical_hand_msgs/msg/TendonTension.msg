# Estimated tendon tension derived from motor current.
#
# IMPORTANT: this is an OBSERVER output, not a direct tension measurement.
# It is affected by gearbox friction, spool/capstan friction, cable routing
# losses, cable slack, hysteresis, acceleration, and motor heating. Treat
# `confidence` and `status` as first-class data, not decoration.

std_msgs/Header header

# Estimation quality (mutually exclusive; see also over_safety_threshold).
uint8 STATUS_OK = 0              # estimate in the model's trusted region
uint8 STATUS_BELOW_DEADBAND = 1  # |current - offset| inside deadband; tension indistinguishable from friction/noise
uint8 STATUS_SLACK = 2           # model produced negative tension (unwinding / slack cable); tension_n clamped to 0
uint8 STATUS_SATURATED = 3       # current near the configured saturation/stall region; estimate unreliable
uint8 status

float64 tension_n          # filtered, clamped (>= 0) tendon tension estimate [N]
float64 raw_tension_n      # unfiltered, unclamped (signed) model output [N]
float64 motor_torque_nm    # estimated motor output torque after offset/deadband [N*m]
float64 motor_current_a    # input current sample this estimate was computed from [A]

float64 confidence         # heuristic 0..1; never 1.0 (it is an observer)

# True when tension_n exceeds the configured safety threshold. Checked
# independently of `status` so a saturated-but-dangerous reading still trips.
bool over_safety_threshold
