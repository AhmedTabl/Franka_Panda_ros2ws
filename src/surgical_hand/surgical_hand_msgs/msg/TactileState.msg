# Tactile state of ONE fingertip sensor (eFlesh-style).
#
# The eFlesh sensor (arXiv:2506.09994) is a magnet-embedded elastomer over a
# magnetometer array; contact deforms the elastomer and shifts the field.
# Raw magnetometer readings are the sensor's native output; forces/slip/
# location come from a calibration or learned model.
#
# Until real eFlesh hardware exists, this message is produced by the
# simulated tactile source (simulated=true). Consumers must treat the
# processed fields as unavailable when their *_valid flag is false.

std_msgs/Header header

string sensor_name        # e.g. "thumb", "index", "middle"
bool simulated            # true until real eFlesh hardware publishes

# Raw magnetometer data, flattened xyz triplets per magnetometer
# (eFlesh reference design: 5 magnetometers -> 15 values). Unit: uT.
float64[] raw_magnetometer_ut

# Processed contact estimates (from calibration model; synthetic when
# simulated). processed_valid=false means only the raw field is meaningful.
bool processed_valid
bool contact
float64 normal_force_n
float64 shear_force_x_n
float64 shear_force_y_n
float64 slip_probability      # 0..1
float64[3] contact_location_m # contact centroid in the sensor frame
