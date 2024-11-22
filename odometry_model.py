"""
At this point, the odometry is calculated from ground-truth robot poses
provided by the simulator. On a real robot, however, you usually would not have
that information and odometry would be calculated, e.g., from the readings of
wheel encoders. Please replace the get_odometry() function in line 432 with one
that computes odometry from the wheel encoder readings of the robot. The
odometry output should use the same parametrization as defined in the odometry-
based motion model. You can use leftSensor.getValue() and 
rightSensor.getValue() to obtain the current encoder readings. The returned
value is the accumulated rotation of the corresponding wheel in radians since
the start of the simulation. For example, a value of means that the wheel
rotated two times forwards and indicates that it rotated twice backward. To
obtain the robot motion from the encoder readings, you should use the motion
equations of the differential drive. Further, assume a wheel diameter of and
the distance between the wheels to be.
"""

def get_odometry(last_left_encoder, last_right_encoder, current_left_encoder, current_right_encoder):
    """
    Computes odometry from wheel encoder readings using the differential drive model.
    Args:
        last_left_encoder (float): Previous left wheel encoder reading in radians.
        last_right_encoder (float): Previous right wheel encoder reading in radians.
        current_left_encoder (float): Current left wheel encoder reading in radians.
        current_right_encoder (float): Current right wheel encoder reading in radians.
    Returns:
        list: [delta_rot1, delta_rot2, delta_trans], where:
              - delta_rot1: First rotation in radians
              - delta_rot2: Second rotation in radians
              - delta_trans: Translation distance in meters
    """
    # Calculate wheel displacements
    delta_left = (current_left_encoder - last_left_encoder) * WHEEL_RADIUS
    delta_right = (current_right_encoder - last_right_encoder) * WHEEL_RADIUS

    # Compute the change in orientation (delta_theta) and translation (delta_trans)
    delta_trans = (delta_left + delta_right) / 2.0
    delta_theta = (delta_right - delta_left) / WHEEL_BASE

    # Compute the first and second rotations
    if abs(delta_trans) > 0.01:
        delta_rot1 = normalize_angle(math.atan2(delta_trans, delta_theta))
        delta_rot2 = normalize_angle(delta_theta - delta_rot1)
    else:
        delta_rot1 = 0.0
        delta_rot2 = delta_theta

    return [delta_rot1, delta_rot2, delta_trans]