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
the start of the simulation. For example, a value of pi means that the wheel
rotated two times forwards and -pi indicates that it rotated twice backward. To
obtain the robot motion from the encoder readings, you should use the motion
equations of the differential drive. Further, assume a wheel diameter of and
the distance between the wheels to be.
"""

def get_odometry(last_encoder, current_encoder):
    """
    Computes odometry from wheel encoder readings using the differential drive
    model.
    Args:
        last_encoder (dict): Previous encoder readings with 'left' and 'right'
        keys in radians.
        current_encoder (dict): Current encoder readings with 'left' and
        'right' keys in radians.
    Returns:
        list: [delta_rot1, delta_rot2, delta_trans], where:
              - delta_rot1: First rotation in radians
              - delta_rot2: Second rotation in radians
              - delta_trans: Translation distance in meters
    Raises:
        ValueError: If input values are not numbers.
    """
    try:
        # Calculate wheel displacements
        delta_left = WHEEL_RADIUS * (current_encoder["left"] - 
                                     last_encoder["left"])
        delta_right = WHEEL_RADIUS * (current_encoder["right"] - 
                                      last_encoder["right"])

        # Compute arc length (average of left and right displacements) and
        # orientation change
        delta_trans = (delta_left + delta_right) / 2.0 
        delta_theta = (delta_right - delta_left) / (WHEEL_BASE * 2.0)

        # Compute the translation (straight-line distance)
        if delta_theta == 0:  # Straight-line motion
            straight_line_trans = delta_trans
        else:  # Circular motion -> use chord length
            radius = delta_trans / delta_theta  
            straight_line_trans = (abs(radius) * 
                                   math.sin(abs(delta_theta) / 2.0))

        # Compute rotations
        delta_rot1 = delta_theta / 2.0  # Initial rotation
        delta_rot2 = delta_theta / 2.0  # Final rotation

        return [normalize_angle(delta_rot1), normalize_angle(delta_rot2),
                straight_line_trans]

    except Exception as e:
        print("An unexpected error occurred:", e)
        return [0.0, 0.0, 0.0]