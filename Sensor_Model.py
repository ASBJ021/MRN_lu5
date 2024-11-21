  
#################################### SENSOR MODEL ############################################


def beam_circle_intersection(A, B, C, r):
    d_0 = np.abs(C)
    if d_0 > r:
        return np.array([])
    x_0 = -A*C
    y_0 = -B*C
    if math.isclose(d_0, r):
        return np.array([[x_0, y_0]])
    d = np.sqrt(r*r - C*C)
    x_1 = x_0 + B*d
    y_1 = y_0 - A*d
    x_2 = x_0 - B*d
    y_2 = y_0 + A*d
    return np.array([[x_1, y_1], [x_2, y_2]])


"""
returns the distance to the closest intersection between a beam and an array of circles
the beam starts at (x,y) and has the angle theta (rad) to the x-axis
circles is a numpy array with the structure [circle1, circle2,...]
where each element is a numpy array [x_c, y_c, r] describing a circle 
with the center (x_c, y_c) and radius r
"""
def distance_to_closest_circle(x, y, theta, circles):
    beam_dir_x = np.cos(theta)
    beam_dir_y = np.sin(theta)
    min_dist = float('inf')
    # iterate over all circles
    for circle in circles:
        # compute the line equation parameters for the beam
        # in a shifted coordinate system, with the circle center at origin
        x_shifted = x - circle[0]
        y_shifted = y - circle[1]
        # vector (A,B) is a normal vector orthogonal to the beam
        A = beam_dir_y
        B = -beam_dir_x
        C = -(A*x_shifted + B*y_shifted)
        intersections = beam_circle_intersection(A, B, C, circle[2])
        for isec in intersections:
            # check if intersection is in front of the robot
            dot_prod = (isec[0]-x_shifted)*beam_dir_x + (isec[1]-y_shifted)*beam_dir_y
            if dot_prod > 0.0:
                dist = np.sqrt(np.square(isec[0]-x_shifted)+np.square(isec[1]-y_shifted))
                if dist < min_dist:
                    min_dist = dist

    return min_dist


# Returns the distance to the closest intersection of a beam
# and the map borders. The beam starts at (x, y) and has
# the angle theta to the positive x-axis.
def distance_to_closest_border(x, y, theta, map_limits):
    # Calculate cosine and sine of the angle
    ct = np.cos(theta)
    st = np.sin(theta)

    # Initialize closest distances to infinity
    x_closest = float('inf')
    y_closest = float('inf')

    # Check horizontal distances (along the x-axis)
    if ct != 0.0:
        if ct > 0:
            # Beam pointing to the right (positive x direction)
            x_closest = (map_limits[1] - x) / ct
        else:
            # Beam pointing to the left (negative x direction)
            x_closest = (map_limits[0] - x) / ct

    # Check vertical distances (along the y-axis)
    if st != 0.0:
        if st > 0:
            # Beam pointing upwards (positive y direction)
            y_closest = (map_limits[3] - y) / st
        else:
            # Beam pointing downwards (negative y direction)
            y_closest = (map_limits[2] - y) / st

    # Return the minimum of the horizontal and vertical distances
    closest_distance = min(x_closest, y_closest)
    print(f"Closest distance to the border: {closest_distance}")


# Returns the expected range measurements for all beams
# given the robot pose (x, y, theta_rob)
# and the map, described by circles and map_limits
def get_z_exp(x, y, theta_rob, n_beams, z_max, circles, map_limits):
    beam_directions = norm_angle_arr(np.linspace(-np.pi/2, np.pi/2, n_beams) + theta_rob)
    z_exp = []
    for theta in beam_directions:
        dist = distance_to_closest_circle(x, y, theta, circles)
        if (dist > z_max):
            dist = distance_to_closest_border(x, y, theta, map_limits)
        z_exp.append(dist)
    return z_exp
    

"""
z_scan and z_scan_exp are numpy arrays containing the measured and expected range values (in cm)
b is the variance parameter of the measurement noise
z_max is the maximum range (in cm)
returns the probability of the scan according to the simplified beam-based model
"""
def beam_based_model(scan, z_scan, z_exp, var, z_max):
    prob_z = []
    for i in range(len(z_scan)):
        if z_exp[i] is not None:  # Check if z_exp[i] is not None
            prob_z_value = norm.pdf(z_scan[i], z_exp[i], np.sqrt(var))
            prob_z.append(prob_z_value)
        else:
            # Handle the case where z_exp[i] is None, e.g., use a default value or skip
            prob_z.append(0)  # or another appropriate default value
    return prob_z


def eval_sensor_model(scan, particles, obstacles, map_limits):
    n_beams = len(scan)
    z_max = 10.0
    std_dev = 1.2
    var = std_dev**2
    weights = []
    for particle in particles:
        z_exp = get_z_exp(particle[0], particle[1], particle[2], n_beams, z_max, obstacles, map_limits)
        weight = beam_based_model(scan, z_exp, var, z_max)
        weights.append(weight)

    weights = weights / sum(weights)

    return weights
