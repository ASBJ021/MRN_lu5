def add_random_particles(particles, weights, map_limits):
    """
    Replace half of the particles with the lowest weights with randomly
    sampled ones.

    Args:
        particles (list): Current set of particles [[x1, y1, theta1],
        [x2, y2, theta2], ...].
        weights (list): Importance weights corresponding to each particle.
        map_limits (list): [x_min, x_max, y_min, y_max] describing map
        boundaries.

    Returns:
        list: New set of particles.
    """
    # Ensure there are particles and weights
    num_particles = len(particles)
    num_replace = num_particles // 2  # Number of particles to replace

    # Sort particles by their weights (ascending order)
    sorted_indices = np.argsort(weights)
    indices_to_replace = sorted_indices[:num_replace]

    # Replace particles with lowest weights
    new_particles = particles.copy()
    for idx in indices_to_replace:
        new_particles[idx] = sample_random_particle(map_limits)

    return new_particles