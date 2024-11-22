"""
Next, implement the resample_particles() function in line 328 using the
stochastic universal sampling algorithm. It takes a list of particles and the
corresponding importance weights as arguments and should return a list of the
resampled particles and weights. Here, the weight of a new particle is the same
as the weight of the particle from which it was sampled.
"""
import numpy as np

def resample_particles(particles, weights):
    """
    Resample particles using the stochastic universal sampling algorithm.
    
    Args:
        particles (list): List of particles to be resampled.
        weights (list): Corresponding importance weights of the particles.
    
    Returns:
        tuple: (new_particles, new_weights), where:
               - new_particles (list): Resampled particles.
               - new_weights (list): Weights of the resampled particles (same as original).
    """
    # Number of particles
    num_particles = len(particles)
    
    # Normalize weights to ensure they sum to 1
    normalized_weights = np.array(weights) / np.sum(weights)
    
    # Cumulative sum of weights
    cumulative_sum = np.cumsum(normalized_weights)
    
    # Generate equally spaced pointers along the cumulative weight distribution
    start = np.random.uniform(0, 1 / num_particles)
    pointers = [start + i / num_particles for i in range(num_particles)]
    
    # Resample particles using the pointers
    new_particles = []
    index = 0
    for pointer in pointers:
        # Find the first particle whose cumulative weight is greater than the pointer
        while pointer > cumulative_sum[index]:
            index += 1
        new_particles.append(particles[index])
    
    # All resampled particles inherit their original weights
    new_weights = [1.0 / num_particles] * num_particles

    return new_particles, new_weights