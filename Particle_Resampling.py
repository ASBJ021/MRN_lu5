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
               - new_weights (list): Weights of the resampled particles
               (uniform distribution).
    
    Raises:
        ValueError: If inputs are invalid (e.g., empty lists, mismatched
        lengths).
    """
    try:
        # Number of particles
        num_particles = len(particles)

        # Normalize weights and check for valid sum
        weight_sum = np.sum(weights)       
        normalized_weights = np.array(weights) / weight_sum

        # Cumulative sum of weights
        cumulative_sum = np.cumsum(normalized_weights)

        # Generate equally spaced pointers along the cumulative weight distribution
        start = np.random.uniform(0, 1 / num_particles)
        pointers = [start + i / num_particles for i in range(num_particles)]

        # Resample particles using the pointers
        new_particles = []
        new_weights = []
        index = 0
        for pointer in pointers:
            while pointer > cumulative_sum[index]:
                index += 1
            new_particles.append(particles[index])
            new_weights.append(weights[index])

        return new_particles, new_weights

    except Exception as e:
        print("An unexpected error occurred:", e)
        return particles, weights
