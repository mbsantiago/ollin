"""
Calibration module for species simulator.

Calibration procedure looks for simulator parameters that minimize error with
respect to several sources:
    1. Home range information as calculated from the simulation and provided by
    biological data.
    2. Occupancy information provided by the article TODO.
"""
from functools import partial
import numpy as np
from multiprocessing import Pool

from species_data import SPECIES
from constants import *

"""
So far simulation depends of the following parameters:
    1. DX: Distance resolution (in Km) for space discretization. The world rectangle is
    cut into squares of area DX^2. Such discretization is used for home-range
    and occupancy calculation.
    2. RANGE: Size in Km of world rectangle side.
    3. ALPHA: Fundamental parameter of species movement distribution. Species movement 
    is modeled through a random selection of direction and a random distance picked
    for a Pareto(ALPHA) distribution. ALPHA in [1,2] is related to the resulting
    hausdorff dimesion of any sample trajectory.
    4. BETA: Unitless parameter to map home-range data into mean velocity information.
    5. SEASON: Duration (in days) of season to be used for occupancy calculation.
    6. CONE_RANGE: Distance (in Km) at which cameras can no longer detect an individual
    moving in its range. Used for detection calculation.
    7. CONE_ANGLE: Angle (in degrees) at which cameras can make detections.
    8. DT: Number of days to be used in the home range calculation.
"""

## Home range MSE

def calculate_home_range_distribution(home_range, parameters=None):
    if parameters is None:
        parameters = {}
    alpha = parameters.get('alpha', ALPHA)
    beta = parameters.get('beta', BETA)
    range = parameters.get('range', RANGE)
    # dx = parameters.get('dx', DX)
    dt = parameters.get('dt', DT)
    n_trials = parameters.get('n_trials', 1000)

    
    # Velocity is a function of home_range and dt
    velocity = beta * np.sqrt(home_range) / float(dt)
    dx = max(velocity, 0.01)

    # Making discretized version of space
    num_sides = int(np.ceil(range / dx))
    space = np.zeros([num_sides, num_sides, n_trials])

    # Initializing positions
    random_positions = np.random.uniform(0, range, [n_trials, 2])
    mean = (velocity * (alpha - 1)/alpha)

    for _ in xrange(dt):
        indices = np.true_divide(random_positions, dx).astype(np.int)
        for x in xrange(n_trials):
            index = indices[x]
            space[index[0], index[1], x] = 1

        random_angles = np.random.uniform(0, 2 * np.pi, [n_trials])
        random_directions = np.stack([np.cos(random_angles), np.sin(random_angles)], axis=-1)
        random_magnitudes = mean * np.random.pareto(alpha, [n_trials])
        random_directions *= random_magnitudes[:, None]

        tmp1 = random_positions + random_directions
        tmp2 = np.mod(tmp1, 2 * range)
        reflections = np.greater(tmp2, range)
        tmp3 = (1 - reflections) * np.mod(tmp2, range) + reflections * np.mod(-tmp2, range)
        random_positions = tmp3

    areas = np.sum(space, axis=(0, 1)) * dx**2
    return areas


def calculate_single_MSE_home_range(home_range, parameters=None):
    distribution = calculate_home_range_distribution(home_range, parameters=parameters)
    error = np.mean( (distribution - home_range) ** 2)
    return error


def calculate_all_home_range_distributions(species=None, parameters=None):
    if species is None:
        species = SPECIES.keys()

    home_ranges = [SPECIES[spec]['ambito'] for spec in species]

    p = Pool()
    errors = p.map(
        partial(calculate_home_range_distribution, parameters=parameters),
        home_ranges)
    p.close()
    p.join()

    return np.array(errors)


def calculate_all_MSE_home_range(species=None, parameters=None):

    if species is None:
        species = SPECIES.keys()

    home_ranges = [SPECIES[spec]['ambito'] for spec in species]

    p = Pool()
    errors = p.map(
        partial(calculate_single_MSE_home_range, parameters=parameters),
        home_ranges)
    p.close()
    p.join()

    return np.mean(errors)

## Occupancy MSE


