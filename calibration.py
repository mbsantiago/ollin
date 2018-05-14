"""
Calibration module for species simulator.

Calibration procedure looks for simulator parameters that minimize error with
respect to several sources:
    1. Home range information as calculated from the simulation and provided by
    biological data.
    2. Occupancy information provided by the article TODO.
"""
from functools import partial
from multiprocessing import Pool
import numpy as np

from species_data import SPECIES
from constants import (ALPHA, BETA, DT, RANGE, SEASON)

"""
So far simulation depends of the following parameters:
    1. DX: Distance resolution (in Km) for space discretization. The world
    rectangle is cut into squares of area DX^2. Such discretization is used for
    home-range and occupancy calculation.
    2. RANGE: Size in Km of world rectangle side.
    3. ALPHA: Fundamental parameter of species movement distribution. Species
    movement is modeled through a random selection of direction and a random
    distance picked for a Pareto(ALPHA) distribution. ALPHA in [1,2] is related
    to the resulting hausdorff dimesion of any sample trajectory.
    4. BETA: Unitless parameter to map home-range data into mean velocity
    information.
    5. SEASON: Duration (in days) of season to be used for occupancy
    calculation.
    6. CONE_RANGE: Distance (in Km) at which cameras can no longer detect an
    individual moving in its range. Used for detection calculation.
    7. CONE_ANGLE: Angle (in degrees) at which cameras can make detections.
    8. DT: Number of days to be used in the home range calculation.
"""

ARTICLE_SPECIES = [
    ("Spilogale putorius", 0.245, 0.023),
    ("Cervus canadensis", 0.585, 0.024),
    ("Puma concolor", 0.600, 0.030),
    ("Canis latrans", 0.861, 0.031),
    ("Lynx rufus", 0.970, 0.040),
    ("Urocyon cinereoargenteus", 0.400, 0.063),
    ("Ursus americanus", 0.504, 0.072),
    ("Odocoileus hemionus", 0.925, 0.141),
    ("Sylvilagus nuttallii", 0.925, 0.190)
]

# Home range MSE


def calculate_home_range_distribution(home_range, parameters=None):
    if parameters is None:
        parameters = {}
    alpha = parameters.get('alpha', ALPHA)
    beta = parameters.get('beta', BETA)
    range = parameters.get('range', RANGE)
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
        random_directions = np.stack(
            [np.cos(random_angles), np.sin(random_angles)], axis=-1)
        random_magnitudes = mean * np.random.pareto(alpha, [n_trials])
        random_directions *= random_magnitudes[:, None]

        tmp1 = random_positions + random_directions
        tmp2 = np.mod(tmp1, 2 * range)
        reflections = np.greater(tmp2, range)
        tmp3 = (1 - reflections) * np.mod(tmp2, range)
        tmp4 = reflections * np.mod(-tmp2, range)
        random_positions = tmp3 + tmp4

    areas = np.sum(space, axis=(0, 1)) * dx**2
    return areas


def calculate_single_mse_home_range(home_range, parameters=None, normalized=True):
    """calculate Mean Square Error of estimated home range vs given
    """
    distribution = calculate_home_range_distribution(
        home_range,
        parameters=parameters)
    if normalized:
        variance = np.var(distribution)
    else:
        variance = 1
    error = np.mean((distribution - home_range) ** 2 / variance)
    return error


def calculate_all_home_range_distributions(species=None, parameters=None):
    """Calculates distribution of simulated home range for all species

    Species simulated are contained in the SPECIES dictionary"""
    if species is None:
        species = SPECIES.keys()

    home_ranges = [SPECIES[spec]['ambito'] for spec in species]

    p_pool = Pool()
    errors = p_pool.map(
        partial(calculate_home_range_distribution, parameters=parameters),
        home_ranges)
    p_pool.close()
    p_pool.join()

    return np.array(errors)


def calculate_all_mse_home_range(species=None, parameters=None):
    """Calculates Mean Square Error average on all species.

    Uses the information stored in the SPECIES dictionary.
    """

    if species is None:
        species = SPECIES.keys()

    home_ranges = [SPECIES[spec]['ambito'] for spec in species]

    p_pool = Pool()
    errors = p_pool.map(
        partial(calculate_single_mse_home_range, parameters=parameters),
        home_ranges)
    p_pool.close()
    p_pool.join()

    return np.mean(errors)


def binary_search_calibration_home_range(
        alpha_range=[1.1, 1.9],
        beta_range=[30, 60],
        max_depth=5):
    for num_step in range(max_depth):
        print('Step {} of calibration process'.format(num_step))
        mini = 9999
        aindex = 0
        bindex = 0
        for nalpha in range(2):
            for nbeta in range(2):
                alpha = alpha_range[nalpha]
                beta = beta_range[nbeta]
                print('\t[+] STEP {}-{}'.format(num_step, 2*nalpha + nbeta + 1))
                print('\t[+] Calculating MSE for alpha={:2.3f} and beta={:2.3f}'.format(alpha, beta))
                error = calculate_all_mse_home_range(
                    parameters={'alpha': alpha, 'beta': beta})
                if error < mini:
                    aindex = nalpha
                    bindex = nbeta
                    mini = error
        print('Minimum error at step {} is {:2.3f}'.format(num_step, mini))

        malpha = (alpha_range[0] + alpha_range[1]) / 2.0
        mbeta = (beta_range[0] + beta_range[1]) / 2.0

        if aindex == 0:
            alpha_range = [alpha_range[0], malpha]
        else:
            alpha_range = [malpha, alpha_range[1]]

        if bindex == 0:
            beta_range = [beta_range[0], mbeta]
        else:
            beta_range = [mbeta, beta_range[1]]

        print('New alpha range: {}'.format(alpha_range))
        print('New beta range: {}'.format(beta_range))
    return alpha_range, beta_range


# Occupancy MSE
def calculate_occupancy_distribution(home_range, density, parameters=None):
    if parameters is None:
        parameters = {}

    alpha = parameters.get('alpha', ALPHA)
    beta = parameters.get('beta', BETA)
    range = parameters.get('range', RANGE)
    dt = parameters.get('dt', DT)
    season = parameters.get('season', SEASON)
    n_trials = parameters.get('n_trials', 1000)

    num = int(density * range ** 2)

    # Velocity is a function of home_range and dt
    velocity = beta * np.sqrt(home_range) / float(dt)
    dx = max(velocity, 0.01)

    # Making discretized version of space
    num_sides = int(np.ceil(range / dx))
    space = np.zeros([n_trials, num_sides, num_sides])

    # Initializing positions
    random_positions = np.random.uniform(0, range, [n_trials, num, 2])
    mean = (velocity * (alpha - 1)/alpha)

    for _ in xrange(season):
        indices = np.true_divide(random_positions, dx).astype(np.int)
        Z = np.linspace(0, n_trials, num * n_trials).astype(np.int).reshape([-1, 1])
        X, Y = np.split(indices.reshape([-1, 2]), 2, -1)
        space[Z, X, Y] = 1

        random_angles = np.random.uniform(0, 2 * np.pi, [n_trials, num])
        random_directions = np.stack(
                [np.cos(random_angles), np.sin(random_angles)], axis=-1)
        random_magnitudes = mean * np.random.pareto(alpha, [n_trials, num])
        random_directions *= random_magnitudes[:, :, None]

        tmp1 = random_positions + random_directions
        tmp2 = np.mod(tmp1, 2 * range)
        reflections = np.greater(tmp2, range)
        tmp3 = (1 - reflections) * np.mod(tmp2, range)
        tmp4 = reflections * np.mod(-tmp2, range)
        random_positions = tmp3 + tmp4

    areas = np.sum(space, axis=(1, 2)) / float(num_sides**2)
    return areas
