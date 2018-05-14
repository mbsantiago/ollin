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
import numpy as np  # pylint: disable=import-error

import movement
import home_range
import occupancy
from species_data import SPECIES, COMMON_SPECIES
from constants import (ALPHA, BETA, DT, RANGE, SEASON, MIN_VELOCITY)

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

# Home range MSE


def calculate_home_range_distribution(hrange, parameters=None):
    if parameters is None:
        parameters = {}
    alpha = parameters.get('alpha', ALPHA)
    beta = parameters.get('beta', BETA)
    range = parameters.get('range', RANGE)
    dt = parameters.get('dt', DT)
    n_trials = parameters.get('n_trials', 1000)

    # Velocity is a function of home_range and dt
    velocity = movement.home_range_to_velocity(hrange, beta=beta, dt=dt)
    dx = max(velocity, 0.01)

    movement_data = movement.make_data(
        velocity,
        num=n_trials,
        steps=dt,
        alpha=alpha,
        range=range)

    grid = home_range.make_grid(movement_data)
    areas = np.sum(grid, axis=(1, 2)) * dx**2
    return areas


def calculate_single_mse_home_range(
        hrange,
        parameters=None,
        normalized=True):
    """calculate Mean Square Error of estimated home range vs given
    """
    distribution = calculate_home_range_distribution(
        hrange,
        parameters=parameters)
    if normalized:
        variance = np.var(distribution)
    else:
        variance = 1
    error = np.mean((distribution - hrange) ** 2 / variance)
    return error


def calculate_all_home_range_distributions(species=None, parameters=None):
    """Calculates distribution of simulated home range for all species

    Species simulated are contained in the SPECIES dictionary"""
    if species is None:
        species = SPECIES.keys()

    home_ranges = [SPECIES[spec]['home_range'] for spec in species]

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

    home_ranges = [SPECIES[spec]['home_range'] for spec in species]

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
                msg = '\t---- STEP {}-{}----'
                msg = msg.format(num_step, 2 * nalpha + nbeta + 1)
                print(msg)
                msg = '\t[+] Calculating MSE for alpha={:2.3f} '
                msg += 'and beta={:2.3f}'
                msg = msg.format(alpha, beta)
                print(msg)
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
    velocity = movement.home_range_to_velocity(home_range, beta=beta, dt=dt)
    dx = max(velocity, MIN_VELOCITY)
    num_sides = int(np.ceil(range / dx))
    movement_data = movement.make_data(
        velocity,
        num=num*n_trials,
        steps=season,
        alpha=alpha, range=range)

    grid = occupancy.make_grid(movement_data, num_trials=n_trials)
    proportions = np.sum(grid, axis=(1, 2)) / float(num_sides ** 2)
    return proportions


def _aux_occupancy(data, parameters=None):
    home_range, density = data
    areas = calculate_occupancy_distribution(
        home_range, density, parameters=parameters)
    return areas


def calculate_all_occupancy_distributions(species=None, parameters=None):
    """Calculates distribution of simulated occupancy for all species

    Species simulated are contained in the COMMON_SPECIES dictionary"""
    if species is None:
        species = COMMON_SPECIES.keys()

    data = [
        (COMMON_SPECIES[spec]['home_range'], COMMON_SPECIES[spec]['density'])
        for spec in species]

    p_pool = Pool()
    dists = p_pool.map(
            partial(_aux_occupancy, parameters=parameters),
            data)
    p_pool.close()
    p_pool.join()

    return np.array(dists)
