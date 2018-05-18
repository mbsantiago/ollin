"""
Calibration module for species simulator.

Calibration procedure looks for simulator parameters that minimize error with
respect to several sources:
    1. Home range information as calculated from the simulation and provided by
    biological data.
    2. Occupancy information provided by the article TODO.
"""
# pylint: disable=F403
from __future__ import print_function

from functools import partial
from multiprocessing import Pool
import numpy as np  # pylint: disable=import-error
from tqdm import tqdm

import movement
import home_range
import occupancy
from constants import * 

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
def calculate_home_range_distribution_0(hrange, occup, parameters=None):
    if parameters is None:
        parameters = {}

    beta = parameters.get('beta', BETA)
    range_ = parameters.get('range', RANGE)
    power = parameters.get('power', POWER)
    dt = parameters.get('dt', DT)
    n_individuals = parameters.get('n_individuals', 100)

    # Velocity is a function of home_range and dt
    velocity = movement.home_range_to_velocity(
        hrange, beta=beta, dt=dt, power=power)

    movement_data = movement.make_data(
        velocity,
        occup,
        num=n_individuals,
        steps=dt,
        range=range_)

    grid = home_range.make_grid(movement_data)
    areas = np.sum(grid, axis=(1, 2))
    return areas


def calculate_home_range_distribution(hrange, occup, parameters=None):
    if parameters is None:
        parameters = {}

    n_trials = parameters.get('n_trials', 10)

    p = Pool()
    data = p.map(
        partial(
            calculate_home_range_distribution_0,
            occup=occup,
            parameters=parameters),
        [hrange for _ in xrange(n_trials)])
    p.close()
    p.join()
    return np.concatenate(data, 0)


def calculate_single_mse_home_range(
        hrange,
        occup,
        parameters=None,
        normalized=True):
    """calculate Mean Square Error of estimated home range vs given
    """
    distribution = calculate_home_range_distribution(
        hrange,
        occup,
        parameters=parameters)
    if normalized:
        variance = np.var(distribution)
    else:
        variance = 1
    error = np.mean((distribution - hrange) ** 2 / variance)
    return error


def _aux_hr_0(x, parameters=None):
    if parameters is None:
        parameters = {}
    (_, hr), (_, oc), _ = x
    dist = calculate_home_range_distribution_0(
        hr, oc, parameters=parameters)
    return dist


def calculate_all_home_range_distributions(parameters=None):
    """Calculates distribution of simulated home range for all species
    """
    if parameters is None:
        parameters = {}

    n_home_range = parameters.get('n_home_range', N_HOME_RANGE)
    n_occupancy = parameters.get('n_occupancy', N_OCCUPANCY)

    min_home_range = parameters.get('min_home_range', MIN_HOME_RANGE)
    max_home_range = parameters.get('max_home_range', MAX_HOME_RANGE)
    min_occupancy = parameters.get('min_occupancy', MIN_OCCUPANCY)
    max_occupancy = parameters.get('max_occupancy', MAX_OCCUPANCY)

    home_ranges = np.linspace(
            min_home_range, max_home_range, n_home_range)
    occupancies = np.linspace(
            min_occupancy, max_occupancy, n_occupancy)

    n_trials = parameters.get('n_trials', N_TRIALS)
    n_individuals = parameters.get('n_individuals', N_INDIVIDUALS)

    all_args = [
            (hr, oc, k)
            for hr in enumerate(home_ranges)
            for oc in enumerate(occupancies)
            for k in xrange(n_trials)]

    p = Pool()
    results = p.map(partial(_aux_hr_0, parameters=parameters), all_args)
    p.close()
    p.join()

    fresults = np.zeros(
        [n_home_range, n_occupancy, n_trials, n_individuals])

    for ((ihr, _), (ioc, _), k), res in zip(all_args, results):
        fresults[ihr, ioc, k, :] = res


    # results = np.array(results).reshape([
        # n_home_range,
        # n_occupancy,
        # n_trials,
        # n_individuals])

    return home_ranges, occupancies, fresults


def calculate_all_mse_home_range(parameters=None):
    """Calculates Mean Square Error average on all species.

    Uses the information stored in the SPECIES dictionary.
    """
    home_ranges, occupancies, areas = calculate_all_home_range_distributions(
            parameters=parameters)
    differences = (areas - home_ranges[:, None, None, None])**2
    errors = np.mean(differences, axis=(1, 2, 3))
    variances = np.maximum(np.var(differences, axis=(1, 2, 3)), 1)

    normalized_error = errors / variances
    return np.mean(normalized_error)


def sweep_search_calibration_home_range(
        power_range=[0.3, 0.6],
        beta_range=[40, 80],
        resolution=10,
        range=RANGE):
    power_range = np.linspace(power_range[0], power_range[1], resolution)
    beta_range = np.linspace(beta_range[0], beta_range[1], resolution)

    mini = 9999999
    amin = 0
    bmin = 0

    arguments = [
        (i, j, {'power': power_range[i], 'beta': beta_range[j]})
        for i in xrange(resolution) for j in xrange(resolution)
    ]
    result = np.zeros([resolution, resolution])

    for i, j, arg in tqdm(arguments):
        err = calculate_all_mse_home_range(parameters=arg)
        result[i, j] = err

        if err < mini:
            mini = err
            amin = power_range[i]
            bmin = beta_range[j]

    return result, mini, amin, bmin


def binary_search_calibration_home_range(
        power_range=[0.3, 0.6],
        beta_range=[40, 80],
        max_depth=5,
        range=RANGE):

    for num_step in xrange(max_depth):
        print('Step {}/{} of calibration process'.format(
            num_step + 1,
            max_depth))
        mini = 9999
        aindex = 0
        bindex = 0
        for npower in xrange(2):
            for nbeta in xrange(2):
                power = power_range[npower]
                beta = beta_range[nbeta]
                msg = '\t---- STEP {}-{}----'
                msg = msg.format(num_step + 1, 2 * npower + nbeta + 1)
                print(msg)
                msg = '\t[+] Calculating MSE for power={:2.3f} '
                msg += 'and beta={:2.3f}'
                msg = msg.format(power, beta)
                print(msg)
                error = calculate_all_mse_home_range(
                    parameters={'range': range, 'power': power, 'beta': beta})
                if error < mini:
                    aindex = npower
                    bindex = nbeta
                    mini = error
        print('Minimum error at step {} is {:2.3f}'.format(num_step + 1, mini))

        mpower = (power_range[0] + power_range[1]) / 2.0
        mbeta = (beta_range[0] + beta_range[1]) / 2.0

        if aindex == 0:
            power_range = [power_range[0], mpower]
        else:
            power_range = [mpower, power_range[1]]

        if bindex == 0:
            beta_range = [beta_range[0], mbeta]
        else:
            beta_range = [mbeta, beta_range[1]]

        print('New power range: {}'.format(power_range))
        print('New beta range: {}'.format(beta_range))
    return power_range, beta_range


# Occupancy MSE
def calculate_occupancy_distribution(home_range, occup, parameters=None):
    if parameters is None:
        parameters = {}

    # Movement parameters
    beta = parameters.get('beta', BETA)
    range = parameters.get('range', RANGE)
    dt = parameters.get('dt', DT)

    # Occupancy parameters
    season = parameters.get('season', SEASON)
    gamma = parameters.get('gamma', GAMMA)
    omega = parameters.get('omega', OMEGA)

    # Statistical parameters
    n_trials = parameters.get('n_trials', 1000)

    num = range**2 * occupancy.occupancy_to_density(occup, home_range, gamma=gamma, omega=omega)

    # Velocity is a function of home_range and dt
    velocity = movement.home_range_to_velocity(home_range, beta=beta, dt=dt)
    dx = max(velocity, MIN_VELOCITY)
    num_sides = int(np.ceil(range / dx))

    movement_data = movement.make_data(
        velocity,
        num=num*n_trials,
        occupancy=occup,
        steps=season,
        gamma=gamma,
        delta=delta,
        alpha=alpha, range=range)

    grid = occupancy.make_grid(movement_data, num_trials=n_trials)
    proportions = np.sum(grid, axis=(1, 2)) / float(num_sides ** 2)
    return proportions


def _aux_occupancy(data, parameters=None):
    home_range, occup = data
    areas = calculate_occupancy_distribution(
        home_range, occup, parameters=parameters)
    return areas


def calculate_all_occupancy_distributions(species=None, parameters=None):
    """Calculates distribution of simulated occupancy for all species

    Species simulated are contained in the COMMON_SPECIES dictionary"""
    if species is None:
        species = COMMON_SPECIES.keys()

    data = [
        (
            COMMON_SPECIES[spec]['home_range'],
            COMMON_SPECIES[spec]['occupancy']
        )
        for spec in species]

    p_pool = Pool()
    dists = p_pool.map(
            partial(_aux_occupancy, parameters=parameters),
            data)
    p_pool.close()
    p_pool.join()

    return np.array(dists)
