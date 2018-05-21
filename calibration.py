"""
Calibration module for species simulator.

Calibration procedure looks for simulator parameters that minimize error with
respect to several sources:
    1. Home range information as calculated from the simulation and provided by
    biological data.
    2. Occupancy information provided by the article TODO.
"""
from __future__ import print_function

from functools import partial
from multiprocessing import Pool
import numpy as np  # pylint: disable=import-error

import movement
import home_range
import occupancy
from constants import *  # noqa: F403

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

    beta = parameters.get('beta', BETA)  # noqa: F405
    range_ = parameters.get('range', RANGE)  # noqa: F405
    power = parameters.get('power', POWER)  # noqa: F405
    dt = parameters.get('dt', DT)  # noqa: F405
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

    n_home_range = parameters.get('n_home_range', N_HOME_RANGE)  # noqa: F405
    n_occupancy = parameters.get('n_occupancy', N_OCCUPANCY)  # noqa: F405

    min_home_range = parameters.get('min_home_range', MIN_HOME_RANGE)  # noqa
    max_home_range = parameters.get('max_home_range', MAX_HOME_RANGE)  # noqa
    min_occupancy = parameters.get('min_occupancy', MIN_OCCUPANCY)  # noqa
    max_occupancy = parameters.get('max_occupancy', MAX_OCCUPANCY)  # noqa

    home_ranges = np.linspace(
            min_home_range, max_home_range, n_home_range)
    occupancies = np.linspace(
            min_occupancy, max_occupancy, n_occupancy)

    n_trials = parameters.get('n_trials', N_TRIALS)  # noqa: F405
    n_individuals = parameters.get('n_individuals', N_INDIVIDUALS)  # noqa

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

    return home_ranges, occupancies, fresults


def calculate_occupancy_distribution_0(hrange, occup, parameters=None):
    if parameters is None:
        parameters = {}

    # Movement parameters
    beta = parameters.get('beta', BETA)  # noqa
    range_ = parameters.get('range', RANGE)  # noqa
    dt = parameters.get('dt', DT)  # noqa
    power = parameters.get('power', POWER)  # noqa

    # Occupancy parameters
    season = parameters.get('season', SEASON)  # noqa
    gamma = parameters.get('gamma', GAMMA)  # noqa
    omega = parameters.get('omega', OMEGA)  # noqa

    # Statistical parameters
    n_trials = parameters.get('n_trials', 100)

    velocity = movement.home_range_to_velocity(
        hrange,
        beta=beta,
        power=power,
        dt=dt)

    num = int(occupancy.occupancy_to_density(
        occup,
        hrange,
        gamma=gamma,
        omega=omega) * range_**2)

    movement_data = movement.make_data(
        velocity,
        occup,
        num=n_trials * num,
        steps=season,
        range=range_)

    occupancy_data = occupancy.make_data(
        movement_data,
        num_trials=n_trials)

    return occupancy_data.occupations


def _aux_occ_0(args, parameters=None):
    (i, hrange), (j, occup), k = args
    # print('[+] Starting {}-th calculation of occupancy for {}-th home_range={}, {}-th occupancy={}'.format(k, i, hrange, j, occup))
    results = calculate_occupancy_distribution_0(
        hrange,
        occup,
        parameters=parameters)
    # print('[!!!] Done {}-th calculation of occupancy for {}-th home_range={}, {}-th occupancy={}'.format(k, i, hrange, j, occup))
    return results


def calculate_occupancy_distribution(hrange, occup, parameters=None):
    if parameters is None:
        parameters = {}

    n_worlds = parameters.get('n_worlds', 10)

    arguments = [hrange for _ in xrange(n_worlds)]

    p = Pool()
    results = p.map(
        partial(calculate_occupancy_distribution_0,
                occup=occup,
                parameters=parameters),
        arguments)
    p.close()
    p.join()

    return np.concatenate(results, 0)


def calculate_all_occupancy_distributions(parameters=None):
    """Calculates distribution of simulated home range for all species
    """
    if parameters is None:
        parameters = {}

    n_home_range = parameters.get('n_home_range', N_HOME_RANGE)  # noqa: F405
    n_occupancy = parameters.get('n_occupancy', N_OCCUPANCY)  # noqa: F405

    min_home_range = parameters.get('min_home_range', MIN_HOME_RANGE)  # noqa
    max_home_range = parameters.get('max_home_range', MAX_HOME_RANGE)  # noqa
    min_occupancy = parameters.get('min_occupancy', MIN_OCCUPANCY)  # noqa
    max_occupancy = parameters.get('max_occupancy', MAX_OCCUPANCY)  # noqa

    home_ranges = np.linspace(
            min_home_range, max_home_range, n_home_range)
    occupancies = np.linspace(
            min_occupancy, max_occupancy, n_occupancy)

    n_worlds = parameters.get('n_worlds', N_WORLDS)  # noqa: F405

    all_args = [
            (hr, oc, k)
            for hr in enumerate(home_ranges)
            for oc in enumerate(occupancies)
            for k in xrange(n_worlds)]

    p = Pool()
    results = p.map(partial(_aux_occ_0, parameters=parameters), all_args)
    p.close()
    p.join()

    return results, all_args
