from __future__ import division

import numpy as np


def normalize(array):
    extent = array.max() - array.min()
    if extent == 0:
        normalized = 0.5 * np.ones_like(array)
    else:
        normalized = (array - array.min()) / extent
    return normalized


def occupancy_resolution(home_range, parameters=None):
    return np.sqrt(home_range)


def home_range_to_velocity(home_range, parameters=None):
    exponent = parameters['home_range']['exponent']
    alpha = parameters['home_range']['alpha']
    return alpha * np.power(home_range, exponent)


def velocity_to_home_range(velocity, parameters=None):
    exponent = parameters['home_range']['exponent']
    alpha = parameters['home_range']['alpha']
    return np.power(velocity / alpha, 1 / exponent)


def density(occupancy, home_range, niche_size, parameters=None):
    alpha = parameters['alpha']
    beta = parameters['beta']
    hr_exp = parameters['hr_exp']
    occ_exp_a = parameters['occ_exp_a']
    occ_exp_b = parameters['occ_exp_b']

    occ_exp = occ_exp_a * niche_size + occ_exp_b
    prop = np.exp(alpha * niche_size + beta)

    density = prop * (occupancy**occ_exp) / (home_range)**hr_exp
    return density


def density_to_occupancy(density, home_range, niche_size, parameters=None):
    alpha = parameters['alpha']
    beta = parameters['beta']
    hr_exp = parameters['hr_exp']
    occ_exp_a = parameters['occ_exp_a']
    occ_exp_b = parameters['occ_exp_b']

    occ_exp = occ_exp_a * niche_size + occ_exp_b
    prop = np.exp(alpha * niche_size + beta)

    occupancy = np.power(density * home_range**hr_exp / prop, 1 / occ_exp)
    return occupancy


def home_range_resolution(velocity, parameters=None):
    return velocity


def velocity_modification(niche_size, parameters):
    alpha = parameters['velocity']['alpha']
    beta = parameters['velocity']['beta']
    return beta + alpha * niche_size
