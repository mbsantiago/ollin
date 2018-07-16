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


def density(occupancy, home_range, parameters=None):
    alpha = parameters['alpha']
    hr_exp = parameters['hr_exp']
    occ_exp = parameters['occ_exp']
    density = alpha * (occupancy**occ_exp) / (home_range)**hr_exp
    return density


def home_range_resolution(velocity, parameters=None):
    return velocity


def velocity_modification(niche_size, parameters):
    alpha = parameters['velocity']['alpha']
    beta = parameters['velocity']['beta']
    return beta + alpha * niche_size
