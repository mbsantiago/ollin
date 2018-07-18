from __future__ import division

import numpy as np


def normalize(array):
    extent = array.max() - array.min()
    if extent == 0:
        normalized = 0.5 * np.ones_like(array)
    else:
        normalized = (array - array.min()) / extent
    return normalized


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(y):
    return -np.log((1/y) - 1)


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


def occupancy_to_density(
        occupancy,
        home_range_proportion,
        niche_size,
        parameters=None):
    alpha = parameters['alpha']
    hr_exp = parameters['hr_exp']
    den_exp = parameters['den_exp']
    nsz_exp = parameters['niche_size_exp']

    density = np.exp(
        (logit(occupancy) - alpha -
         np.log(home_range_proportion) * hr_exp -
         np.log(niche_size) * nsz_exp) / den_exp)
    return density


def density_to_occupancy(
        density,
        home_range_proportion,
        niche_size,
        parameters=None):
    alpha = parameters['alpha']
    hr_exp = parameters['hr_exp']
    den_exp = parameters['density_exp']
    nsz_exp = parameters['niche_size_exp']

    occupancy = sigmoid(
        alpha + np.log(density) * den_exp +
        np.log(home_range_proportion) * hr_exp +
        np.log(niche_size) * nsz_exp)
    return occupancy


def home_range_resolution(velocity, parameters=None):
    return velocity


def velocity_modification(niche_size, parameters):
    alpha = parameters['velocity']['alpha']
    beta = parameters['velocity']['beta']
    return beta + alpha * niche_size
