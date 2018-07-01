import numpy as np


def occupancy_resolution(home_range, parameters=None):
    return np.sqrt(home_range)


def home_range_to_velocity(home_range, parameters=None):
    beta = parameters['BETA']
    power = parameters['POWER']
    dt = parameters['DT']
    return beta * np.power(home_range, power) / float(dt)


def velocity_to_home_range(velocity, parameters=None):
    dt = parameters['DT']
    power = parameters['POWER']
    beta = parameters['BETA']
    return np.power(velocity * dt / beta, 1/power)


def density(occupancy, home_range, parameters=None):
    gamma = parameters['GAMMA']
    omega = parameters['OMEGA']
    tau = parameters['TAU']
    density = gamma * (occupancy**tau) / (home_range)**omega
    return density


def home_range_resolution(velocity, parameters=None):
    return velocity
