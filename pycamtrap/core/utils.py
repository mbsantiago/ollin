import numpy as np


from constants import *


def home_range_to_resolution(home_range):
    return np.sqrt(home_range)


def home_range_to_velocity(home_range, beta=BETA, power=POWER, dt=DT, steps_per_day=STEPS_PER_DAY):
    return beta * np.power(home_range, power) / float(dt)


def velocity_to_home_range(velocity, beta=BETA, power=POWER, dt=DT, steps_per_day=STEPS_PER_DAY):
    return np.power(velocity * dt / beta, 1/power)


def occupancy_to_num(occupancy, home_range, gamma=GAMMA, omega=OMEGA, range=RANGE, tau=TAU):
    num = gamma * (occupancy**tau) * range**2 / (home_range)**omega
    return int(num)


def velocity_to_resolution(velocity):
    return velocity
