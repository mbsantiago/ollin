"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
# pylint: disable=unbalanced-tuple-unpacking
import numpy as np  # pylint: disable=import-error
from numba import jit, float64, int64
import math
from cycler import cycler  # pylint: disable=import-error

import initial_conditions
from constants import (RANGE, BETA, STEPS, DT, POWER)


@jit(
    float64[:, :, :](
        float64[:, :],
        float64[:, :],
        float64,
        int64,
        float64,
        float64,
        int64),
    nopython=True)
def _movement(
        heatmap,
        random_positions,
        resolution,
        num,
        velocity,
        range_,
        steps):
    movement = np.zeros((num, steps, 2), dtype=float64)
    random_angles = np.random.uniform(0.0, 2 * np.pi, size=(steps, num))

    for k in xrange(steps):
        movement[:, k, :] = random_positions
        for j in xrange(num):
            angle = random_angles[k, j]
            heading = (math.cos(angle), math.sin(angle))
            index = (
                random_positions[j, 0] // resolution,
                random_positions[j, 1] // resolution)
            value = heatmap[int(index[0]), int(index[1])]
            exponent = 1.1 + 0.9 * value
            magnitude = (velocity * (exponent - 1)) / \
                (math.pow((1 - np.random.rand()), 1/exponent) * exponent)
            direction = (magnitude * heading[0], magnitude * heading[1])
            tmp1 = (
                random_positions[j, 0] + direction[0],
                random_positions[j, 1] + direction[1])
            tmp2 = (tmp1[0] % (2 * range_), tmp1[1] % (2 * range_))

            if tmp2[0] < range_:
                random_positions[j, 0] = tmp2[0] % range_
            else:
                random_positions[j, 0] = (-tmp2[0]) % range_

            if tmp2[1] < range_:
                random_positions[j, 1] = tmp2[1] % range_
            else:
                random_positions[j, 1] = (-tmp2[1]) % range_
    return movement


class MovementData(object):
    def __init__(self, initial_data, steps=STEPS):
        self.initial_data = initial_data
        self.velocity = initial_data.velocity
        self.num = initial_data.num
        self.range = initial_data.range
        self.steps = steps

        self.data = self.make_data()

    def make_data(self):
        """Main function for movement data creation."""
        initial_data = self.initial_data
        steps = self.steps

        random_positions = initial_data.initial_points

        heatmap = initial_data.kde_approximation
        heatmap = heatmap / heatmap.max()
        resolution = initial_data.resolution

        num = initial_data.num
        velocity = initial_data.velocity
        range_ = initial_data.range

        mov = _movement(
            heatmap,
            random_positions,
            resolution,
            num,
            velocity,
            range_,
            steps)

        return mov

    def plot(
            self,
            include=None,
            num=10,
            steps=365,
            axis=None,
            cmap='Dark2',
            **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        if include is None:
            include = [
                'heatmap',
                'niche',
                'occupation_zone',
                'rectangle',
                'trajectories']

        if axis is None:
            fig, axis = plt.subplots()

        initial_conditions_options = [
            opt for opt in include
            if opt in initial_conditions.PLOT_OPTIONS]

        if len(initial_conditions_options) != 0:
            axis = self.initial_data.plot(
                include=initial_conditions_options, axis=axis, **kwargs)

        if 'trajectories' in include:

            cmap = plt.get_cmap(cmap)
            colors = [cmap(i) for i in np.linspace(0.05, .8, 10)]
            axis.set_prop_cycle(cycler('color', colors))

            num = min(self.num, num)
            steps = min(self.steps, steps)
            trajectories = self.data[:num, :steps, :]

            for trajectory in trajectories:
                xcoord, ycoord = zip(*trajectory)
                axis.plot(xcoord, ycoord)

        ticks = np.linspace(0, self.range, 2)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)

        return axis


def make_data_from_init_cond(initial_data, steps=STEPS):
    mov_data = MovementData(initial_data, steps=steps)
    return mov_data


def make_data(velocity, occupancy, num=100, steps=STEPS, range=RANGE):
    initial_data = initial_conditions.make_data(
        range, occupancy, num, velocity)
    mov_data = MovementData(initial_data, steps=steps)
    return mov_data


def home_range_to_velocity(home_range, beta=BETA, power=POWER, dt=DT):
    return beta * np.power(home_range, power) / float(dt)
