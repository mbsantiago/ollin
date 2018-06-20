"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
# pylint: disable=unbalanced-tuple-unpacking
import math
import numpy as np  # pylint: disable=import-error
from cycler import cycler  # pylint: disable=import-error
from numba import jit, float64, int64

import initial_conditions
from constants import handle_parameters

from utils import density


@jit(
    float64[:, :, :](
        float64[:, :],
        float64[:, :],
        float64,
        int64,
        float64,
        float64[:],
        int64,
        int64),
    nopython=True)
def _movement(
        heatmap,
        random_positions,
        resolution,
        num,
        velocity,
        range_,
        days,
        steps_per_day):
    steps = days * steps_per_day
    movement = np.zeros((num, steps, 2), dtype=float64)
    random_angles = np.random.uniform(0.0, 2 * np.pi, size=(steps, num))
    velocity = velocity / float(steps_per_day)
    rangex, rangey = range_

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
            tmp2 = (tmp1[0] % (2 * rangex), tmp1[1] % (2 * rangey))

            if tmp2[0] < rangex:
                random_positions[j, 0] = tmp2[0] % rangex
            else:
                random_positions[j, 0] = (-tmp2[0]) % rangex

            if tmp2[1] < rangey:
                random_positions[j, 1] = tmp2[1] % rangey
            else:
                random_positions[j, 1] = (-tmp2[1]) % rangey
    return movement


class MovementData(object):
    def __init__(self, initial_data, num=None, days=None, parameters=None):
        if parameters is None:
            parameters = initial_data.parameters
        else:
            parameters = handle_parameters(parameters)
        self.parameters = parameters

        self.initial_data = initial_data
        self.occupancy = initial_data.occupancy
        self.home_range = initial_data.home_range
        self.velocity = initial_data.velocity
        self.range = initial_data.range

        if num is None:
            dens = density(
                self.occupancy, self.home_range, parameters=self.parameters)
            num = int(self.range[0] * self.range[1] * dens)
        self.num = num

        if days is None:
            days = parameters['DAYS']
        self.days = days

        self.steps_per_day = parameters['STEPS_PER_DAY']
        self.steps = days * self.steps_per_day

        self.initial_positions = initial_data.sample(num)
        self.data = self.make_data()

    def make_data(self):
        """Main function for movement data creation."""
        initial_data = self.initial_data
        days = self.days
        steps_per_day = self.steps_per_day

        random_positions = self.initial_positions

        heatmap = initial_data.kde_approximation
        heatmap = heatmap / heatmap.max()
        resolution = initial_data.home_range_resolution

        num = self.num
        velocity = initial_data.velocity
        range_ = initial_data.range

        mov = _movement(
            heatmap,
            random_positions,
            resolution,
            num,
            velocity,
            range_,
            days,
            steps_per_day)

        return mov

    def plot(
            self,
            include=None,
            num=10,
            days=365,
            ax=None,
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

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        initial_conditions_options = [
            opt for opt in include
            if opt in initial_conditions.PLOT_OPTIONS]

        if len(initial_conditions_options) != 0:
            ax = self.initial_data.plot(
                include=initial_conditions_options, ax=ax, **kwargs)

        if 'trajectories' in include:

            cmap = plt.get_cmap(cmap)
            colors = [cmap(i) for i in np.linspace(0.05, .8, 10)]
            ax.set_prop_cycle(cycler('color', colors))

            num = min(self.num, num)
            steps = min(self.steps, days * self.steps_per_day)
            trajectories = self.data[:num, :steps, :]

            for trajectory in trajectories:
                xcoord, ycoord = zip(*trajectory)
                ax.plot(xcoord, ycoord)

        xticks = np.linspace(0, self.range[0], 2)
        yticks = np.linspace(0, self.range[1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        return ax
