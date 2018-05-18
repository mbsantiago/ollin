"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
# pylint: disable=unbalanced-tuple-unpacking
import numpy as np  # pylint: disable=import-error
from cycler import cycler  # pylint: disable=import-error

import initial_conditions
from constants import (RANGE, BETA, STEPS, DT, POWER)


class MovementData(object):
    def __init__(self, velocity, occupancy, num=100, steps=STEPS, range=RANGE):
        self.velocity = velocity
        self.num = num
        self.steps = steps
        self.range = range

        self.initial_data = initial_conditions.make_data(
            range, occupancy, self.num, self.velocity)

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
        stack = [random_positions]
        for _ in xrange(steps - 1):
            random_angles = np.random.uniform(0, 2 * np.pi, [num])
            random_directions = np.stack(
                    [np.cos(random_angles), np.sin(random_angles)], axis=-1)

            indices = np.floor_divide(
                random_positions, resolution).astype(np.int)
            xindex, yindex = np.split(indices, 2, -1)
            values = heatmap[xindex, yindex].reshape([num])

            exponents = (1.1 + 0.9 * values)
            random_magnitudes = velocity * (exponents - 1) / (np.power(
                (1 - np.random.rand(num)), 1/exponents) * exponents)
            random_directions *= random_magnitudes[:, None]

            tmp1 = random_positions + random_directions
            tmp2 = np.mod(tmp1, 2 * range_)
            reflections = np.greater(tmp2, range_)
            tmp3 = (1 - reflections) * np.mod(tmp2, range_)
            tmp4 = reflections * np.mod(-tmp2, range_)
            random_positions = tmp3 + tmp4

            stack.append(random_positions)
        return np.stack(stack, 1)

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


def make_data(velocity, occupancy, num=100, steps=STEPS, range=RANGE):
    mov_data = MovementData(
        velocity, occupancy, num=num, steps=steps, range=range)
    return mov_data


def home_range_to_velocity(home_range, beta=BETA, power=POWER, dt=DT):
    return beta * np.power(home_range, power) / float(dt)
