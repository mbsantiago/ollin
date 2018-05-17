"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
import numpy as np  # pylint: disable=import-error
from scipy.stats import gaussian_kde  # pylint: disable=import-error
from tqdm import tqdm  # pylint: disable=import-error


from initial_conditions import InitialCondition, PLOT_OPTIONS
from constants import (MAX_INDIVIDUALS, RANGE, ALPHA, BETA,
                       STEPS, DT, MIN_VELOCITY, MAX_POINTS,
                       GAMMA, DELTA)


class MovementData(object):
    def __init__(self, velocity, occupancy, num=100, steps=3650, range=RANGE):
        self.velocity = velocity
        self.num = num
        self.steps = steps
        self.range = range

        self.initial_data = InitialCondition(
            range, occupancy, self.num, self.velocity)

        self.data = self.make_data()

    def make_data(self):
        return make_data(self.initial_data, self.steps)

    def plot(self, include=None, num=10, steps=365, axis=None):
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
            if opt in PLOT_OPTIONS]

        if len(initial_conditions_options) != 0:
            axis = self.initial_data.plot(
                include=initial_conditions_options, axis=axis)

        if 'trajectories' in include:
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


def make_data(initial_data, steps):
    """Main function for movement data creation."""

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

        indices = np.floor_divide(random_positions, resolution).astype(np.int)
        xindex, yindex = np.split(indices, 2, -1)
        values = heatmap[xindex, yindex].reshape([num])

        exponents = (1.05 + 0.95*values)
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


def home_range_to_velocity(home_range, beta=BETA, dt=DT):
    return beta * np.sqrt(home_range) / float(dt)
