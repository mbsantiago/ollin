"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
from __future__ import division
# pylint: disable=unbalanced-tuple-unpacking
from abc import abstractmethod
import math
import numpy as np  # pylint: disable=import-error
from cycler import cycler  # pylint: disable=import-error
from numba import jit, float64, int64

import initial_conditions
from constants import handle_movement_parameters

from utils import density, home_range_to_velocity


def normalize(array):
    extent = array.max() - array.min()
    if extent == 0:
        normalized = 0.5 * np.ones_like(array)
    else:
        normalized = (array - array.min()) / extent
    return normalized


class MovementModel(object):
    name = None

    def __init__(self, parameters):
        self.parameters = handle_movement_parameters(parameters, self.name)

    @abstractmethod
    def generate_movement(self, initial_position, initial_conditions):
        pass


class ConstantLevyMovement(MovementModel):
    name = 'Constant Levy Model'

    def __init__(self, parameters):
        super(ConstantLevyMovement, self).__init__(parameters)

    def generate_movement(self, initial_positions, initial_conditions, days):
        exponent = self.parameters['movement']['pareto']
        steps_per_day = self.parameters['global']['steps_per_day']

        base_vel = home_range_to_velocity(
            initial_conditions.home_range, parameters=self.parameters)
        velocity = base_vel / steps_per_day
        range_ = initial_conditions.range

        steps = days * steps_per_day
        mov = self._movement(
            initial_positions,
            velocity,
            range_,
            steps,
            exponent)
        return mov

    @staticmethod
    @jit(
        float64[:, :, :](
            float64[:, :],
            float64,
            float64[:],
            int64,
            float64),
        nopython=True)
    def _movement(
            random_positions,
            velocity,
            range_,
            steps,
            exponent):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        random_angles = np.random.uniform(0.0, 2 * np.pi, size=(steps, num))
        rangex, rangey = range_
        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                angle = random_angles[k, j]
                heading = (math.cos(angle), math.sin(angle))
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


class VariableLevyMovement(MovementModel):
    name = 'Variable Levy Model'

    def __init__(self, parameters=None):
        super(VariableLevyMovement, self).__init__(parameters)

    def generate_movement(self, initial_positions, initial_conditions, days):
        min_exponent = self.parameters['movement']['min_pareto']
        max_exponent = self.parameters['movement']['max_pareto']
        steps_per_day = self.parameters['global']['steps_per_day']

        base_vel = home_range_to_velocity(
                initial_conditions.home_range, parameters=self.parameters)
        range_ = initial_conditions.range
        heatmap = normalize(initial_conditions.kde_approximation)
        resolution = initial_conditions.home_range_resolution

        velocity = base_vel / steps_per_day
        steps = days * steps_per_day

        mov = self._movement(
            heatmap,
            initial_positions,
            resolution,
            velocity,
            range_,
            steps,
            min_exponent,
            max_exponent)
        return mov

    @staticmethod
    @jit(
        float64[:, :, :](
            float64[:, :],
            float64[:, :],
            float64,
            float64,
            float64[:],
            int64,
            float64,
            float64),
        nopython=True)
    def _movement(
            heatmap,
            random_positions,
            resolution,
            velocity,
            range_,
            steps,
            min_exponent,
            max_exponent):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        random_angles = np.random.uniform(0.0, 2 * np.pi, size=(steps, num))
        rangex, rangey = range_
        exponent_var = max_exponent - min_exponent

        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                angle = random_angles[k, j]
                heading = (math.cos(angle), math.sin(angle))
                index = (
                    random_positions[j, 0] // resolution,
                    random_positions[j, 1] // resolution)
                value = heatmap[int(index[0]), int(index[1])]
                exponent = min_exponent + exponent_var * value
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


class ConstantBrownianMovement(MovementModel):
    name = 'Constant Brownian Model'

    def __init__(self, parameters=None):
        super(ConstantBrownianMovement, self).__init__(parameters)

    def generate_movement(self, initial_positions, initial_conditions, days):
        steps_per_day = self.parameters['global']['steps_per_day']

        base_vel = home_range_to_velocity(
                initial_conditions.home_range, parameters=self.parameters)
        range_ = initial_conditions.range

        velocity = base_vel / steps_per_day
        steps = days * steps_per_day
        mov = self._movement(
            initial_positions,
            velocity,
            range_,
            steps)
        return mov

    @staticmethod
    @jit(
        float64[:, :, :](
            float64[:, :],
            float64,
            float64[:],
            int64),
        nopython=True)
    def _movement(
            random_positions,
            velocity,
            range_,
            steps):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        sigma = velocity / 1.2533141373155003
        rangex, rangey = range_
        directions = np.random.normal(
            0, sigma, size=(steps, num, 2))

        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                direction = directions[k, j]
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


class VariableBrownianMovement(MovementModel):
    name = 'Variable Brownian Model'

    def __init__(self, parameters=None):
        super(VariableBrownianMovement, self).__init__(parameters)

    def generate_movement(self, initial_positions, initial_conditions, days):
        steps_per_day = self.parameters['global']['steps_per_day']
        niche_weight = self.parameters['movement']['niche_weight']

        heatmap = normalize(initial_conditions.kde_approximation)
        resolution = initial_conditions.home_range_resolution
        base_vel = home_range_to_velocity(
                initial_conditions.home_range, parameters=self.parameters)
        range_ = initial_conditions.range

        velocity = base_vel / steps_per_day
        steps = days * steps_per_day
        mov = self._movement(
            heatmap,
            initial_positions,
            resolution,
            velocity,
            range_,
            steps,
            niche_weight)
        return mov

    @staticmethod
    @jit(
            float64[:, :, :](
                float64[:, :],
                float64[:, :],
                float64,
                float64,
                float64[:],
                int64,
                float64),
            nopython=True)
    def _movement(
            heatmap,
            random_positions,
            resolution,
            velocity,
            range_,
            steps,
            niche_weight):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        sigma = velocity / 1.2533141373155003
        rangex, rangey = range_
        directions = np.random.normal(
                0, sigma, size=(steps, num, 2))

        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                direction = directions[k, j]
                index = (
                        random_positions[j, 0] // resolution,
                        random_positions[j, 1] // resolution)
                value = heatmap[int(index[0]), int(index[1])]
                direction *= 1 + niche_weight * (1 / (2 * value + 0.1))
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


class GradientLevyMovement(MovementModel):
    name = 'Gradient Levy Model'

    def __init__(self, parameters):
        super(GradientLevyMovement, self).__init__(parameters)

    def generate_movement(self, initial_positions, initial_conditions, days):
        steps_per_day = self.parameters['global']['steps_per_day']
        min_exponent = self.parameters['movement']['min_pareto']
        max_exponent = self.parameters['movement']['max_pareto']
        grad_weight = self.parameters['movement']['grad_weight']

        heatmap = normalize(initial_conditions.kde_approximation)
        gradient = np.stack(np.gradient(heatmap), -1)
        resolution = initial_conditions.home_range_resolution
        range_ = initial_conditions.range
        base_vel = home_range_to_velocity(
                initial_conditions.home_range, parameters=self.parameters)

        velocity = base_vel / steps_per_day
        steps = days * steps_per_day

        mov = self._movement(
            gradient,
            heatmap,
            initial_positions,
            resolution,
            velocity,
            range_,
            steps,
            min_exponent,
            max_exponent,
            grad_weight)
        return mov

    @staticmethod
    @jit(
        float64[:, :, :](
            float64[:, :, :],
            float64[:, :],
            float64[:, :],
            float64,
            float64,
            float64[:],
            int64,
            float64,
            float64,
            float64),
        nopython=True)
    def _movement(
            gradient,
            heatmap,
            random_positions,
            resolution,
            velocity,
            range_,
            steps,
            min_exponent,
            max_exponent,
            grad_weight):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        rangex, rangey = range_
        directions = np.exp(1j * np.random.uniform(
                0, 2 * np.pi, size=(steps, num)))
        directions = np.stack((directions.real, directions.imag), -1)
        magnitudes = np.random.random(
                (steps, num))
        exponent_var = max_exponent - min_exponent

        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                direction = directions[k, j]
                magnitude = magnitudes[k, j]
                index = (
                        random_positions[j, 0] // resolution,
                        random_positions[j, 1] // resolution)
                grad = gradient[int(index[0]), int(index[1])]
                value = heatmap[int(index[0]), int(index[1])]

                exponent = min_exponent + exponent_var * value
                magnitude = (velocity * (exponent - 1)) / \
                    (math.pow((1 - magnitude), 1/exponent) * exponent)
                direction = magnitude * (grad_weight * grad + direction)

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


class GradientBrownianMovement(MovementModel):
    name = 'Gradient Brownian Model'

    def __init__(self, parameters):
        super(GradientBrownianMovement, self).__init__(parameters)

    def generate_movement(self, initial_positions, initial_conditions, days):
        steps_per_day = self.parameters['global']['steps_per_day']
        grad_weight = self.parameters['movement']['grad_weight']

        heatmap = normalize(initial_conditions.kde_approximation)
        gradient = np.stack(np.gradient(heatmap), -1)
        resolution = initial_conditions.home_range_resolution
        range_ = initial_conditions.range
        base_vel = home_range_to_velocity(
                initial_conditions.home_range, parameters=self.parameters)

        velocity = base_vel / steps_per_day
        steps = days * steps_per_day

        mov = self._movement(
                gradient,
                heatmap,
                initial_positions,
                resolution,
                velocity,
                range_,
                steps,
                grad_weight)
        return mov

    @staticmethod
    @jit(
        float64[:, :, :](
            float64[:, :, :],
            float64[:, :],
            float64[:, :],
            float64,
            float64,
            float64[:],
            int64,
            float64),
        nopython=True)
    def _movement(
            gradient,
            heatmap,
            random_positions,
            resolution,
            velocity,
            range_,
            steps,
            grad_weight):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        rangex, rangey = range_
        directions = np.random.normal(0, 1, (num, steps, 2))

        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                direction = directions[j, k]
                index = (
                        random_positions[j, 0] // resolution,
                        random_positions[j, 1] // resolution)
                grad = gradient[int(index[0]), int(index[1])]
                value = heatmap[int(index[0]), int(index[1])]

                new_direction = velocity * (
                    (1 - value) * grad_weight * grad + value * direction)

                tmp1 = (
                        random_positions[j, 0] + new_direction[0],
                        random_positions[j, 1] + new_direction[1])
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


def make_movement_model(model, parameters):
    if model == 'variable_levy':
        return VariableLevyMovement(parameters)
    elif model == 'constant_levy':
        return ConstantLevyMovement(parameters)
    elif model == 'constant_brownian':
        return ConstantBrownianMovement(parameters)
    elif model == 'variable_brownian':
        return VariableBrownianMovement(parameters)
    elif model == 'gradient_levy':
        return GradientLevyMovement(parameters)
    elif model == 'gradient_brownian':
        return GradientBrownianMovement(parameters)
    else:
        msg = 'Movement model {} not implemented'.format(model)
        raise NotImplementedError(msg)


class MovementData(object):
    def __init__(
            self,
            initial_conditions,
            movement_data,
            movement_model):

        self.initial_conditions = initial_conditions
        self.movement_model = movement_model
        self.data = movement_data

        self.num_experiments, self.num, self.steps, _ = movement_data.shape
        steps_per_day = movement_model.parameters['global']['steps_per_day']
        self.days = self.steps / steps_per_day

    @classmethod
    def simulate(
            cls,
            initial_conditions,
            days=None,
            num=None,
            num_experiments=1,
            parameters=None,
            movement_model='variable_levy'):

        movement_model = make_movement_model(movement_model, parameters)
        params = movement_model.parameters

        if days is None:
            days = params['global']['days']

        if num is None:
            occupancy = initial_conditions.occupancy
            home_range = initial_conditions.home_range
            rangex, rangey = initial_conditions.range
            dens = density(
                    occupancy, home_range, parameters=params['density'])
            num = int(rangex * rangey * dens)

        num_ = num * num_experiments
        steps_per_day = params['global']['steps_per_day']
        steps = days * steps_per_day
        initial_positions = initial_conditions.sample(num_)
        movement_data = movement_model.generate_movement(
            initial_positions, initial_conditions, days).reshape(
                [num_experiments, num, steps, 2])

        return cls(
            initial_conditions,
            movement_data,
            movement_model)

    def plot(
            self,
            include=None,
            num=10,
            days=365,
            ax=None,
            mov_cmap='Greens',
            simplify=None,
            experiment_number=0,
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

        self.initial_conditions.plot(
            include=include, ax=ax, **kwargs)

        if 'trajectories' in include:

            cmap = plt.get_cmap(mov_cmap)
            colors = [cmap(i) for i in np.linspace(0.05, .8, 10)]
            ax.set_prop_cycle(cycler('color', colors))

            steps_per_day = self.movement_model.steps_per_day
            num = min(self.num, num)
            steps = min(self.steps, days * steps_per_day)

            if simplify is None:
                stride = 1
            else:
                stride = max(int(steps / simplify), 1)
            experiment_number = max(
                min(experiment_number, self.num_experiments - 1), 0)
            trajectories = self.data[experiment_number, :num, :steps:stride, :]

            for trajectory in trajectories:
                xcoord, ycoord = zip(*trajectory)
                ax.plot(xcoord, ycoord)

        range = self.initial_conditions.range
        xticks = np.linspace(0, range[0], 2)
        yticks = np.linspace(0, range[1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        return ax


class MovementAnalyzer(object):
    def __init__(self, mov):
        self.movement_data = mov
        self.velocities, self.bearings, self.turn_angles = self.analyze()

    @property
    def mean_velocity(self):
        return self.velocities.mean(axis=1).mean()

    def analyze(self):
        mov = self.movement_data.movement_model
        steps_per_day = mov.parameters['global']['steps_per_day']
        num_experiments, num, steps, _ = self.movement_data.data.shape
        data = self.movement_data.data.reshape([-1, steps, 2])

        directions = (data[:, 1:, :] - data[:, :-1, :])
        complex_directions = directions[:, :, 0] + 1j * directions[:, :, 1]
        velocities = steps_per_day * np.abs(complex_directions)
        bearings = np.angle(complex_directions)
        turn_angles = np.angle(complex_directions[:, 1:] /
                               complex_directions[:, :-1])
        return velocities, bearings, turn_angles

    def plot_velocities(self, num_individual=0, ax=None, bins=20):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        if num_individual == 'all':
            velocities = self.velocities.ravel()
        else:
            velocities = self.velocities[num_individual]

        plt.hist(velocities, bins=bins)
