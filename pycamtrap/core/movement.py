"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
# pylint: disable=unbalanced-tuple-unpacking
from abc import abstractmethod
import math
import numpy as np  # pylint: disable=import-error
from cycler import cycler  # pylint: disable=import-error
from numba import jit, float64, int64

import initial_conditions
from constants import handle_parameters

from utils import density


def normalize(array):
    extent = array.max() - array.min()
    if extent == 0:
        normalized = 0.5 * np.ones_like(array)
    else:
        normalized = (array - array.min()) / extent
    return normalized


class MovementModel(object):
    def __init__(self):
        pass

    @abstractmethod
    def generate_movement(self, initial_position, initial_conditions):
        pass


class ConstantLevyMovement(MovementModel):
    name = 'Constant Levy Model'

    def __init__(self, parameters=None):
        super(ConstantLevyMovement, self).__init__()

        if parameters is None:
            parameters = {}
        parameters = handle_parameters(parameters)

        self.alpha = parameters['ALPHA']
        self.steps_per_day = parameters['STEPS_PER_DAY']

    def generate_movement(self, initial_positions, initial_conditions, days):
        velocity = initial_conditions.velocity / float(self.steps_per_day)
        num = len(initial_positions)
        range_ = initial_conditions.range
        alpha = self.alpha
        steps = days * self.steps_per_day
        mov = self._movement(
            initial_positions,
            num,
            velocity,
            range_,
            steps,
            alpha)
        return mov

    @staticmethod
    @jit(
        float64[:, :, :](
            float64[:, :],
            int64,
            float64,
            float64[:],
            int64,
            float64),
        nopython=True)
    def _movement(
            random_positions,
            num,
            velocity,
            range_,
            steps,
            alpha):
        movement = np.zeros((num, steps, 2), dtype=float64)
        random_angles = np.random.uniform(0.0, 2 * np.pi, size=(steps, num))
        rangex, rangey = range_
        exponent = alpha
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
        super(VariableLevyMovement, self).__init__()

        if parameters is None:
            parameters = {}
        parameters = handle_parameters(parameters)

        self.min_alpha = parameters['MIN_ALPHA']
        self.max_alpha = parameters['MAX_ALPHA']
        self.steps_per_day = parameters['STEPS_PER_DAY']

    def generate_movement(self, initial_positions, initial_conditions, days):
        heatmap = normalize(initial_conditions.kde_approximation)
        resolution = initial_conditions.home_range_resolution
        num = len(initial_positions)
        velocity = initial_conditions.velocity / float(self.steps_per_day)
        range_ = initial_conditions.range
        steps = days * self.steps_per_day
        min_alpha = self.min_alpha
        max_alpha = self.max_alpha
        mov = self._movement(
            heatmap,
            initial_positions,
            resolution,
            num,
            velocity,
            range_,
            steps,
            min_alpha,
            max_alpha)
        return mov

    @staticmethod
    @jit(
        float64[:, :, :](
            float64[:, :],
            float64[:, :],
            float64,
            int64,
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
            num,
            velocity,
            range_,
            steps,
            min_alpha,
            max_alpha):
        movement = np.zeros((num, steps, 2), dtype=float64)
        random_angles = np.random.uniform(0.0, 2 * np.pi, size=(steps, num))
        rangex, rangey = range_
        alpha_var = max_alpha - min_alpha

        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                angle = random_angles[k, j]
                heading = (math.cos(angle), math.sin(angle))
                index = (
                    random_positions[j, 0] // resolution,
                    random_positions[j, 1] // resolution)
                value = heatmap[int(index[0]), int(index[1])]
                exponent = min_alpha + alpha_var * value
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
        super(ConstantBrownianMovement, self).__init__()

        parameters = handle_parameters(parameters)
        self.steps_per_day = parameters['STEPS_PER_DAY']

    def generate_movement(self, initial_positions, initial_conditions, days):
        num = len(initial_positions)
        velocity = initial_conditions.velocity / float(self.steps_per_day)
        range_ = initial_conditions.range
        steps = days * self.steps_per_day
        mov = self._movement(
            initial_positions,
            num,
            velocity,
            range_,
            steps)
        return mov

    @staticmethod
    @jit(
            float64[:, :, :](
                float64[:, :],
                int64,
                float64,
                float64[:],
                int64),
            nopython=True)
    def _movement(
            random_positions,
            num,
            velocity,
            range_,
            steps):
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
        super(VariableBrownianMovement, self).__init__()
        parameters = handle_parameters(parameters)

        self.dev = parameters['DEV']
        self.steps_per_day = parameters['STEPS_PER_DAY']

    def generate_movement(self, initial_positions, initial_conditions, days):
        heatmap = normalize(initial_conditions.kde_approximation)
        resolution = initial_conditions.home_range_resolution
        num = len(initial_positions)
        velocity = initial_conditions.velocity / float(self.steps_per_day)
        range_ = initial_conditions.range
        steps = days * self.steps_per_day
        mov = self._movement(
            heatmap,
            initial_positions,
            resolution,
            num,
            velocity,
            range_,
            steps,
            self.dev)
        return mov

    @staticmethod
    @jit(
            float64[:, :, :](
                float64[:, :],
                float64[:, :],
                float64,
                int64,
                float64,
                float64[:],
                int64,
                float64),
            nopython=True)
    def _movement(
            heatmap,
            random_positions,
            resolution,
            num,
            velocity,
            range_,
            steps,
            dev):
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
                direction *= 1 + dev * (1 / (2 * value + 0.1))
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
        super(GradientLevyMovement, self).__init__()
        parameters = handle_parameters(parameters)

        self.steps_per_day = parameters['STEPS_PER_DAY']
        self.nu = parameters.get('NU', 1)
        self.min_alpha = parameters['MIN_ALPHA']
        self.max_alpha = parameters['MAX_ALPHA']

    def generate_movement(self, initial_positions, initial_conditions, days):
        heatmap = normalize(initial_conditions.kde_approximation)
        gradient = np.stack(np.gradient(heatmap), -1)
        resolution = initial_conditions.home_range_resolution
        velocity = initial_conditions.velocity / float(self.steps_per_day)
        range_ = initial_conditions.range
        steps = days * self.steps_per_day

        mov = self._movement(
            gradient,
            heatmap,
            initial_positions,
            resolution,
            velocity,
            range_,
            steps,
            self.min_alpha,
            self.max_alpha,
            self.nu)
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
            min_alpha,
            max_alpha,
            nu):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        rangex, rangey = range_
        directions = np.exp(1j * np.random.uniform(
                0, 2 * np.pi, size=(steps, num)))
        directions = np.stack((directions.real, directions.imag), -1)
        magnitudes = np.random.random(
                (steps, num))
        alpha_var = max_alpha - min_alpha

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

                exponent = min_alpha + alpha_var * value
                magnitude = (velocity * (exponent - 1)) / \
                    (math.pow((1 - magnitude), 1/exponent) * exponent)
                direction = magnitude * (nu * grad + direction)

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
        super(GradientBrownianMovement, self).__init__()
        parameters = handle_parameters(parameters)

        self.steps_per_day = parameters['STEPS_PER_DAY']
        self.nu = parameters.get('NU', 1.0)
        self.GBM_alpha = parameters.get('GBM_alpha', 1.0)

    def generate_movement(self, initial_positions, initial_conditions, days):
        heatmap = normalize(initial_conditions.kde_approximation)
        gradient = np.stack(np.gradient(heatmap), -1)
        resolution = initial_conditions.home_range_resolution
        velocity = (self.GBM_alpha * initial_conditions.velocity /
                    float(self.steps_per_day))
        range_ = initial_conditions.range
        steps = days * self.steps_per_day

        mov = self._movement(
                gradient,
                heatmap,
                initial_positions,
                resolution,
                velocity,
                range_,
                steps,
                self.nu)
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
            nu):
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
                    (1 - value) * nu * grad + value * direction)

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
            movement_model,
            parameters):

        if parameters is None:
            parameters = initial_conditions.parameters
        else:
            parameters = handle_parameters(parameters)
        self.parameters = parameters

        self.initial_conditions = initial_conditions
        self.velocity = initial_conditions.velocity
        self.movement_model = movement_model
        self.data = movement_data

        self.num_experiments, self.num, self.steps, _ = movement_data.shape
        self.days = self.steps / parameters['STEPS_PER_DAY']

    @classmethod
    def simulate(
            cls,
            initial_conditions,
            days=None,
            num=None,
            num_experiments=1,
            parameters=None,
            movement_model='variable_levy'):
        parameters = handle_parameters(parameters)

        if days is None:
            days = parameters['DAYS']

        if num is None:
            occupancy = initial_conditions.occupancy
            home_range = initial_conditions.home_range
            rangex, rangey = initial_conditions.range
            dens = density(
                    occupancy, home_range, parameters=parameters)
            num = int(rangex * rangey * dens)

        movement_model = make_movement_model(movement_model, parameters)

        num_ = num * num_experiments
        steps = days * movement_model.steps_per_day
        initial_positions = initial_conditions.sample(num_)
        movement_data = movement_model.generate_movement(
            initial_positions, initial_conditions, days).reshape(
                [num_experiments, num, steps, 2])

        return cls(
            initial_conditions,
            movement_data,
            movement_model,
            parameters)

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

        initial_conditions_options = [
            opt for opt in include
            if opt in initial_conditions.PLOT_OPTIONS]

        if len(initial_conditions_options) != 0:
            ax = self.initial_conditions.plot(
                include=initial_conditions_options, ax=ax, **kwargs)

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
