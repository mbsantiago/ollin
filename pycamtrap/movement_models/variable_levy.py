import math
import numpy as np
from numba import jit, float64, int64

from .basemodel import MovementModel
from ..core.utils import normalize


class Model(MovementModel):
    name = 'Variable Levy Model'
    default_parameters = {
        'velocity_mod': 6.0,
        'velocity': {
            'alpha': 35.0,
            'exponent': 0.54},
        'density': {
            'alpha': 22.0,
            'hr_exp': 1.2,
            'occ_exp': 1.75},
        'movement': {
            'min_pareto': 1.1,
            'max_pareto': 1.9},
    }

    def __init__(self, parameters=None):
        super(Model, self).__init__(parameters)

    def generate_movement(
            self,
            initial_positions,
            initial_conditions,
            days,
            velocity):
        min_exponent = self.parameters['movement']['min_pareto']
        max_exponent = self.parameters['movement']['max_pareto']
        heatmap = normalize(initial_conditions.kde_approximation)
        resolution = initial_conditions.resolution
        steps_per_day = self.parameters['steps_per_day']
        range_ = initial_conditions.range
        velocity = velocity / steps_per_day
        steps = days * steps_per_day
        steps_per_day = self.parameters['steps_per_day']
        range_ = initial_conditions.range
        velocity = velocity / steps_per_day
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
