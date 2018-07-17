import numpy as np
import math
from numba import jit, float64, int64

from .basemodel import MovementModel


class Model(MovementModel):
    name = 'Constant Levy Model'
    default_parameters = {
        'velocity': {
            'beta': 1.05,
            'alpha': 0.0},
        'home_range': {
            'alpha': 35.0,
            'exponent': 0.54},
        'density': {
            'alpha': 0.0,
            'beta': 22.0,
            'hr_exp': 1.2,
            'occ_exp_a': 1.75,
            'occ_exp_b': 1.75},
        'movement': {
            'pareto': 1.8},
    }

    def __init__(self, parameters):
        super(Model, self).__init__(parameters)

    def generate_movement(
            self,
            initial_positions,
            initial_conditions,
            days,
            velocity):
        exponent = self.parameters['movement']['pareto']
        steps_per_day = self.parameters['steps_per_day']
        range_ = initial_conditions.range
        velocity = velocity / steps_per_day
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
