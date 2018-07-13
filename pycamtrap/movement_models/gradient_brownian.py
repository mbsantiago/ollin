import numpy as np
from numba import jit, float64, int64

from pycamtrap.core.utils import normalize
from .basemodel import MovementModel


class Model(MovementModel):
    name = 'Gradient Brownian Model'
    default_parameters = {
        'velocity': {
            'alpha': 35.0,
            'exponent': 0.54},
        'density': {
            'alpha': 22.0,
            'hr_exp': 1.2,
            'occ_exp': 1.75},
        'movement': {
            'grad_weight': 0.5},
    }

    def __init__(self, parameters):
        super(Model, self).__init__(parameters)

    def generate_movement(
            self,
            initial_positions,
            initial_conditions,
            days,
            velocity):
        grad_weight = self.parameters['movement']['grad_weight']
        heatmap = normalize(initial_conditions.kde_approximation)
        gradient = np.stack(np.gradient(heatmap), -1)
        resolution = initial_conditions.resolution
        steps_per_day = self.parameters['steps_per_day']
        range_ = initial_conditions.range
        velocity = velocity / steps_per_day
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
