from six.moves import xrange
import numpy as np
import math
from numba import jit, float64, int64

from .basemodel import MovementModel


class Model(MovementModel):
    name = 'Gradient Levy Model'
    default_parameters = {
        "density": {
            "alpha": 5.5687635749135715,
            "density_exp": 0.9094048898627347,
            "hr_exp": 0.9442292138564827,
            "niche_size_exp": 0.5609397724240494},
        "home_range": {
            "alpha": 102.95793964057235,
            "exponent": 1.8355253220927104},
        "velocity": {
            "alpha": -1.2919142788970375,
            "beta": 2.3185432601405385},
        'movement': {
            'min_pareto': 1.1,
            'max_pareto': 1.9,
            'grad_weight': 8.0},
    }

    def generate_movement(
            self,
            initial_positions,
            site,
            steps,
            velocity):
        min_exponent = self.parameters['movement']['min_pareto']
        max_exponent = self.parameters['movement']['max_pareto']
        grad_weight = self.parameters['movement']['grad_weight']

        heatmap = site.niche
        range_ = site.range
        resolution = site.resolution

        gradient = np.stack(np.gradient(heatmap), -1)

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
        directions = np.random.uniform(0, 1, size=(steps, num))
        magnitudes = np.random.random((steps, num))
        exponent_var = max_exponent - min_exponent
        gradient = gradient[:, :, 0] + 1j * gradient[:, :, 1]

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
                new_angle = (np.angle(grad) +
                             direction / (grad_weight * np.abs(grad) + 1e-10))
                new_direction = (
                    magnitude * math.cos(new_angle),
                    magnitude * math.sin(new_angle))

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
