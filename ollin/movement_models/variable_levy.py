from six.moves import xrange
import math
import numpy as np
from numba import jit, float64, int64

from .base import MovementModel


class Model(MovementModel):
    name = 'Variable Levy Model'
    default_parameters = {
        'movement': {
            'min_pareto': 1.1,
            'max_pareto': 1.9},
        "density": {
            "alpha": 5.496771005099745,
            "density_exp": 0.9089427578933271,
            "hr_exp": 0.9313289770069533,
            "niche_size_exp": 0.552259555326986
            },
        "home_range": {
            "alpha": 84.94331014956782,
            "exponent": 1.866467806903153
            },
        "velocity": {
            "alpha": -3.1454639316707027,
            "beta": 7.029550127524596
            }
    }

    def generate_movement(
            self,
            initial_positions,
            site,
            steps,
            velocity):
        min_exponent = self.parameters['movement']['min_pareto']
        max_exponent = self.parameters['movement']['max_pareto']

        heatmap = site.niche
        resolution = site.resolution
        range_ = site.range

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
