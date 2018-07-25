from six.moves import xrange
import numpy as np
import math
from numba import jit, float64, int64

from .basemodel import MovementModel


class Model(MovementModel):
    name = 'Constant Levy Model'
    default_parameters = {
        'movement': {
            'pareto': 1.8},
        "density": {
            "alpha": 5.139654979615388,
            "density_exp": 0.8673231302202313,
            "hr_exp": 0.8709250205414867,
            "niche_size_exp": 0.8105129468063047
            },
        "home_range": {
            "alpha": 82.0335925804581,
            "exponent": 1.9018678531419373
            },
        "velocity": {
            "alpha": 0.0005826019701355142,
            "beta": 1.0525954033363216
            }
    }

    def generate_movement(
            self,
            initial_positions,
            site,
            steps,
            velocity):
        exponent = self.parameters['movement']['pareto']
        range_ = site.range

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
