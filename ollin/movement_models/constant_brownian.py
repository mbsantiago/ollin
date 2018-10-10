from six.moves import xrange
import numpy as np
from numba import jit, float64, int64

from .base import MovementModel


class Model(MovementModel):
    name = 'Constant Brownian Model'
    default_parameters = {
        'movement': {},
        "density": {
            "alpha": 4.950773510732457,
            "density_exp": 0.8501718469622543,
            "hr_exp": 0.8441724458551622,
            "niche_size_exp": 0.8760285191282491
            },
        "home_range": {
            "alpha": 59.02095500748234,
            "exponent": 1.903072119815655
            },
        "velocity": {
            "alpha": 0.002223569915673946,
            "beta": 1.013044346939526
            }
    }

    def generate_movement(
            self,
            initial_positions,
            site,
            steps,
            velocity):
        range_ = site.range
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
