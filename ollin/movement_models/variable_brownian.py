from six.moves import xrange
import numpy as np
from numba import jit, float64, int64

from .base import MovementModel


class Model(MovementModel):
    name = 'Variable Brownian Model'
    default_parameters = {
        'movement': {
            'niche_weight': 0.2},
        "density": {
            "alpha": 4.956627725964207,
            "density_exp": 0.8509470680476414,
            "hr_exp": 0.8449833387462566,
            "niche_size_exp": 0.8752336080881222
            },
        "home_range": {
            "alpha": 57.61382053222021,
            "exponent": 1.9096337585646315
            },
        "velocity": {
            "alpha": 0.14473388093667755,
            "beta": 0.8918520334954713
            }
    }

    def generate_movement(
            self,
            initial_positions,
            site,
            steps,
            velocity):
        niche_weight = self.parameters['movement']['niche_weight']

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
                direction *= 1 + niche_weight * (0.5 - value)
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
