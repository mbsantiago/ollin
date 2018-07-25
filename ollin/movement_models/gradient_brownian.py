from six.moves import xrange
import math
import numpy as np
from numba import jit, float64, int64

from .basemodel import MovementModel


class Model(MovementModel):
    name = 'Gradient Brownian Model'
    default_parameters = {
        'movement': {
            'grad_weight': 10.0,
            'niche_weight': 1.0},
        "density": {
            "alpha": 5.054484212556172,
            "density_exp": 0.8546502084649101,
            "hr_exp": 0.8556172207468531,
            "niche_size_exp": 0.9376571357760055
            },
        "home_range": {
            "alpha": 37.23282280460764,
            "exponent": 1.8750683470837832
            },
        "velocity": {
            "alpha": 0.6601881286894935,
            "beta": 0.5790070345559946
            }
    }

    def generate_movement(
            self,
            initial_positions,
            site,
            steps,
            velocity):
        grad_weight = self.parameters['movement']['grad_weight']
        niche_weight = self.parameters['movement']['niche_weight']

        heatmap = site.niche
        resolution = site.resolution
        range_ = site.range

        gradient = np.stack(np.gradient(heatmap), -1)

        mov = self._movement(
            gradient,
            heatmap,
            initial_positions,
            resolution,
            velocity,
            range_,
            steps,
            grad_weight,
            niche_weight)
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
            grad_weight,
            niche_weight):
        num, _ = random_positions.shape
        movement = np.zeros((num, steps, 2), dtype=float64)
        rangex, rangey = range_
        directions = np.random.normal(0, 1, (num, steps))
        gradient = gradient[:, :, 0] + 1j * gradient[:, :, 1]

        for k in xrange(steps):
            movement[:, k, :] = random_positions
            for j in xrange(num):
                direction = directions[j, k]
                index = (
                    random_positions[j, 0] // resolution,
                    random_positions[j, 1] // resolution)
                grad = gradient[int(index[0]), int(index[1])]
                value = heatmap[int(index[0]), int(index[1])]

                magnitude = velocity * (1 + niche_weight * (0.5 - value))
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
