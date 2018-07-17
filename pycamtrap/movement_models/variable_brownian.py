import numpy as np
from numba import jit, float64, int64

from .basemodel import MovementModel
from ..core.utils import normalize


class Model(MovementModel):
    name = 'Variable Brownian Model'
    default_parameters = {
        'velocity': {
            'beta': 0.97,
            'alpha': 0.0},
        'home_range': {
            'alpha': 35.0,
            'exponent': 0.54},
        'density': {
            'alpha': 0.0,
            'beta': 22.0,
            'hr_exp': 1.2,
            'den_exp_a': 1.75,
            'den_exp_b': 1.75},
        'movement': {
            'niche_weight': 0.2},
    }

    def __init__(self, parameters=None):
        super(Model, self).__init__(parameters)

    def generate_movement(
            self,
            initial_positions,
            initial_conditions,
            days,
            velocity):
        niche_weight = self.parameters['movement']['niche_weight']
        heatmap = normalize(initial_conditions.kde_approximation)
        resolution = initial_conditions.resolution
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
