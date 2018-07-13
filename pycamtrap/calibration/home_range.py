from __future__ import print_function
from multiprocessing import Pool

import sys
import numpy as np
import pycamtrap as pc

TRIALS_PER_WORLD = 1000
NUM_WORLDS = 10
VELOCITIES = [0.1, 0.3, 0.5, 0.8, 1.0, 1.4]
OCCUPANCIES = np.linspace(0.2, 0.9, 4)


class HomeRangeCalibrator(object):
    def __init__(
            self,
            movement_model,
            velocities=VELOCITIES,
            occupancies=OCCUPANCIES,
            trials_per_world=TRIALS_PER_WORLD,
            num_worlds=NUM_WORLDS):

        self.movement_model = movement_model
        self.velocities = velocities
        self.occupancies = occupancies
        self.trials_per_world = TRIALS_PER_WORLD
        self.num_worlds = NUM_WORLDS

        self.hr_info = self.calculate_hr_info()

    def calculate_hr_info(self):
        n_vel = len(self.velocities)
        n_oc = len(self.occupancies)
        all_info = np.zeros(
            [n_vel, n_oc, self.num_worlds, self.trials_per_world])
        arguments = [
            (self.movement_model, (i, vel), (j, occ), k)
            for i, vel in enumerate(self.velocities)
            for j, occ in enumerate(self.occupancies)
            for k in xrange(self.num_worlds)]

        pool = Pool()
        try:
            results = pool.map(get_single_home_range_info, arguments)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            sys.exit()
            quit()

        for arg, res in zip(arguments, results):
            _, (i, vel), (j, occ), k = arg
            all_info[i, j, k, :] = res

        return all_info

    def plot(self, cmap='Set2', figsize=(10, 10), ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cmap = get_cmap(cmap)
        max_hrange = self.hr_info.max()
        for n, oc in enumerate(self.occupancies):
            color = cmap(float(n) / len(self.occupancies))
            data = self.hr_info[:, n, :, :]
            mean = data.mean(axis=(1, 2))
            std = data.std(axis=(1, 2))

            ax.plot(
                self.velocities,
                mean,
                c=color,
                label='Occupancy: {}'.format(oc))
            ax.fill_between(
                self.velocities,
                mean - std,
                mean + std,
                color=color,
                alpha=0.6,
                edgecolor='white')
        ax.set_yticks(np.linspace(0, max_hrange, 20))
        ax.set_xticks(self.velocities)
        ax.legend()
        return ax


def get_single_home_range_info(argument):
    mov_model, (i, vel), (j, occ), _ = argument
    init = pc.InitialCondition(
        occ,
        velocity=vel)
    mov = pc.MovementData.simulate(
        init,
        num=TRIALS_PER_WORLD,
        velocity=vel,
        days=mov_model.parameters['hr_days'],
        movement_model=mov_model)
    hr = pc.HomeRange(mov)
    return hr.home_ranges
