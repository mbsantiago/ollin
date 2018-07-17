from __future__ import print_function
from __future__ import division
from multiprocessing import Pool

import sys
import numpy as np
import pycamtrap as pc

from ..core.utils import density_to_occupancy


RANGE = 20
TRIALS_PER_WORLD = 100
MAX_INDIVIDUALS = 10000
NUM_WORLDS = 10
HOME_RANGES = np.linspace(0.1, 3, 10)
NICHE_SIZES = np.linspace(0.3, 0.9, 6)
NUMS = np.linspace(10, 1000, 6, dtype=np.int64)


class OccupancyCalibrator(object):
    def __init__(
            self,
            movement_model,
            home_ranges=HOME_RANGES,
            niche_sizes=NICHE_SIZES,
            nums=NUMS,
            trials_per_world=TRIALS_PER_WORLD,
            num_worlds=NUM_WORLDS,
            range=RANGE,
            max_individuals=MAX_INDIVIDUALS):

        self.movement_model = movement_model
        self.home_ranges = home_ranges
        self.niche_sizes = niche_sizes
        self.nums = nums
        self.trials_per_world = trials_per_world
        self.num_worlds = num_worlds
        self.max_individuals = max_individuals
        if isinstance(range, (int, float)):
            range = (range, range)
        self.range = range

        self.oc_info = self.calculate_oc_info()

    def calculate_oc_info(self):
        n_hr = len(self.home_ranges)
        n_nsz = len(self.niche_sizes)
        n_num = len(self.nums)
        tpw = self.trials_per_world
        nw = self.num_worlds
        mov = self.movement_model
        mx_ind = self.max_individuals

        all_info = np.zeros(
                [n_hr, n_nsz, nw, n_num, tpw])
        arguments = [
                Info(mov, hr, nsz, self.nums, tpw, self.range, mx_ind)
                for hr in self.home_ranges
                for nsz in self.niche_sizes
                for k in xrange(self.num_worlds)]

        nargs = len(arguments)
        msg = 'Making {} runs of the simulator\n\tSimulating a total of {} individuals'
        msg = msg.format(nargs, n_hr * n_nsz * tpw * np.sum(self.nums))
        print(msg)
        pool = Pool()
        try:
            results = pool.map(get_single_oc_info, arguments)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            sys.exit()
            quit()

        arguments = [
                (i, j, k)
                for i in xrange(n_hr)
                for j in xrange(n_nsz)
                for k in xrange(self.num_worlds)]

        for (i, j, k), res in zip(arguments, results):
            all_info[i, j, k, :, :] = res

        return all_info

    def plot(self, cmap='Set2', figsize=(10, 10), ax=None, ncols=3):
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cmap = get_cmap(cmap)

        n_hr = len(self.home_ranges)
        nrows = int(np.ceil(n_hr / ncols))
        params = self.movement_model.parameters['density']

        for m, hr in enumerate(self.home_ranges):
            nax = plt.subplot(nrows, ncols, m + 1)
            for n, nsz in enumerate(self.niche_sizes):
                color = cmap(float(n) / len(self.niche_sizes))
                data = self.oc_info[m, n, :, :, :]
                mean = data.mean(axis=(0, 2))
                std = data.std(axis=(0, 2))

                area = self.range[0] * self.range[1]
                density = self.nums / area

                nax.plot(
                    density,
                    mean,
                    c=color,
                    label='Niche Size: {}'.format(nsz))
                nax.fill_between(
                    density,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.6,
                    edgecolor='white')

                target = density_to_occupancy(
                    density, hr, parameters=params)
                nax.plot(
                    density,
                    target,
                    color='red',
                    label='target')

            nax.set_ylim(0, 1)
            nax.set_xlabel('Density (1/Km^2)')
            nax.set_ylabel('Occupancy (%)')
            nax.set_title('Home range: {} Km^2'.format(hr))
            nax.legend()
        plt.tight_layout()

        msg = 'Occupancy Calibration\n{}'.format(self.movement_model.name)
        ax.set_title(msg)
        return ax


class Info(object):
    __slots__ = [
        'movement_model',
        'home_range',
        'niche_size',
        'nums',
        'trials',
        'range',
        'max_individuals']

    def __init__(
            self,
            movement_model,
            home_range,
            niche_size,
            nums,
            trials,
            range_,
            max_individuals):

        self.movement_model = movement_model
        self.home_range = home_range
        self.niche_size = niche_size
        self.nums = nums
        self.max_individuals = max_individuals
        self.trials = trials
        self.range = range_


def get_single_oc_info(info):
    init = pc.InitialCondition(info.niche_size, range=info.range)
    mov = pc.MovementData.simulate(
        init,
        num=info.max_individuals,
        home_range=info.home_range,
        days=info.movement_model.parameters['season'],
        movement_model=info.movement_model)

    n_nums = len(info.nums)
    results = np.zeros([n_nums, info.trials])

    for n, num in enumerate(info.nums):
        for k in xrange(info.trials):
            submov = mov.sample(num)
            oc = pc.Occupancy(submov)
            results[n, k] = oc.get_mean_occupancy()

    return results
