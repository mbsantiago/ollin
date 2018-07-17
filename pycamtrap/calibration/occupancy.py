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
HOME_RANGES = np.linspace(0.1, 3, 6)
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
        msg = 'Making {} runs of the simulator'
        msg += '\n\tSimulating a total of {} individuals'
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

    def plot(self, cmap='Set2', figsize=(10, 10), ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ncols = len(self.home_ranges)
        nrows = len(self.niche_sizes)
        params = self.movement_model.parameters['density']

        counter = 1
        for m, hr in enumerate(self.home_ranges):
            for n, nsz in enumerate(self.niche_sizes):
                nax = plt.subplot(nrows, ncols, counter)
                data = self.oc_info[m, n, :, :, :]
                mean = data.mean(axis=(0, 2))
                std = data.std(axis=(0, 2))

                area = self.range[0] * self.range[1]
                density = self.nums / area

                nax.plot(
                    density,
                    mean)
                nax.fill_between(
                    density,
                    mean - std,
                    mean + std,
                    alpha=0.6,
                    edgecolor='white')

                target = density_to_occupancy(
                    density, hr, nsz, parameters=params)

                nax.plot(
                    density,
                    target,
                    color='red',
                    label='target')

                nax.set_ylim(0, 1)
                nax.set_xlim(0, density.max())
                nax.text(0.1, 0.8, 'HR={} Km^2\nNS={} (%)'.format(hr, nsz))

                if m == ncols - 1:
                    nax.set_ylabel('Home Range={}'.format(hr))

                if n == 0:
                    nax.set_xlabel('Niche Size={}'.format(nsz))

                if m < ncols - 1:
                    nax.xaxis.set_major_formatter(NullFormatter())
                if n > 0:
                    nax.yaxis.set_major_formatter(NullFormatter())

                counter += 1

        plt.subplots_adjust(wspace=0, hspace=0)
        return ax

    def fit(self):
        from sklearn.linear_model import LinearRegression
        data = self.oc_info
        nhrs, nnsz, nwrl, nnums, ntpw = data.shape

        home_range_exponents = np.zeros([nnsz])
        occupancy_exponents = np.zeros([nnsz])
        proportionality_constants = np.zeros([nnsz])

        area = self.range[0] * self.range[1]
        density = self.nums / float(area)

        for i, nsz in enumerate(self.niche_sizes):
            X = []
            Y = []
            for j, hr in enumerate(self.home_ranges):
                for k, dens in enumerate(density):
                    oc_data = data[j, i, :, k, :].ravel()
                    hr_data = hr * np.ones_like(oc_data)
                    dens_data = dens * np.ones_like(oc_data)
                    Y.append(dens_data)
                    X.append(np.stack([hr_data, oc_data], -1))
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)[:, None]

            lrm = LinearRegression()
            lrm.fit(np.log(X), np.log(Y))

            home_range_exponents[i] = lrm.coef_[0, 0]
            occupancy_exponents[i] = lrm.coef_[0, 1]
            proportionality_constants[i] = lrm.intercept_[0]

        lrm_occ = LinearRegression()
        lrm_occ.fit(self.niche_sizes[:, None], occupancy_exponents[:, None])

        occ_exp_a = lrm_occ.coef_[0, 0]
        occ_exp_b = lrm_occ.intercept_[0]

        lrm_prop = LinearRegression()
        lrm_prop.fit(
            self.niche_sizes[:, None], proportionality_constants[:, None])

        alpha = lrm_prop.coef_[0, 0]
        beta = lrm_prop.intercept_[0]

        print(home_range_exponents, home_range_exponents.mean())

        parameters = {
            'hr_exp': home_range_exponents.mean(),
            'alpha': alpha,
            'beta': beta,
            'occ_exp_a': occ_exp_a,
            'occ_exp_b': occ_exp_b}
        return parameters


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
