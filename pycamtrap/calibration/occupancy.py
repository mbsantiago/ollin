from __future__ import print_function
from __future__ import division
from multiprocessing import Pool

import sys
import numpy as np
import pycamtrap as pc

from ..core.utils import density_to_occupancy, logit


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
                [n_hr, n_nsz, n_num, nw, tpw])
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
            all_info[i, j, :, k, :] = res

        return all_info

    def plot(
            self,
            figsize=(10, 10),
            ax=None,
            w_target=True,
            xscale=None,
            yscale=None):
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
                mean = data.mean(axis=(1, 2))
                std = data.std(axis=(1, 2))
                uplim = mean + std
                dnlim = mean - std

                area = self.range[0] * self.range[1]
                density = self.nums / area

                xtext = 0.1
                ytext = 0.8

                ylim0, ylim1 = 0, 1

                if xscale == 'log':
                    density = np.log(density)
                    xtext = np.log(xtext)

                if yscale == 'log':
                    mean = np.log(mean)
                    uplim = np.log(uplim)
                    dnlim = np.log(dnlim)
                    ylim0 = -4
                    ylim1 = 0
                    ytext = np.log(ytext)

                if yscale == 'logit':
                    mean = logit(mean)
                    uplim = logit(uplim)
                    dnlim = logit(dnlim)
                    ylim0 = -4
                    ylim1 = 4
                    ytext = logit(ytext)

                nax.plot(
                    density,
                    mean)
                nax.fill_between(
                    density,
                    dnlim,
                    uplim,
                    alpha=0.6,
                    edgecolor='white')

                if w_target:
                    target = density_to_occupancy(
                        density,
                        hr,
                        nsz,
                        self.range,
                        parameters=params)

                    if yscale == 'log':
                        target = np.log(target)
                    if yscale == 'logit':
                        target = logit(target)

                    nax.plot(
                        density,
                        target,
                        color='red',
                        label='target')

                nax.set_ylim(ylim0, ylim1)
                nax.set_xlim(density.min(), density.max())

                nax.text(xtext, ytext, 'HR={} Km^2\nNS={} (%)'.format(hr, nsz))

                if m == ncols - 1:
                    nax.set_xlabel('NS={}'.format(nsz))

                if n == 0:
                    nax.set_ylabel('HR={}'.format(hr))

                if m < ncols - 1:
                    nax.xaxis.set_major_formatter(NullFormatter())
                if n > 0:
                    nax.yaxis.set_major_formatter(NullFormatter())

                counter += 1
        plt.subplots_adjust(wspace=0, hspace=0)

        font = {'fontsize': 18}
        plt.figtext(0.4, 0.05, "Density (Km^-2)", fontdict=font)
        plt.figtext(0.035, 0.5, "Occupancy (%)", fontdict=font, rotation=90)
        plt.figtext(0.38, 0.92, "Occupancy Calibration", fontdict=font)
        return ax

    def fit(self):
        from sklearn.linear_model import LinearRegression
        data = self.oc_info
        nhrs, nnsz, nwrl, nnums, ntpw = data.shape

        home_range_exponents = np.zeros([nnsz])
        density_exponents = np.zeros([nnsz])
        proportionality_constants = np.zeros([nnsz])

        area = self.range[0] * self.range[1]
        density = self.nums / float(area)

        for i, nsz in enumerate(self.niche_sizes):
            X = []
            Y = []
            for j, hr in enumerate(self.home_ranges):
                for k, dens in enumerate(density):
                    oc_data = data[j, i, k, :, :].ravel()
                    hr_data = hr * np.ones_like(oc_data)
                    dens_data = dens * np.ones_like(oc_data)
                    Y.append(logit(oc_data))
                    X.append(
                        np.stack(
                            [np.log(hr_data), np.log(dens_data)], -1))
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)

            lrm = LinearRegression()
            lrm.fit(X, Y)

            home_range_exponents[i] = lrm.coef_[0]
            density_exponents[i] = lrm.coef_[1]
            proportionality_constants[i] = lrm.intercept_

        lrm_dens = LinearRegression()
        lrm_dens.fit(self.niche_sizes[:, None], density_exponents)

        den_exp_a = lrm_dens.coef_[0]
        den_exp_b = lrm_dens.intercept_

        lrm_prop = LinearRegression()
        lrm_prop.fit(
            self.niche_sizes[:, None], proportionality_constants)

        alpha = lrm_prop.coef_[0]
        beta = lrm_prop.intercept_

        parameters = {
            'hr_exp': -home_range_exponents.mean(),
            'alpha': alpha,
            'beta': beta,
            'den_exp_a': den_exp_a,
            'den_exp_b': den_exp_b}
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
