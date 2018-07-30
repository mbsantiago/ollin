from __future__ import print_function
from __future__ import division
from multiprocessing import Pool

import sys
import numpy as np
import ollin

from ..core.utils import density_to_occupancy, logit
from ..core.constants import GLOBAL_CONSTANTS


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
            range=GLOBAL_CONSTANTS['range'],
            season=GLOBAL_CONSTANTS['season'],
            max_individuals=MAX_INDIVIDUALS):

        self.movement_model = movement_model
        self.home_ranges = home_ranges
        self.niche_sizes = niche_sizes
        self.nums = nums
        self.trials_per_world = trials_per_world
        self.num_worlds = num_worlds
        self.max_individuals = max_individuals
        self.season = season
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
        season = self.season

        all_info = np.zeros(
                [n_hr, n_nsz, n_num, nw, tpw])
        arguments = [
                Info(mov, hr, nsz, self.nums, tpw, self.range, season, mx_ind)
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
            x_var='density',
            w_target=True,
            xscale=None,
            yscale=None,
            beta=1,
            lwidth=0.1,
            wtext=False,
            wspace=0):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        area = self.range[0] * self.range[1]

        density = self.nums / area
        hr_proportions = self.home_ranges / area

        if x_var == 'density':
            iterator1 = hr_proportions
            var2 = 'HRP'
            iterator2 = self.niche_sizes
            var3 = 'NS'
        elif x_var == 'home_range':
            iterator1 = density
            var2 = 'D'
            iterator2 = self.niche_sizes
            var3 = 'NS'
        elif x_var == 'niche_sizes':
            iterator1 = hr_proportions
            var2 = 'HRP'
            iterator2 = density
            var3 = 'D'

        ncols = len(iterator1)
        nrows = len(iterator2)
        params = self.movement_model.parameters['density']

        counter = 1
        for m, x in enumerate(iterator1):
            for n, y in enumerate(iterator2):
                nax = plt.subplot(nrows, ncols, counter)

                if x_var == 'density':
                    data = self.oc_info[m, n, :, :, :]
                    xcoords = density
                elif x_var == 'home_range':
                    data = self.oc_info[:, n, m, :, :]
                    xcoords = beta * hr_proportions
                elif x_var == 'niche_sizes':
                    data = self.oc_info[m, :, n, :, :]
                    xcoords = self.niche_sizes

                mean = data.mean(axis=(1, 2))
                std = data.std(axis=(1, 2))
                uplim = mean + std
                dnlim = mean - std

                xtext = 0.1
                ytext = 0.8

                ylim0, ylim1 = -0.1, 1.1

                if xscale == 'log':
                    xcoords = np.log(xcoords)
                    xtext = np.log(xtext)

                if yscale == 'log':
                    mean = np.log(mean)
                    uplim = np.log(uplim)
                    dnlim = np.log(dnlim)
                    ylim0 = -6
                    ylim1 = 0
                    ytext = np.log(ytext)

                if yscale == 'logit':
                    mean = logit(mean)
                    uplim = logit(uplim)
                    dnlim = logit(dnlim)
                    ylim0 = -6
                    ylim1 = 4
                    ytext = logit(ytext)

                nax.plot(
                    xcoords,
                    mean,
                    linewidth=lwidth)
                nax.fill_between(
                    xcoords,
                    dnlim,
                    uplim,
                    alpha=0.6,
                    edgecolor='white')

                if w_target:
                    if x_var == 'density':
                        target = density_to_occupancy(
                            density,
                            x,
                            y,
                            parameters=params)
                    elif x_var == 'home_range':
                        target = density_to_occupancy(
                            x,
                            hr_proportions,
                            y,
                            parameters=params)
                    elif x_var == 'niche_sizes':
                        target = density_to_occupancy(
                            y,
                            x,
                            self.niche_sizes,
                            parameters=params)

                    if yscale == 'log':
                        target = np.log(target)
                    if yscale == 'logit':
                        target = logit(target)

                    nax.plot(
                        xcoords,
                        target,
                        color='red',
                        label='target')

                nax.set_ylim(ylim0, ylim1)
                nax.set_xlim(xcoords.min(), xcoords.max())

                if wtext:
                    nax.text(
                        xtext, ytext, '{}={}\n{}={}'.format(var2, x, var3, y))

                if m == ncols - 1:
                    nax.set_xlabel('{}={}'.format(var3, y))
                if n == 0:
                    nax.set_ylabel('{}={}'.format(var2, x))

                if m < ncols - 1:
                    nax.xaxis.set_major_formatter(NullFormatter())
                if n > 0:
                    nax.yaxis.set_major_formatter(NullFormatter())

                counter += 1
        plt.subplots_adjust(wspace=wspace, hspace=wspace)

        font = {'fontsize': 18}
        plt.figtext(0.4, 0.035, x_var, fontdict=font)
        plt.figtext(0.035, 0.5, "Occupancy (%)", fontdict=font, rotation=90)
        title = "Occupancy Calibration\n{}"
        title = title.format(self.movement_model.name)
        plt.figtext(0.38, 0.92, title, fontdict=font)
        return ax

    def fit(self):
        from sklearn.linear_model import LinearRegression
        data = self.oc_info
        area = self.range[0] * self.range[1]
        density = self.nums / float(area)
        hr_proportions = self.home_ranges / float(area)

        X = []
        Y = []
        for i, nsz in enumerate(self.niche_sizes):
            for j, hr in enumerate(hr_proportions):
                for k, dens in enumerate(density):
                    oc_data = data[j, i, k, :, :].ravel()
                    hr_data = hr * np.ones_like(oc_data)
                    dens_data = dens * np.ones_like(oc_data)
                    nsz_data = nsz * np.ones_like(oc_data)
                    Y.append(logit(oc_data))
                    X.append(
                        np.stack([np.log(hr_data),
                                  np.log(dens_data),
                                  np.log(nsz_data)], -1))
        X = np.concatenate(X, 0)
        Y = np.concatenate(Y, 0)

        lrm = LinearRegression()
        lrm.fit(X, Y)

        alpha = lrm.intercept_
        hr_exp = lrm.coef_[0]
        den_exp = lrm.coef_[1]
        nsz_exp = lrm.coef_[2]

        parameters = {
            'alpha': alpha,
            'hr_exp': hr_exp,
            'density_exp': den_exp,
            'niche_size_exp': nsz_exp}
        return parameters


class Info(object):
    __slots__ = [
        'movement_model',
        'home_range',
        'niche_size',
        'nums',
        'trials',
        'range',
        'season',
        'max_individuals']

    def __init__(
            self,
            movement_model,
            home_range,
            niche_size,
            nums,
            trials,
            range_,
            season,
            max_individuals):

        self.movement_model = movement_model
        self.home_range = home_range
        self.niche_size = niche_size
        self.nums = nums
        self.max_individuals = max_individuals
        self.season = season
        self.trials = trials
        self.range = range_


def get_single_oc_info(info):
    site = ollin.Site.make_random(info.niche_size, range=info.range)
    mov = ollin.Movement.simulate(
        site,
        num=info.max_individuals,
        home_range=info.home_range,
        days=info.season,
        movement_model=info.movement_model)

    n_nums = len(info.nums)
    results = np.zeros([n_nums, info.trials])

    for n, num in enumerate(info.nums):
        for k in xrange(info.trials):
            submov = mov.sample(num)
            oc = ollin.Occupancy(submov)
            results[n, k] = oc.occupancy

    return results
