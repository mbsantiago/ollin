from __future__ import print_function
from multiprocessing import Pool

import sys
import numpy as np
import ollin

from ..core.utils import velocity_to_home_range

TRIALS_PER_WORLD = 1000
NUM_WORLDS = 10
VELOCITIES = [0.1, 0.3, 0.5, 0.8, 1.0, 1.4]
NICHE_SIZES = np.linspace(0.2, 0.9, 4)
RANGE = 20


class HomeRangeCalibrator(object):
    def __init__(
            self,
            movement_model,
            velocities=VELOCITIES,
            niche_sizes=NICHE_SIZES,
            trials_per_world=TRIALS_PER_WORLD,
            num_worlds=NUM_WORLDS,
            range=RANGE):

        self.movement_model = movement_model
        self.velocities = velocities
        self.niche_sizes = niche_sizes
        self.trials_per_world = TRIALS_PER_WORLD
        self.num_worlds = NUM_WORLDS
        self.range = range

        self.hr_info = self.calculate_hr_info()

    def calculate_hr_info(self):
        n_vel = len(self.velocities)
        n_nsz = len(self.niche_sizes)
        num = self.trials_per_world
        mov = self.movement_model

        all_info = np.zeros(
            [n_vel, n_nsz, self.num_worlds, self.trials_per_world])
        arguments = [
            Info(mov, vel, nsz, num, self.range)
            for vel in self.velocities
            for nsz in self.niche_sizes
            for k in xrange(self.num_worlds)]

        print('Simulating {} scenarios'.format(len(arguments)))
        pool = Pool()
        try:
            results = pool.map(get_single_hr_info, arguments)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            sys.exit()
            quit()

        arguments = [
                (i, j, k)
                for i in xrange(n_vel)
                for j in xrange(n_nsz)
                for k in xrange(self.num_worlds)]

        for (i, j, k), res in zip(arguments, results):
            all_info[i, j, k, :] = res

        return all_info

    def plot(self, cmap='Set2', figsize=(10, 10), ax=None, plotfit=True):
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cmap = get_cmap(cmap)
        max_hrange = self.hr_info.max()
        for n, oc in enumerate(self.niche_sizes):
            color = cmap(float(n) / len(self.niche_sizes))
            data = self.hr_info[:, n, :, :]
            mean = data.mean(axis=(1, 2))
            std = data.std(axis=(1, 2))

            ax.plot(
                self.velocities,
                mean,
                c=color,
                label='Niche size: {}'.format(oc))
            ax.fill_between(
                self.velocities,
                mean - std,
                mean + std,
                color=color,
                alpha=0.6,
                edgecolor='white')

        if plotfit:
            target_hr = velocity_to_home_range(
                np.array(self.velocities),
                parameters=self.movement_model.parameters['home_range'])
            ax.plot(
                self.velocities,
                target_hr,
                color='red',
                label='target')

        ax.set_yticks(np.linspace(0, max_hrange, 20))
        ax.set_xticks(self.velocities)
        ax.set_xlabel('Velocity (Km/day)')
        ax.set_ylabel('Home range (Km^2)')
        title = 'Home Range Calibration\n{}'
        title = title.format(self.movement_model.name)
        ax.set_title(title)
        ax.legend()
        return ax

    def fit(self):
        from sklearn.linear_model import LinearRegression

        exponents = np.zeros(len(self.niche_sizes))
        alphas = np.zeros(len(self.niche_sizes))
        for num, niche_size in enumerate(self.niche_sizes):
            data = self.hr_info[:, num, :, :]
            concat = []

            for k, vel in enumerate(self.velocities):
                hrdata = data[k, :, :].ravel()
                hrdata = np.stack([vel * np.ones_like(hrdata), hrdata], -1)
                concat.append(hrdata)

            data = np.concatenate(concat, 0)
            velocity, home_range = data.T
            model = LinearRegression()
            model.fit(np.log(velocity)[:, None], np.log(home_range))

            exponents[num] = model.coef_[0]
            alphas[num] = np.exp(model.intercept_)

        fit = {
            'alpha': alphas.mean(),
            'exponent': exponents.mean()}
        return fit


class Info(object):
    __slots__ = [
        'movement_model',
        'velocity',
        'niche_size',
        'num',
        'range']

    def __init__(
            self,
            movement_model,
            velocity,
            niche_size,
            num,
            range_):
        self.movement_model = movement_model
        self.velocity = velocity
        self.niche_size = niche_size
        self.num = num
        self.range = range_


def get_single_hr_info(info):
    site = ollin.Site.make_random(info.niche_size, range=info.range)
    mov = ollin.Movement.simulate(
        site,
        num=info.num,
        velocity=info.velocity,
        days=info.movement_model.parameters['hr_days'],
        movement_model=info.movement_model)
    hr = ollin.HomeRange(mov)
    return hr.home_ranges
