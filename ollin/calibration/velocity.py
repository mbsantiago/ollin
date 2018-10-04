from multiprocessing import Pool
import logging
from functools import partial

import sys
import numpy as np
import ollin
from ollin.core.utils import velocity_modification


logger = logging.getLogger(__name__)

RANGE = 20
TRIALS_PER_WORLD = 100
NUM_WORLDS = 10
VELOCITIES = np.linspace(0.1, 3, 10)
NICHE_SIZES = np.linspace(0.3, 0.9, 6)


class VelocityCalibrator(object):
    def __init__(
            self,
            movement_model,
            velocities=VELOCITIES,
            niche_sizes=NICHE_SIZES,
            trials_per_world=TRIALS_PER_WORLD,
            num_worlds=NUM_WORLDS,
            range=RANGE,
            **kwargs):

        self.movement_model = movement_model
        self.velocities = velocities
        self.niche_sizes = niche_sizes
        self.trials_per_world = trials_per_world
        self.num_worlds = num_worlds
        self.range = range

        self.velocity_info = self.calculate_velocity_info()

    def calculate_velocity_info(self):
        n_vel = len(self.velocities)
        n_nsz = len(self.niche_sizes)
        num = self.trials_per_world
        mov = self.movement_model

        all_info = np.zeros(
            [n_vel, n_nsz, self.num_worlds, self.trials_per_world])
        arguments = [
            (vel, nsz, num)
            for vel in self.velocities
            for nsz in self.niche_sizes
            for k in xrange(self.num_worlds)]

        logger.info('Simulating {} scenarios'.format(len(arguments)))
        pool = Pool()
        try:
            results = pool.map(
                partial(
                    _get_single_velocity_info,
                    model=mov,
                    range=self.range),
                arguments)
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

        for n, nsz in enumerate(self.niche_sizes):
            color = cmap(float(n) / len(self.niche_sizes))
            data = self.velocity_info[:, n, :, :]
            mean = data.mean(axis=(1, 2))
            std = data.std(axis=(1, 2))

            ax.plot(
                self.velocities,
                mean,
                c=color,
                label='Niche Size: {}'.format(round(nsz, 2)))
            ax.fill_between(
                self.velocities,
                mean - std,
                mean + std,
                color=color,
                alpha=0.6,
                edgecolor='white')

            if plotfit:
                vel_mod = velocity_modification(
                    nsz, self.movement_model.parameters)
                target_velocities = vel_mod * self.velocities

                ax.plot(
                    self.velocities,
                    target_velocities,
                    c='red',
                    label='Niche Size: {} (fit)'.format(round(nsz, 2)))

        ax.set_title('Velocity Calibration')

        ax.set_xlabel('target velocity (Km/day)')
        ax.set_ylabel('calculated velocity (Km/day)')

        ax.set_xticks(self.velocities)
        ax.set_yticks(self.velocities)

        ax.legend()
        return ax

    def fit(self):
        from sklearn.linear_model import LinearRegression

        coefficients = np.zeros(len(self.niche_sizes))
        for num, niche_size in enumerate(self.niche_sizes):
            given_vel = []
            simulated_vel = []

            for k, vel in enumerate(self.velocities):
                vdata = self.velocity_info[k, num, :, :].ravel()
                given_vel.append(vel * np.ones_like(vdata))
                simulated_vel.append(vdata)

            given_vel = np.concatenate(given_vel, 0)
            simulated_vel = np.concatenate(simulated_vel, 0)
            model = LinearRegression(fit_intercept=False)
            model.fit(given_vel[:, None], simulated_vel)
            coefficients[num] = 1 / model.coef_[0]

        model = LinearRegression()
        model.fit(self.niche_sizes[:, None], coefficients)

        alpha = model.coef_[0]
        beta = model.intercept_

        fit = {
            'alpha': alpha,
            'beta': beta}
        return fit


def _get_single_velocity_info(args, model, range):
    velocity, niche_size, num_individuals = args
    site = ollin.Site.make_random(niche_size, range=range)
    mov = ollin.Movement.simulate(
        site,
        num=num_individuals,
        velocity=velocity,
        days=model.parameters['hr_days'],
        movement_model=model)
    analyzer = ollin.MovementAnalysis(mov)
    return analyzer.velocities.mean(axis=1)
