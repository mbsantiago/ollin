from __future__ import division

from multiprocessing import Pool
import logging
from functools import partial

from six.moves import range
import numpy as np
import ollin

from ..core.utils import velocity_modification
from .config import BASE_CONFIG


logger = logging.getLogger(__name__)


class VelocityCalibrator(object):
    """Class to calibrate velocity parameters.

    This class also holds all data generated in calibration simulations.

    Attributes
    ----------
    config : :py:obj:`dict`
        Dictionary holding all configuration settings. See :py:mod:`.config` to
        see all relevant settings.
    movement_model : :py:obj:`.MovementModel`
        Reference to Movement model instance being calibrated.
    velocity_info : :py:obj:`array`
        Numpy array of size::

            [num_velocities, num_niches, num_worlds, trials_per_world]

        containing velocity information. Here:

        :num_velocities:
            Refers to the number of simulated mean velocities held in the
            configuration array ``config['velocities']``.
        :num_niches:
            Refers to the number of simulated niche sizes held in the
            configuration array ``config['niche_sizes']``.
        :num_worlds:
            Refers to the number of sites created per selection of
            ``(velocity, niche_size)``.
        :trials_per_world:
            Refers to the number of simulations made per site.

        Hence if::

            vel = self.velocity_info[i, j, k, l]

        this means that ``vel`` was the mean velocity in the l-th simulation at
        the k-th world generated with niche size ``config['niche_sizes'][j]``
        and "model"-mean velocity ``config['velocities'][i]``.

    """

    def __init__(self, movement_model, config=None):
        # Handle configurations
        if config is None:
            config = {}
        copy = BASE_CONFIG.copy()
        copy.update(config)
        self.config = copy

        # Point to movement model
        self.movement_model = movement_model

        # Calculate calibrations
        self.velocity_info = self.calculate_velocity_info()

    def calculate_velocity_info(self):
        """Simulate multiple scenarios in parallel and record mean velocity."""
        trials_per_world = self.config['trials_per_world']
        velocities = self.config['velocities']
        niche_sizes = self.config['niche_sizes']
        num_worlds = self.config['num_worlds']
        range_ = self.config['range']
        days = self.config['days']

        model = self.movement_model

        num_velocities = len(velocities)
        num_niches = len(niche_sizes)

        all_info = np.zeros([
            num_velocities,
            num_niches,
            num_worlds,
            trials_per_world
        ])

        arguments = [
            (velocity, niche_size, trials_per_world)
            for velocity in velocities
            for niche_size in niche_sizes
            for k in range(num_worlds)]

        logger.info('Simulating %d scenarios', len(arguments))
        pool = Pool()
        try:
            results = pool.map_async(
                partial(
                    _get_single_velocity_info,
                    model=model,
                    range=range_,
                    days=days),
                arguments
            ).get(999999999999999)

            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            raise KeyboardInterrupt
        logger.info('Simulations done')

        arguments = [
            (i, j, k)
            for i in range(num_velocities)
            for j in range(num_niches)
            for k in range(num_worlds)]

        for (i, j, k), result in zip(arguments, results):
            all_info[i, j, k, :] = result

        return all_info

    def plot(self, cmap='Set2', figsize=(10, 10), ax=None, plotfit=True):
        """Plot graph of generated velocity data and fit.

        Plots a graph of internal Model's "mean-velocity" parameter versus
        resulting simulated mean velocity. Adds a fitted line to the plot if
        desired to visually check calibration.

        Arguments
        ---------
        cmap : :py:obj:`str`, optional
            Name of colormap to use for diferent niche sizes.
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Ax in which to draw the plot. If not given a new one will be
            created.
        fisize : :py:obj:`tuple`, optional
            Size of figure to create. Used only if no ax is given.
        plotfit : bool, optional
            If True will plot a line fitted to velocity data. Defaults to True.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Ax object for further plotting.

        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        niche_sizes = np.array(self.config['niche_sizes'])
        velocities = np.array(self.config['velocities'])
        num_niches = len(niche_sizes)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cmap = get_cmap(cmap)

        for n, nsz in enumerate(niche_sizes):
            color = cmap(n / num_niches)
            data = self.velocity_info[:, n, :, :]

            mean = data.mean(axis=(1, 2))
            std = data.std(axis=(1, 2))

            ax.plot(
                velocities,
                mean,
                c=color,
                label='Niche Size: {}'.format(round(nsz, 2)))
            ax.fill_between(
                velocities,
                mean - std,
                mean + std,
                color=color,
                alpha=0.6,
                edgecolor='white')

            if plotfit:
                vel_mod = velocity_modification(
                    nsz, self.movement_model.parameters)
                target_velocities = vel_mod * velocities

                ax.plot(
                    velocities,
                    target_velocities,
                    c='red',
                    label='Niche Size: {} (fit)'.format(round(nsz, 2)))

        ax.set_title('Velocity Calibration')

        ax.set_xlabel('target velocity (Km/day)')
        ax.set_ylabel('calculated velocity (Km/day)')

        ax.set_xticks(velocities)
        ax.set_yticks(velocities)

        ax.legend()
        return ax

    def fit(self):
        """Fit correction parameter to simulated velocity data.

        Returns
        -------
        fit : :py:obj:`dict`
            Dictionary holding the fitted parameters.

        """
        from sklearn.linear_model import LinearRegression

        niche_sizes = np.array(self.config['niche_sizes'])
        velocities = np.array(self.config['velocities'])

        num_niches = len(niche_sizes)

        coefficients = np.zeros(num_niches)
        for num in range(num_niches):

            given_vel = []
            simulated_vel = []
            for k, vel in enumerate(velocities):
                vdata = self.velocity_info[k, num, :, :].ravel()
                given_vel.append(vel * np.ones_like(vdata))
                simulated_vel.append(vdata)

            given_vel = np.concatenate(given_vel, 0)
            simulated_vel = np.concatenate(simulated_vel, 0)
            model = LinearRegression(fit_intercept=False)
            model.fit(given_vel[:, None], simulated_vel)
            coefficients[num] = 1 / model.coef_[0]

        model = LinearRegression()
        model.fit(niche_sizes[:, None], coefficients)

        alpha = model.coef_[0]
        beta = model.intercept_

        fit = {
            'alpha': alpha,
            'beta': beta}
        return fit


def _get_single_velocity_info(args, model, range, days):
    velocity, niche_size, num_individuals = args
    site = ollin.Site.make_random(niche_size, range=range)
    mov = ollin.Movement.simulate(
        site,
        num=num_individuals,
        velocity=velocity,
        days=days,
        movement_model=model)
    analyzer = mov.analyze('velocity')
    return analyzer.results.mean(axis=1)
