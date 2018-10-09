from __future__ import division

from functools import partial
from multiprocessing import Pool
import logging

from six.moves import range
import numpy as np
import ollin

from ..core.utils import velocity_to_home_range
from .config import BASE_CONFIG


logger = logging.getLogger(__name__)


class HomeRangeCalibrator(object):
    """Class to calibrate Home Range parameters.

    This class also holds all data generated in calibration simulations.

    Attributes
    ----------
    config : :py:obj:`dict`
        Dictionary holding all configuration settings. See :py:mod:`.config` to
        see all relevant settings.
    movement_model : :py:obj:`.MovementModel`
        Reference to Movement model instance being calibrated.
    home_range_info : :py:obj:`array`
        Numpy array of size::

            [num_velocities, num_niches, num_worlds, trials]

        containing occupancy information. Here:

        :num_velocities:
            Refers to the number of simulated mean velocities held in the
            configuration array ``config['velocities']``.
        :num_niches:
            Refers to the number of simulated niche sizes held in the
            configuration array ``config['niche_sizes']``.
        :num_worlds:
            Refers to the number of sites created per selection of
            ``(velocity, niche_size)``.
        :trials:
            Refers to the number of simulations made per site.

        Hence if::

            hr = home_range_info[i, j, k, l]

        this means that ``hr`` was the mean home range in the l-th
        simulation at the k-th world generated with niche size
        ``config['niche_sizes'][j]`` and mean velocity
        ``config['velocities'][i]``.

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
        self.home_range_info = self.calculate_hr_info()

    def calculate_hr_info(self):
        """Simulate multiple scenarios in parallel and record mean home range."""
        trials_per_world = self.config['trials_per_world']
        velocities = self.config['velocities']
        niche_sizes = self.config['niche_sizes']
        num_worlds = self.config['num_worlds']
        days = self.config['days']
        range_ = self.config['range']

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
                    _get_single_hr_info,
                    model=model,
                    days=days,
                    range=range_),
                arguments).get(99999999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            raise KeyboardInterrupt

        logger.info('Simulations done.')

        arguments = [
            (i, j, k)
            for i in range(num_velocities)
            for j in range(num_niches)
            for k in range(num_worlds)]

        for (i, j, k), result in zip(arguments, results):
            all_info[i, j, k, :] = result

        return all_info

    def plot(self, cmap='Set2', figsize=(10, 10), ax=None, plotfit=True):
        """Plot graph of generated home range data and fit.

        Plots a graph of mean velocity versus
        resulting simulated home ranges. Adds a fitted line to the plot if
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

        # Get all relevant data
        niche_sizes = np.array(self.config['niche_sizes'])
        velocities = np.array(self.config['velocities'])
        num_niches = len(niche_sizes)

        # Make new figure if no ax was passed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Get color map
        cmap = get_cmap(cmap)

        max_hrange = self.home_range_info.max()
        for n, oc in enumerate(niche_sizes):
            color = cmap(n / num_niches)
            data = self.home_range_info[:, n, :, :]

            mean = data.mean(axis=(1, 2))
            std = data.std(axis=(1, 2))

            ax.plot(
                velocities,
                mean,
                c=color,
                label='Niche size: {}'.format(oc))
            ax.fill_between(
                velocities,
                mean - std,
                mean + std,
                color=color,
                alpha=0.6,
                edgecolor='white')

        if plotfit:
            target_hr = velocity_to_home_range(
                np.array(velocities),
                parameters=self.movement_model.parameters['home_range'])
            ax.plot(
                velocities,
                target_hr,
                color='red',
                label='target')

        ax.set_yticks(np.linspace(0, max_hrange, 20))
        ax.set_xticks(velocities)

        ax.set_xlabel('Velocity (Km/day)')
        ax.set_ylabel('Home range (Km^2)')

        title = 'Home Range Calibration\n{}'
        title = title.format(self.movement_model.name)
        ax.set_title(title)

        ax.legend()
        return ax

    def fit(self):
        """Fit correction parameter to simulated home range data.

        Returns
        -------
        fit : :py:obj:`dict`
            Dictionary holding the fitted parameters.

        """
        from sklearn.linear_model import LinearRegression

        niche_sizes = np.array(self.config['niche_sizes'])
        velocities = np.array(self.config['velocities'])
        num_niches = niche_sizes.size

        exponents = np.zeros(num_niches)
        alphas = np.zeros(num_niches)

        for num in range(num_niches):
            data = self.home_range_info[:, num, :, :]

            concat = []
            for k, vel in enumerate(velocities):
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


def _get_single_hr_info(args, model, range, days):
    velocity, niche_size, num_individuals = args
    site = ollin.Site.make_random(niche_size, range=range)
    mov = ollin.Movement.simulate(
        site,
        num=num_individuals,
        velocity=velocity,
        days=days,
        movement_model=model)
    hr = ollin.HomeRange(mov)
    return hr.home_ranges
