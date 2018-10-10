import numpy as np

from .base import MovementAnalyzer


class Analyzer(MovementAnalyzer):
    r"""Turn Angle analyzer.

    Turn angle refers to the angle formed by two successive steps of an
    individual motion.

    Hence if :math:`p(t_0) = (x_0, y_0)`, :math:`p(t_1) = (x_1, y_1)` and
    :math:`p(t_2) = (x_2, y_2)` are the
    positions of an individual at times :math:`t_0, t_0` and :math:`t_2` then
    the turn angle at time :math:`t_1` is defined by:

    .. math::

        d_1 = (x_1 - x_0, y_1 - y_0)

        d_2 = (x_2 - x_1, y_2 - y_1)

        ta = \arccos\left(\frac{d_1 \cdot d_2}{||d_1|| \cdot ||d_2||}\right)

    Here :math:`ta` is turn angle.

    """
    name = 'Turn Angle'

    def analyze(self, movement):
        """Extract turn angles from movement data.

        Get the turn angles of all individuals in movement data at every time
        step. Will result in an array of size [num_individuals, time_steps - 2]
        so that::

            turn_angle = results[i, k]

        Means that the ``i``-th individual had a turn angle of ``turn_angle``
        at the ``k``-th time step.

        Arguments
        ---------
        movement : :py:obj:`.Movement`
            Movement data to analyze

        Returns
        -------
        results : :py:obj:`array`
            Array of shape [num_individuals, time_steps - 1] containing the
            turn angles of all individuals at every time step.

        """
        data = movement.data
        times = movement.times

        dtimes = (times[1:] - times[:-1])[None, :, None]
        directions = (data[:, 1:, :] - data[:, :-1, :]) / dtimes
        complex_directions = directions[:, :, 0] + 1j * directions[:, :, 1]
        turn_angles = np.angle(
            complex_directions[:, 1:] /
            complex_directions[:, :-1])
        return turn_angles

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            num_individual=0,
            bins=20,
            width=None,
            cmap='Reds',
            alpha=0.8,
            log=True):
        """Plot distribution of turning angles.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot.
        figsize : tuple or list, optional
            Size of figure to create if no axes are provided.
        num_individual : int or tuple or list or array, optional
            Selection of individuals from which to draw turning angle
            information. If num_individual='all', all information will
            be plotted.
        bins : int, optional
            Number of bins to use in histogram of turn angle distribution.
            Defaults to 20.
        width : float, optional
            Width of bars in histogram. If none is given, width will be
            maximum possible before overlap.
        cmap : str, optional
            Colormap to use to assign colors to histogram bars. Defaults
            to 'Reds'.
        alpha : float, optional
            Alpha value of plot. Defaults to 0.8.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        if ax is None:
            _, ax = plt.subplots(
                    figsize=figsize, subplot_kw={"polar": True})

        if num_individual == 'all':
            tdata = self.results.ravel()
        elif isinstance(num_individual, int):
            tdata = self.results[num_individual, :]
        else:
            tdata = self.results[num_individual, :].ravel()

        range_ = (-np.pi, np.pi)
        histogram, _ = np.histogram(
                tdata, bins=bins, range=range_, normed=True)

        if width is None:
            width = (range_[1] - range_[0]) / bins
        theta = np.linspace(range_[0], range_[1], bins, endpoint=False)

        bars = ax.bar(theta, histogram, width=width, bottom=0.05)
        mins, maxs = histogram.min(), histogram.max()

        # Use custom colors and opacity
        cmap = get_cmap(cmap)
        for value, pbar in zip(histogram, bars):
            pbar.set_facecolor(
                    cmap((value - mins + 0.1) / (0.1 + maxs - mins)))
            pbar.set_alpha(alpha)
        ticks = np.linspace(0.05, 0.05 + histogram.max(), 4)
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.round(ticks - 0.05, 2))

        ax.set_title('Turn Angle distribution')
        return ax
