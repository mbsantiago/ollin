"""Module for Home Range calculation.

Even though home range is a fuzzy concept with several competing definitions,
and estimation methods, in the literature, in this package home range is
defined as:

    **Area occupied by an individual in the course of some fixed time span.**

Home range calculation follows the next pipeline:

1. Inputs are Movement objects that contain position history of all
   simulated individuals for some number of steps
2. Space is discretized using some resolution that only depends on the mean
   velocity of the simulation. See :py:func:`.home_range_resolution` to see the
   functional relationship between mean velocity and space resolution
3. For each individual, all pixels in discretized space occupied by the
   individual are counted, and an array with all presence-absence information
   is calculated
4. Each individual is assigned the total area of discretized space occupied
   along its movement

"""

import numpy as np

from .utils import home_range_resolution


class HomeRange(object):
    """Home Range class for storing home range values.

    Movement data is processed into a grid of shape [num_individuals, x, y],
    where x and y are the sizes of discretized space and where::

        grid[i, x, y] = 1

    means that the (x, y) pixel was occupied by the i-th individual.

    Attributes
    ----------
    movement : :py:obj:`.Movement`
        Underlying movement data.
    grid : array
        Array of shape [num_individuals, x, y] with presence-absence
        information.
    resolution : float
        Spatial resolution for space discretization.
    home_ranges : array
        Array of shape [num_individuals] holding the home range in Km^2 for
        each individual.
    mean_home_range : float
        Average home range.

    """

    def __init__(self, movement):
        """Construct Home Range data.

        Arguments
        ---------
        movement : :py:obj:`.Movement`
            Movement data for which to calculate home ranges.

        """
        self.movement = movement
        self.resolution = home_range_resolution(movement.velocity)
        self.grid = self._make_grid()
        self.home_ranges = self.grid.sum(axis=(1, 2))
        self.mean_home_range = self.home_ranges.mean()

    def _make_grid(self):
        mov_data = self.movement
        num, steps, _ = mov_data.data.shape
        range_ = mov_data.site.range
        array = mov_data.data

        grid = make_grid(array, range_, self.resolution)
        return grid

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            include=None,
            n_individual=0,
            hr_cmap='Blues',
            **kwargs):
        """Plot home range.

        Plots all pixels occupied by an individual in discretized space.

        Home Range plotting adds the following optional components to the plot:
            1. "home_range:
                If present in include list, all pixels occupied by some subset
                of individuals will be colored. Diferent colors will be chosen
                for different individuals based on some colormap.

        All other components in the include list will be passed down to the
        Movement plotting method. See
        :py:meth:`.Movement.plot` for all plot
        components defined at that level.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot detection information.
        figsize : list or tuple, optional
            Size of figure to create if no axes object was given.
        include : list or tuple, optional
            List of components to plot. Components list will be passed first
            to the Movement Data object to add the corresponding
            components. Then components corresponding to Movement
            included in the list will be plotted.
        n_individual : int or list or tuple or array or str, optional
            If int will plot the home range of the corresponding individual.
            If it will plot the home range of all individuals in
            list/tuple/array. Otherwise it must be n_idividual='all', in which
            case it will plot all home ranges.
        hr_cmap : str, optional
            Colormap to use for home ranges, see :py:mod:`matplotlib.cm` to
            see all options. Defaults to 'Blues'.
        kwargs : dict, optional
            All other keyworded arguments will be passed to the Movement
            plotting method.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Returns axes for further plotting.

        """
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        from matplotlib.colors import ListedColormap
        from matplotlib.cm import get_cmap

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if include is None:
            include = [
                    'niche_boundary',
                    'rectangle',
                    'home_range']
        self.movement.plot(ax=ax, include=include, **kwargs)

        if 'home_range' in include:
            is_list = False
            if n_individual == 'all':
                n_individual = range(self.movement.num)
            if isinstance(n_individual, (list, tuple, np.ndarray)):
                home_range = self.grid[np.array(n_individual)]
                is_list = True
            elif n_individual == 'mean':
                home_range = self.grid.mean(axis=0)
            else:
                home_range = self.grid[n_individual]

            _, sizex, sizey = self.grid.shape
            range_ = self.movement.site.range
            rangex, rangey = np.meshgrid(
                np.linspace(0, range_[0], sizex),
                np.linspace(0, range_[1], sizey))

            cmap = get_cmap(hr_cmap)

            if is_list:
                for hr in home_range:
                    color = cmap(np.random.rand())
                    cMap = ListedColormap([(0.0, 0.0, 0.0, 0.0), color])
                    ax.pcolormesh(rangex, rangey, hr.T, cmap=cMap)
            else:
                ax.pcolormesh(rangex, rangey, home_range.T, cmap=cmap)
        return ax


def make_grid(array, range, resolution):
    """Make and return presence absence grid in discretized space.

    Given an array of shape [num, steps, 2] which encodes individual positions
    at different time steps in a rectangular area of dimensions=range, and a
    resolution for space discretization, calculates an array of presence and
    absence data of shape [num, x, y], where::

        array[i, x, y] = 1

    means that the i-th individual passed though the (x, y) pixel at some
    point.

    Arguments
    ---------
    array : array
        Array of shape [num, steps, 2] of individuals positions.
    range : tuple or list or array
        Two dimensions of rectangular arena.
    resolution : float
        Resolution of space discretization.

    Returns
    -------
    grid : array
        Array of shape [num, x, y] where x ~ range[0] / resolution and
        y ~ range[1] / resolution.

    """
    num_sides_x = int(np.ceil(range[0] / resolution))
    num_sides_y = int(np.ceil(range[1] / resolution))

    num, steps, _ = array.shape

    grid = np.zeros([num, num_sides_x, num_sides_y])
    indices = np.true_divide(array, resolution).astype(np.int)

    nums = np.linspace(
        0, num,
        num * steps,
        endpoint=False).astype(np.int).reshape([-1, 1])
    xcoords, ycoords = np.split(indices.reshape([-1, 2]), 2, -1)

    ycoords = np.minimum(ycoords, num_sides_y - 1)
    xcoords = np.minimum(xcoords, num_sides_x - 1)

    grid[nums, xcoords, ycoords] = resolution ** 2
    return grid
