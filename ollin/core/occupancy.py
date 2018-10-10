"""Module for occupancy class and calculation.

Occupancy can be loosely defined as proportion of area occupied by some target
species. Unfortunately this definition is not rigorous enough to define a
single concept. Hence definitions have proliferated in the literature, mainly
guided by study restriction and convenience.

Three main issues arise when trying to clearly define occupancy:

    1. The definition of space being occupied by some species: In some cases,
       where niches are sharply delimited by geographical features, such as
       ponds, occupancy refers to the proportion of such areas occupied by
       some target species at an instant in time. But when movement is not
       restrained by clear boundaries occupancy is harder to
       define. One approach is defining occupancy as the proportion of points
       at which individuals of the target species are observed, having
       preselected the points as being "representative". This seems to be an
       ad hoc definition for camera trap studies. Finally when movement data
       is available occupancy can be defined, more faithful to the original
       concept, as the area visited by the individuals of the target species.
       But this also introduces the following two problems:
    2. Calculation of area visited by an individual usually starts by some sort
       discretization of space, and is done by counting the number of cells of
       discretized space that individuals touch along its movement. This
       implies that there will be some dependence of the final occupancy value
       on the method of space discretization.
    3. Finally, when occupancy is calculated from movement data, there must be
       a choice of time span for which to calculate occupancy. Occupancy values
       will be sensitive to choice of time span, or could even be result in a
       useless feature if, for example, individuals of species fully roam the
       available land given enough time, even though they usually take a small
       proportion of land area at smaller time scales.

Due to the nature of the data generated in movement simulations we have chosen
to define occupancy as the result of the following calculations:

1. Select some definite time span, say 90 days of movement.
2. Use some pre defined function to obtain spatial resolution from home range
   data and discretize study site with such resolution. Symbolically::

      resolution = f(home_range, parameters)

   See :py:func:`.occupancy_resolution`

3. For each time step mark as visited all cells that have some individual of
   the target species contained in it.
4. For each cell of discretized space, calculate the proportion of times the
   cell was visited.
5. Occupancy is defined as the average of this proportion of times cells where
   visited.

We have chosen to use proportion of occupancy, or rate of occupancy, since
this value is less sensitive to changes in time span selected for occupancy
calculation. In simulated situations it will always converge to some specific
value as time span grows.

All values used in the occupancy calculation, such as the grid of visited cells
per time step, number of times a cell was visited and others, are stored within
a :py:obj:`Occupancy` object.

"""
from __future__ import division

from six.moves import xrange
import numpy as np
from numba import jit, float64

from .utils import occupancy_resolution


class Occupancy(object):
    """Occupancy information and calculation class.

    Occupancy can be calculated from movement data in the following way:

    1. Select some spatial resolution to use in space discretization.
    2. Discretize site to obtain n x m cells.
    3. Initialize an array of shape [time_steps, n, m] to zeros.
    4. Set ::

          array[t, i, j] = 1

       if some individual is in cell (i,j) at time step t.
    5. Occupancy is the average of this array.

    Attributes
    ----------
    movement : :py:obj:`.MovementData`
        Movement data for which to calculate occupancy.
    steps : int
        Number of movement steps contained in movement data.
    resolution : float
        Spatial resolution (in Km) for site discretization.
    grid : array
        Array of shape [time_steps, x, y] where [x, y] is the
        size of the discretized site. Holds cell occupancy at
        each time step.
    cell_occupancy : array
        Array of shape [x, y] where [x, y] is the size of the
        discretized site. Holds cell occupancy.
    occupancy : float
        Occupancy measure.
    """

    def __init__(self, movement, resolution=None):
        """Construct Occupancy object.

        Arguments
        ---------
        movement : :py:obj:`.MovementData`
            Movement data to be analized.
        resolution : float, optional
            Resolution for space discretization. If none is given,
            resolution will be calculated from home_range data
            stored in the movement data.

        """
        self.movement = movement
        self.steps = movement.steps

        if resolution is None:
            resolution = occupancy_resolution(movement.home_range)
        self.resolution = resolution

        range_ = movement.site.range
        self.grid = _make_grid(movement.data, range_, self.resolution)
        self.cell_occupancy = self.grid.sum(axis=0) / self.steps
        self.occupancy = self.cell_occupancy.mean()

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            include=None,
            occupancy_cmap='Blues',
            occupancy_level=0.2,
            occupancy_alpha=0.3,
            **kwargs):
        """Plot discretized space occupancy information.

        Occupancy plotting adds the following optional components to the
        plot:

        1. "occupancy":
            If present in include list, cells of discretized space will be
            shown with a color representing the rate of occupation of such
            cell. Colors will be selected using a color map. See
            :py:mod:`matplotlib.cm` to see all colormap options.
        2. "occupancy_contour":
            If present in include list, the border of the region containing all
            cells with occupancy higher than some threshold will be plotted.

        All other components in the include list will be passed down to the
        MovementData plotting method. See :py:meth:`.MovementData.plot`
        for all plot components defined at that level.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot occupancy information.
        figsize : list or tuple, optional
            Size of figure to create if no axes object was given. Defaults to
            (10, 10).
        include : list or tuple, optional
            List of components to add to the plot. Components list will be
            passed to the Site object to add the corresponding components.
        occupancy_cmap : str, optional
            Colormap to use to codify occupancy level to color. Defaults to
            'Blues'.
        occupancy_level : float, optional
            Threshold at which to draw boundary for occupancy contour.
        occupancy_alpha : float, optional
            Alpha value of cell colors.
        kwargs : dict, optional
            All other keyword arguments will be passed to the MovementData's
            plotting method.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Return axes for further plotting.

        """
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if include is None:
            include = [
                'rectangle',
                'niche_boundary',
                'occupancy',
                'occupancy_contour']

        self.movement.plot(include=include, ax=ax, **kwargs)

        if 'occupancy' in include:
            grid = self.cell_occupancy

            range_ = self.movement.site.range
            h, w = grid.shape
            xcoord, ycoord = np.meshgrid(
                np.linspace(0, range_[0], h),
                np.linspace(0, range_[1], w))
            cm = ax.pcolormesh(
                xcoord,
                ycoord,
                grid.T,
                cmap=occupancy_cmap,
                alpha=occupancy_alpha,
                vmax=1.0,
                vmin=0.0)
            plt.colorbar(cm, ax=ax)

            if 'occupancy_contour' in include:
                mask = (grid >= occupancy_level)
                ax.contour(xcoord, ycoord, mask.T, levels=[0.5], cmap='Blues')

        return ax

    def plot_occupancy_timeseries(
            self,
            ax=None,
            figsize=(10, 10),
            color='red',
            label=''):
        """Plot occupancy values at different times.

        Occupancy is defined as the mean rate of occupation. The rate of
        occupation is calculated over some timespan. This method calculates
        occupancy at all possible timespans and plots the resulting timeseries
        for visual comparison.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes in which to plot timeseries. If not provided new axes will be
            created.
        figsize : tuple or list, optional
            Size of figure to be created if no axes where provided.
        color : str, optional
            Name of color to use for plot.
        label : str, optional
            Label of plot.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes of plot, returned for further plotting.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        times = self.movement.times
        occupancies = [self.grid[:n].mean() for n in xrange(2, times.size)]

        ax.plot(times[2:], occupancies, color=color, label=label)
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Occupancy')
        ax.set_ylim(0, 1)

        return ax


@jit(
    float64[:, :, :](
        float64[:, :, :],
        float64[:],
        float64),
    nopython=True)
def _make_grid(array, range, resolution):
    num_sides_x = int(np.ceil(range[0] / resolution))
    num_sides_y = int(np.ceil(range[1] / resolution))

    num, steps, _ = array.shape

    space = np.zeros((steps, num_sides_x, num_sides_y))

    denominator = np.zeros((2,))
    denominator[0] = range[0] / num_sides_x
    denominator[1] = range[1] / num_sides_y
    indices = np.floor_divide(array, denominator).astype(np.int64)

    disc_range = np.zeros((2,), dtype=np.int64)
    disc_range[0] = num_sides_x - 1
    disc_range[1] = num_sides_y - 1
    indices = np.minimum(indices, disc_range)

    for s in xrange(steps):
        for i in xrange(num):
            x, y = indices[i, s]
            space[s, x, y] = 1
    return space
