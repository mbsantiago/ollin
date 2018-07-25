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

   See :py:func:`ollin.core.utils.occupancy_resolution`

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
import numpy as np
from numba import jit, float64

from utils import occupancy_resolution


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
    movement : :py:obj:`ollin.MovementData`
        Movement data for which to calculate occupancy.
    steps : int
        Number of movement steps contained in movement data.
    resolution : float
        Spatial resolution (in Km) for site discretization.
    grid : array
        Array of shape [time_steps, x, y] where [x, y] is the
        size of the discretized site. Holds cell ocupancy at
        each time step.
    """

    def __init__(self, movement, grid=None, resolution=None):
        self.movement = movement
        self.steps = movement.steps

        if resolution is None:
            resolution = occupancy_resolution(movement.home_range)
        self.resolution = resolution

        if grid is None:
            grid = make_grid(self.movement, self.resolution)
        self.grid = grid

    def get_occupancy_nums(self):
        occupancy_nums = np.sum(self.grid, axis=0)
        return occupancy_nums

    def get_occupancy(self):
        occupancy_nums = self.get_occupancy_nums()
        occupancy = np.mean(occupancy_nums / float(self.steps))
        return occupancy

    def plot(
            self,
            include=None,
            ax=None,
            occupancy_cmap='Blues',
            occupancy_level=0.2,
            occupancy_alpha=0.3,
            **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if include is None:
            include = [
                'rectangle', 'niche', 'occupancy', 'occupancy_contour']

        if 'occupancy' in include:
            grid = self.get_occupancy_nums()
            grid = grid / float(self.steps)

            range_ = self.movement_data.site.range
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

        self.movement_data.plot(include=include, ax=ax, **kwargs)

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
    indices = np.floor_divide(array, resolution).astype(np.int64)

    for s in xrange(steps):
        for i in xrange(num):
            x, y = indices[i, s]
            space[s, x, y] = 1
    return space


def make_grid(mov, resolution):
    return _make_grid(mov.data, mov.site.range, resolution)
