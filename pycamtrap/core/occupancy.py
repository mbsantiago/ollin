import numpy as np  # pylint: disable=import-error
from numba import jit, float64

from utils import occupancy_resolution


class Occupancy(object):
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
