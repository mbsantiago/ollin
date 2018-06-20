import numpy as np  # pylint: disable=import-error
from numba import jit, float64, int64

from utils import occupancy_resolution


class Occupancy(object):
    def __init__(self, movement_data, num_trials=1):
        self.movement_data = movement_data
        self.steps = movement_data.steps
        self.num_trials = num_trials
        self.resolution = occupancy_resolution(movement_data.home_range)

        self.grid = make_grid(
            self.movement_data,
            self.resolution,
            num_trials=num_trials)

        self.occupation_nums = np.sum(self.grid, axis=1)
        self.occupations = np.mean(
            self.occupation_nums / float(self.steps),
            axis=(1, 2))
        self.mean_occupation = np.mean(self.occupations)

    def plot(
            self,
            include=None,
            ax=None,
            show=0,
            lev=0.2,
            transpose=False,
            alpha=0.3,
            **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if include is None:
            include = [
                'rectangle', 'niche', 'occupation', 'occupation_contour']

        if 'occupation' in include:
            if isinstance(show, int):
                grid = self.occupation_nums[show]
            elif show == 'mean':
                grid = np.mean(self.occupation_nums, axis=0)

            grid = grid / float(self.steps)

            range_ = self.movement_data.range
            h, w = grid.shape
            xcoord, ycoord = np.meshgrid(
                np.linspace(0, range_[0], h),
                np.linspace(0, range_[1], w))
            plt.pcolormesh(xcoord, ycoord, grid.T, cmap='Blues', alpha=alpha, vmax=1.0, vmin=0.0)
            plt.colorbar()

            if 'occupation_contour' in include:
                mask = (grid >= lev)
                ax.contour(xcoord, ycoord, mask.T, levels=[0.5], cmap='Blues')


        self.movement_data.plot(
                include, ax=ax, transpose=transpose, **kwargs)

        return ax


# def make_grid(movement_data, num_trials=1):
    # array = movement_data.data
    # range = movement_data.range
    # dx = max(movement_data.velocity, MIN_VELOCITY)
    # num_sides = int(np.ceil(range / dx))

    # runs = array.shape[0]
    # steps = array.shape[1]

    # space = np.zeros([num_trials, num_sides, num_sides])
    # indices = np.floor_divide(array, dx).astype(np.int)

    # xcoords = np.linspace(
        # 0, num_trials,
        # runs * steps,
        # endpoint=False).astype(np.int).reshape([-1, 1])
    # ycoords, zcoords = np.split(
        # indices.reshape([-1, 2]), 2, -1)  # pylint: disable=unbalanced-tuple-unpacking  noqa
    # space[xcoords, ycoords, zcoords] = 1

    # return space


@jit(float64[:, :, :, :](int64, float64[:, :, :], float64[0], int64, float64, int64), nopython=True)
def _make_grid(steps, array, range, num, resolution, num_trials):

    num_sides_x = int(np.ceil(range[0] / resolution))
    num_sides_y = int(np.ceil(range[1] / resolution))

    space = np.zeros((num_trials, steps, num_sides_x, num_sides_y))
    indices = np.floor_divide(array, resolution).astype(np.int64)

    for s in xrange(steps):
        for i in xrange(num):
            trial = i % num_trials
            x, y = indices[i, s]
            space[trial, s, x, y] = 1
    return space


def make_grid(mov, resolution, num_trials=1):
    return _make_grid(mov.steps, mov.data, mov.range, mov.num, resolution, num_trials)


def make_data(movement_data, num_trials=1):
    data = Occupancy(movement_data, num_trials=num_trials)
    return data

