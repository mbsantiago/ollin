import numpy as np  # pylint: disable=import-error

from constants import MIN_VELOCITY, GAMMA, OMEGA, RANGE, TAU
from numba import jit, float64, int64


class Occupancy(object):
    def __init__(self, movement_data, num_trials=1):
        self.movement_data = movement_data
        self.num_trials = num_trials

        self.grid = make_grid(self.movement_data, num_trials=num_trials)
        self.occupations = np.mean(self.grid, (1, 2))
        self.mean_occupation = np.mean(self.occupations)

    def plot(
            self,
            include=None,
            axis=None,
            show=0,
            lev=0.5,
            transpose=False,
            **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        if axis is None:
            fig, axis = plt.subplots()

        if include is None:
            include = [
                'rectangle', 'niche', 'occupation', 'occupation_contour']

        self.movement_data.plot(
            include, axis=axis, transpose=transpose, **kwargs)

        if 'occupation' in include:
            if isinstance(show, int):
                grid = self.grid[show]
            elif show == 'mean':
                grid = np.mean(self.grid, axis=0)

            init_cond = self.movement_data.initial_data
            size = init_cond.kde_approximation.shape[0]-1
            range_ = self.movement_data.range
            xcoord, ycoord = np.meshgrid(
                np.linspace(0, range_, size),
                np.linspace(0, range_, size))
            axis.pcolormesh(xcoord, ycoord, grid, cmap='Blues', alpha=0.2)

            if 'occupation_contour' in include:
                mask = (grid >= lev)
                axis.contour(xcoord, ycoord, mask, levels=[0.5], cmap='Blues')

        return axis


def make_grid(movement_data, num_trials=1):
    array = movement_data.data
    range = movement_data.range
    dx = max(movement_data.velocity, MIN_VELOCITY)
    num_sides = int(np.ceil(range / dx))

    runs = array.shape[0]
    steps = array.shape[1]

    space = np.zeros([num_trials, num_sides, num_sides])
    indices = np.floor_divide(array, dx).astype(np.int)

    xcoords = np.linspace(
        0, num_trials,
        runs * steps,
        endpoint=False).astype(np.int).reshape([-1, 1])
    ycoords, zcoords = np.split(
        indices.reshape([-1, 2]), 2, -1)  # pylint: disable=unbalanced-tuple-unpacking  noqa
    space[xcoords, ycoords, zcoords] = 1

    return space


def make_data(movement_data, num_trials=1):
    data = Occupancy(movement_data, num_trials=num_trials)
    return data


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def occupancy_to_num(occupancy, home_range, gamma=GAMMA, omega=OMEGA, range=RANGE, tau=TAU):
    num = gamma * (occupancy**tau) * range**2 / (home_range)**omega
    return int(num)


def plot(grid, t=0, transpose=True):
    import matplotlib.pyplot as plt  # pylint: disable=import-error
    fig, axis = plt.subplots()
    if transpose:
        array = grid[t, :, :].T
    else:
        array = grid[t, :, :]
    axis.pcolormesh(array)
    return fig
