# pylint: disable=unbalanced-tuple-unpacking
import numpy as np  # pylint: disable=import-error

from .utils import home_range_resolution


class HomeRange(object):
    def __init__(self, movement_data):
        self.movement_data = movement_data
        self.grid = self.make_grid()
        self.home_ranges = self.grid.sum(axis=(1, 2))
        self.mean_home_range = self.home_ranges.mean()

    def make_grid(self):
        mov_data = self.movement_data
        num_experiments, num, steps, _ = mov_data.data.shape
        array = mov_data.data.reshape([-1, steps, 2])
        range_ = mov_data.initial_conditions.range
        resolution = home_range_resolution(mov_data.velocity)
        grid = _make_grid(array, range_, resolution)
        return grid

    def plot(self, ax=None, n_individual=0, include=None, hr_cmap='Blues', **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        from matplotlib.colors import ListedColormap
        from matplotlib.cm import get_cmap

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if include is None:
            include = [
                    'niche',
                    'rectangle',
                    'home_range']
        self.movement_data.plot(ax=ax, include=include, **kwargs)

        if 'home_range' in include:
            is_list = False
            if isinstance(n_individual, (list, tuple, np.ndarray)):
                home_range = self.grid[np.array(n_individual)]
                is_list = True
            elif n_individual == 'mean':
                home_range = self.grid.mean(axis=0)
            else:
                home_range = self.grid[n_individual]

            _, sizex, sizey = self.grid.shape
            range_ = self.movement_data.initial_conditions.range
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


def _make_grid(array, range, resolution):
    num_sides_x = int(np.ceil(range[0] / resolution))
    num_sides_y = int(np.ceil(range[1] / resolution))

    num_trials = array.shape[0]
    steps = array.shape[1]

    grid = np.zeros([num_trials, num_sides_x, num_sides_y])
    indices = np.true_divide(array, resolution).astype(np.int)

    xcoords = np.linspace(
        0, num_trials,
        num_trials * steps,
        endpoint=False).astype(np.int).reshape([-1, 1])
    ycoords, zcoords = np.split(indices.reshape([-1, 2]), 2, -1)
    grid[xcoords, ycoords, zcoords] = resolution ** 2
    return grid
