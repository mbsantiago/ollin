import numpy as np  # pylint: disable=import-error

from constants import MIN_VELOCITY


def make_grid(movement_data, num_trials=1):
    array = movement_data['data']
    range = movement_data['range']
    dx = max(movement_data['velocity'], MIN_VELOCITY)
    num_sides = int(np.ceil(range / dx))

    runs = array.shape[0]
    steps = array.shape[1]

    space = np.zeros([num_trials, num_sides, num_sides])
    indices = np.true_divide(array, dx).astype(np.int)

    xcoords = np.linspace(
        0, num_trials,
        runs * steps,
        endpoint=False).astype(np.int).reshape([-1, 1])
    ycoords, zcoords = np.split(indices.reshape([-1, 2]), 2, -1)
    space[xcoords, ycoords, zcoords] = 1

    return space


def plot(grid, t=0, transpose=True):
    import matplotlib.pyplot as plt  # pylint: disable=import-error
    fig, axis = plt.subplots()
    if transpose:
        array = grid[t, :, :].T
    else:
        array = grid[t, :, :]
    axis.pcolormesh(array)
    return fig
