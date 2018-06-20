# pylint: disable=unbalanced-tuple-unpacking
import numpy as np  # pylint: disable=import-error


def make_grid(movement_data):
    array = movement_data.data

    range_ = movement_data.range
    resolution = movement_data.resolution
    num_sides = int(np.ceil(range_ / resolution))
    resolution = range_ / float(num_sides)
    num_trials = array.shape[0]
    steps = array.shape[1]

    space = np.zeros([num_trials, num_sides, num_sides])
    indices = np.true_divide(array, resolution).astype(np.int)

    xcoords = np.linspace(
        0, num_trials,
        num_trials * steps,
        endpoint=False).astype(np.int).reshape([-1, 1])
    ycoords, zcoords = np.split(indices.reshape([-1, 2]), 2, -1)
    space[xcoords, ycoords, zcoords] = resolution ** 2

    return space


def plot(grid, t=0, transpose=True):
    import matplotlib.pyplot as plt  # pylint: disable=import-error
    fig, axis = plt.subplots()
    array = grid[t, :, :]
    if transpose:
        array = array.T
    axis.pcolormesh(array)
    return fig


def calculate(movement_data):
    grid = make_grid(movement_data)
    dx = max(movement_data.velocity, MIN_VELOCITY)
    areas = np.sum(grid, axis=(1, 2)) * dx**2
    return np.mean(areas)
