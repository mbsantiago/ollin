"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
import numpy as np  # pylint: disable=import-error
from scipy.stats import gaussian_kde  # pylint: disable=import-error
from tqdm import tqdm  # pylint: disable=import-error

from constants import (MAX_INDIVIDUALS, RANGE, ALPHA, BETA,
                       STEPS, DT, MIN_VELOCITY, MAX_POINTS,
                       GAMMA, DELTA)


def plot(data, num=10, show=False, time=365):
    """Use data to plot sample trajectories"""
    import matplotlib.pyplot as plt  # pylint: disable=import-error
    plotdata = data['data'][:num, :time]

    fig, axis = plt.subplots()
    for trajectory in plotdata:
        xcoords, ycoords = zip(*trajectory)
        axis.plot(xcoords, ycoords)
    rect = plt.Rectangle([0, 0], data['range'], data['range'], fill=False)
    axis.add_patch(rect)

    title = 'Rango: {} km;'
    title += ' Alpha: {};  Velocidad: {:2.3f} km/dia;'
    title = title.format(
        data['range'],
        data['alpha'],
        data['velocity'])

    plt.title(title)

    extra = 0.05 * data['range']
    plt.xlim(-extra, data['range'] + extra)
    plt.ylim(-extra, data['range'] + extra)

    plt.axis('off')
    if show:
        plt.show()
    return fig


# def make_initial_data(
#         num,
#         occupancy,
#         range,
#         velocity,
#         gamma=GAMMA,
#         delta=DELTA):
#     random_positions = np.random.uniform(0, range, [MAX_POINTS, 2])

#     dx = max(velocity, MIN_VELOCITY)
#     num_sides = int(np.ceil(range / dx))
#     grid = np.zeros([num_sides, num_sides])

#     points = []
#     for q in random_positions:
#         if np.mean(grid) > occupancy * GAMMA:
#             break
#         points.append(q)
#         indices = np.floor_divide(q, dx).astype(np.int)
#         grid[indices[0], indices[1]] = 1

#     kde = gaussian_kde(np.array(points).T, velocity * delta)
#     random_positions = np.minimum(np.maximum(kde.resample(num), 0), range)
#     return random_positions.T, kde


def make_initial_data(
        num,
        occupancy,
        range,
        velocity,
        gamma=GAMMA,
        delta=DELTA):
    random_positions = np.random.uniform(0, range, [num, 2])
    return random_positions, None


def plot_initial_conditions(data):
    import matplotlib.pyplot as plt  # pylint: disable=import-error
    fig, axis = plt.subplots()

    rect = plt.Rectangle([0, 0], data['range'], data['range'], fill=False)
    axis.add_patch(rect)

    X, Y = zip(*data['data'][:, 0, :])
    axis.scatter(X, Y)

    plt.axis('off')
    return fig


def plot_niche(data):
    import matplotlib.pyplot as plt  # pylint: disable=import-error
    kde = data['kde']
    range = data['range']
    xcoords, ycoords = np.mgrid[0: range: 100j, 0: range: 100j]
    positions = np.vstack([xcoords.ravel(), ycoords.ravel()])
    values = kde(positions).reshape(xcoords.shape)

    fig, axis = plt.subplots()
    plt.pcolormesh(values.T)
    plt.colorbar()
    plt.axis('off')
    return fig


def make_data(
        velocity,
        num=MAX_INDIVIDUALS,
        occupancy=1,
        steps=STEPS,
        alpha=ALPHA,
        gamma=GAMMA,
        delta=DELTA,
        range=RANGE):
    """Main function for movement data creation."""

    random_positions, kde = make_initial_data(
        num, occupancy, range, velocity, gamma=gamma)

    mean = (velocity * (alpha - 1)/alpha)
    stack = [random_positions]

    iterator = xrange(steps - 1)

    for _ in iterator:
        random_angles = np.random.uniform(0, 2 * np.pi, [num])
        random_directions = np.stack(
            [np.cos(random_angles), np.sin(random_angles)], axis=-1)

        # exponents = 1 + kde(random_positions.T)*alpha
        exponents = alpha
        random_magnitudes = mean / np.power(
            (1 - np.random.rand(num)), 1/exponents)
        random_directions *= random_magnitudes[:, None]

        tmp1 = random_positions + random_directions
        tmp2 = np.mod(tmp1, 2 * range)
        reflections = np.greater(tmp2, range)
        tmp3 = (1 - reflections) * np.mod(tmp2, range)
        tmp4 = reflections * np.mod(-tmp2, range)
        random_positions = tmp3 + tmp4

        stack.append(random_positions)
    data = {
        'data': np.stack(stack, 1),
        'alpha': alpha,
        'range': range,
        'velocity': velocity,
        'kde': kde,
    }
    return data


def home_range_to_velocity(home_range, beta=BETA, dt=DT):
    return beta * np.sqrt(home_range) / float(dt)
