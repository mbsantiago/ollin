"""Module for creation of movement data

Movement of individuals is assumed to happen in a square space.
TODO docstring
"""
import numpy as np  # pylint: disable=import-error

from constants import (MAX_INDIVIDUALS, RANGE, ALPHA, BETA, STEPS, DT)


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


def make_data(
        velocity,
        num=MAX_INDIVIDUALS,
        steps=STEPS,
        alpha=ALPHA,
        range=RANGE):
    """Main function for movement creation."""

    random_positions = np.random.uniform(0, range, [num, 2])

    mean = (velocity * (alpha - 1)/alpha)
    stack = [random_positions]
    for _ in xrange(steps - 1):
        random_angles = np.random.uniform(0, 2 * np.pi, [num])
        random_directions = np.stack(
            [np.cos(random_angles), np.sin(random_angles)], axis=-1)
        random_magnitudes = mean * np.random.pareto(alpha, [num])
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
    }
    return data


def home_range_to_velocity(home_range, beta=BETA, dt=DT):
    return beta * np.sqrt(home_range) / float(dt)
