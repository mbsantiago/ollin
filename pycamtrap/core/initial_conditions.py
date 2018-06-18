import numpy as np  # pylint: disable=import-error
from scipy.stats import gaussian_kde  # pylint: disable=import-error

from movement import home_range_to_velocity
from constants import (MIN_CLUSTERS, MAX_CLUSTERS, MIN_NEIGHBORS,
                       MAX_NEIGHBORS, MIN_VELOCITY, MAX_ITERS)


PLOT_OPTIONS = [
    'rectangle', 'kde_points', 'heatmap', 'niche',
    'initial_positions'
]


class InitialCondition(object):
    def __init__(self, range, occupancy, num, home_range):
        self.range = range
        self.occupancy = occupancy
        self.num = num
        self.home_range = home_range
        self.velocity = home_range_to_velocity(home_range)
        self.resolution = home_range_to_resolution(home_range)

        self.kde_points = self.make_cluster_points()
        self.kde, self.kde_approximation = self.make_kde()
        self.initial_points = self.make_initial_points()

    def make_cluster_points(self):
        return make_cluster_points(self.range)

    def make_kde(self):
        return make_kde(
            self.kde_points, self.occupancy, resolution=self.resolution)

    def make_initial_points(self):
        initial_points = self.kde.resample(size=self.num)
        initial_points = np.maximum(
            np.minimum(initial_points, self.range - 0.01), 0.01)
        return initial_points.T

    def plot(self, include=None, ax=None, transpose=False):
        import matplotlib.pyplot as plt  # pylint: disable=import-error

        if include is None:
            include = [
                    'heatmap',
                    'initial_positions',
                    'niche',
                    'rectangle']

        if ax is None:
            fig, ax = plt.subplots()

        if 'rectangle' in include:
            rect = plt.Rectangle([0, 0], self.range, self.range, fill=False)
            ax.add_patch(rect)

        if 'heatmap' in include or 'niche' in include:
            heatmap = self.kde_approximation
            size = heatmap.shape[0]
            if transpose:
                heatmap = heatmap.T
            Xrange, Yrange = np.meshgrid(
                    np.linspace(0, self.range, size),
                    np.linspace(0, self.range, size))

        if 'heatmap' in include:
            ax.pcolormesh(Xrange, Yrange, heatmap, cmap='Reds')

        if 'niche' in include:
            zone = occupation_space_from_approximation(heatmap)
            ax.contour(Xrange, Yrange, zone, levels=0.5)

        if 'kde_points' in include:
            X, Y = zip(*self.kde_points.T)
            ax.scatter(X, Y, label='KDE Points')

        if 'initial_positions' in include:
            X, Y = zip(*self.initial_points)
            ax.scatter(X, Y, label='initial_positions', c='black', s=2)

        ax.set_xticks(np.linspace(0, self.range, 2))
        ax.set_yticks(np.linspace(0, self.range, 2))

        ax.set_xlim(0, self.range)
        ax.set_ylim(0, self.range)

        return ax


def make_cluster_points(range):
    n_clusters = np.random.randint(MIN_CLUSTERS, MAX_CLUSTERS)

    cluster_centers = np.random.uniform(0, range, size=[n_clusters, 2])

    points = []
    for k in xrange(n_clusters):
        n_neighbors = np.random.randint(MIN_NEIGHBORS, MAX_NEIGHBORS)
        centered_points = np.random.normal(size=[n_neighbors, 2])
        variances = np.random.normal(size=[2, 2])
        sheared_points = np.tensordot(
            variances, centered_points, (1, 1))
        shifted_points = sheared_points + cluster_centers[k][:, None]
        points.append(shifted_points)
    point_data = {
        'points': np.concatenate(points, 1),
        'range': range,
        'n_clusters': n_clusters
    }
    return point_data


def make_density_approximation(density, range, resolution=1.0):
    num_sides = int(np.ceil(range / float(resolution)))
    xcoords, ycoords = np.meshgrid(
        np.linspace(0, range, num_sides + 1, endpoint=False),
        np.linspace(0, range, num_sides + 1, endpoint=False))
    points = np.stack([xcoords.ravel(), ycoords.ravel()], 0)
    values = density(points).reshape(xcoords.shape)
    return values


def occupation_space_from_approximation(aprox):
    return (aprox >= .25 * aprox.mean())


def make_kde(points, t_occupancy, resolution=1.0, epsilon=0.05):

    max_bw = points['range'] / 2
    min_bw = 0.01

    mid_bw = (max_bw + min_bw) / 2

    kde = gaussian_kde(points['points'], mid_bw)

    counter = 0
    while True:
        occ, kde_approx = calculate_occupancy(
            kde, points['range'], resolution=resolution)

        err = abs(occ - t_occupancy)
        if err < epsilon:
            break
        elif occ < t_occupancy:
            min_bw = mid_bw
            mid_bw = (max_bw + min_bw) / 2
            kde.set_bandwidth(mid_bw)
        else:
            max_bw = mid_bw
            mid_bw = (max_bw + min_bw) / 2
            kde.set_bandwidth(mid_bw)

        counter += 1
        if counter == MAX_ITERS:
            break

    return kde, kde_approx


def calculate_occupancy(density, range, resolution=1.0):
    density_approximation = make_density_approximation(
        density, range, resolution=resolution)
    occupation_space = occupation_space_from_approximation(
        density_approximation)
    return np.mean(occupation_space), density_approximation


def make_data(range, occupancy, num, velocity):
    initial_data = InitialCondition(range, occupancy, num, velocity)
    return initial_data

def home_range_to_resolution(home_range):
    return np.sqrt(home_range)
