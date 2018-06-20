import numpy as np  # pylint: disable=import-error
from scipy.stats import gaussian_kde  # pylint: disable=import-error

from utils import home_range_to_velocity, home_range_resolution
from constants import handle_parameters


PLOT_OPTIONS = [
    'rectangle', 'kde_points', 'heatmap', 'niche',
    'initial_positions'
]


class InitialCondition(object):
    def __init__(
            self,
            occupancy,
            home_range,
            range=None,
            kde_points=None,
            parameters=None):

        if parameters is None:
            parameters = {}
        parameters = handle_parameters(parameters)
        self.parameters = parameters

        if range is None:
            range = parameters['RANGE']

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)
        self.range = range.astype(np.float64)

        self.occupancy = occupancy
        self.home_range = home_range

        self.velocity = home_range_to_velocity(
            home_range, parameters=parameters)
        self.home_range_resolution = home_range_resolution(
            self.velocity, parameters=parameters)

        if kde_points is None:
            kde_points = self.make_cluster_points()
        self.kde_points = kde_points
        self.kde, self.kde_approximation = self.make_kde()

    def make_cluster_points(self):
        return make_cluster_points(self.range, parameters=self.parameters)

    def make_kde(self):
        kde = make_kde(
            self.kde_points,
            self.occupancy,
            resolution=self.home_range_resolution,
            parameters=self.parameters)
        return kde

    def sample(self, num):
        initial_points = self.kde.resample(size=num)
        initial_points = np.maximum(
                np.minimum(initial_points, self.range[:, None] - 0.01), 0.01)
        return initial_points.T

    @classmethod
    def from_points(
            cls,
            points,
            range='auto',
            home_range='auto',
            occupancy='auto'):
        pass

    def plot(self, include=None, ax=None, transpose=False):
        import matplotlib.pyplot as plt  # pylint: disable=import-error

        if include is None:
            include = [
                    'heatmap',
                    'niche',
                    'rectangle']

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if 'rectangle' in include:
            rect = plt.Rectangle([0, 0], self.range[0], self.range[1], fill=False)
            ax.add_patch(rect)

        if 'heatmap' in include or 'niche' in include:
            heatmap = self.kde_approximation
            sizex = heatmap.shape[0]
            sizey = heatmap.shape[1]
            if transpose:
                heatmap = heatmap.T
            Xrange, Yrange = np.meshgrid(
                    np.linspace(0, self.range[0], sizex),
                    np.linspace(0, self.range[1], sizey))

        if 'heatmap' in include:
            ax.pcolormesh(Xrange, Yrange, heatmap.T, cmap='Reds')

        if 'niche' in include:
            zone = occupation_space_from_approximation(heatmap)
            ax.contour(Xrange, Yrange, zone.T, levels=0.5)

        if 'kde_points' in include:
            X, Y = zip(*self.kde_points.T)
            ax.scatter(X, Y, label='KDE Points')

        ax.set_xticks(np.linspace(0, self.range[0], 2))
        ax.set_yticks(np.linspace(0, self.range[1], 2))

        ax.set_xlim(0, self.range[0])
        ax.set_ylim(0, self.range[1])

        return ax


def make_cluster_points(range, parameters=None):
    min_clusters = parameters['MIN_CLUSTERS']
    max_clusters = parameters['MAX_CLUSTERS']
    min_neighbors = parameters['MIN_NEIGHBORS']
    max_neighbors = parameters['MAX_NEIGHBORS']

    n_clusters = np.random.randint(min_clusters, max_clusters)

    cluster_centers_x = np.random.uniform(0, range[0], size=[n_clusters])
    cluster_centers_y = np.random.uniform(0, range[1], size=[n_clusters])
    cluster_centers = np.stack([cluster_centers_x, cluster_centers_y], -1)

    points = []
    for k in xrange(n_clusters):
        n_neighbors = np.random.randint(min_neighbors, max_neighbors)
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
    num_sides_x = int(np.ceil(range[0] / float(resolution)))
    num_sides_y = int(np.ceil(range[1] / float(resolution)))
    xcoords, ycoords = np.meshgrid(
        np.linspace(0, range[0], num_sides_x + 1, endpoint=False),
        np.linspace(0, range[1], num_sides_y + 1, endpoint=False))
    points = np.stack([xcoords.ravel(), ycoords.ravel()], 0)
    values = density(points).reshape(xcoords.shape)
    return values


def occupation_space_from_approximation(aprox):
    return (aprox >= .25 * aprox.mean())


def make_kde(
        points,
        t_occupancy,
        resolution=1.0,
        epsilon=0.05,
        parameters=None):
    max_iters = parameters['MAX_ITERS']

    max_bw = points['range'].max() / 2
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
        if counter == max_iters:
            break

    return kde, kde_approx


def calculate_occupancy(density, range, resolution=1.0):
    density_approximation = make_density_approximation(
        density, range, resolution=resolution)
    occupation_space = occupation_space_from_approximation(
        density_approximation)
    return np.mean(occupation_space), density_approximation
