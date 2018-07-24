"""Module for virtual world creation.

All simulations take place withing a rectangular region of the plane called a
site. The dimensions of such site are specified in the range variable.

Additionally, since such a site is meant to be the moving field of some
specific species, any site is complemented with niche information. This niche
information is encoded in a scalar field, a function of the two coordinates,
that provides a measure of "adequacy" of position for the species. The function
values should be in the [0, 1] range, taking a value of 1 to mean the highest
level of adequacy. The function is stored as an array representing the
rectangular region at some spatial resolution. The niche information can then
be exploited by to guide individuals in their movements.

Sites can be created randomly by placing a random number of cluster points
in range and making a gaussian kernel density estimation, or by specifying
points at which niche values are known to be high and extrapolating by some
kernel density estimation. This data could possibly arise from ecological and
climatic variables, real telemetric data, or presence/absence data from camera
traps studies.
"""
from __future__ import division

from abc import abstractmethod
import numpy as np
from scipy.stats import gaussian_kde

from constants import GLOBAL_CONSTANTS


class BaseSite(object):
    def __init__(
            self,
            range,
            niche):

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)
        self.range = range.astype(np.float64)

        self.niche = niche
        self.niche_size = self.get_niche_size(niche)
        self.resolution = self.get_niche_resolution(niche, range)

    @staticmethod
    def get_true_niche(niche):
        threshold = niche.mean() / 4
        true_niche = niche >= threshold
        return true_niche

    @staticmethod
    def get_niche_size(niche):
        true_niche = BaseSite.get_true_niche(niche)
        return true_niche.mean()

    @staticmethod
    def get_niche_resolution(niche, range):
        x, y = range
        n, m = niche.shape
        xres = x / n
        yres = y / m
        return (xres + yres)/2

    def plot(
            self,
            include=None,
            figsize=(10, 10),
            ax=None,
            niche_cmap='Reds',
            niche_alpha=1.0,
            boundary_color='black',
            **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error

        if include is None:
            include = ['rectangle', 'niche_boundary', 'niche']

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if 'niche_boundary' in include or 'niche' in include:
            sizex, sizey = self.niche.shape
            rangex, rangey = np.meshgrid(
                np.linspace(0, self.range[0], sizex),
                np.linspace(0, self.range[1], sizey))

            if 'niche_boundary' in include:
                zone = self.get_true_niche(self.niche)
                ax.contour(
                    rangex,
                    rangey,
                    zone.T,
                    levels=0.5,
                    colors=boundary_color)

            if 'niche' in include:
                plt.pcolormesh(
                    rangex,
                    rangey,
                    self.niche.T,
                    cmap=niche_cmap,
                    alpha=niche_alpha)

        if 'rectangle' in include:
            ax.set_xticks(np.linspace(0, self.range[0], 2))
            ax.set_yticks(np.linspace(0, self.range[1], 2))

            ax.set_xlim(0, self.range[0])
            ax.set_ylim(0, self.range[1])

        return ax

    @abstractmethod
    def sample(self, num):
        pass


class Site(BaseSite):
    def __init__(self, range, points, resolution, kde_bandwidth):
        self.points = points
        self.kde_bandwidth = kde_bandwidth

        niche, kde = self.make_niche(points, range, kde_bandwidth, resolution)
        self.kde = kde
        super(Site, self).__init__(range, niche)

    def sample(self, num):
        points = self.kde.resample(num).T
        points = np.maximum(
            np.minimum(points, self.range),
            [0, 0])
        return points

    @staticmethod
    def make_niche(points, range, kde_bandwidth, resolution=1.0):
        kde = gaussian_kde(points.T, kde_bandwidth)
        niche = Site.make_niche_from_kde(kde, range, resolution=resolution)
        return niche, kde

    @staticmethod
    def make_niche_from_kde(kde, range, resolution=1.0):
        num_sides_x = int(np.ceil(range[0] / float(resolution)))
        num_sides_y = int(np.ceil(range[1] / float(resolution)))

        shift_x = range[0] / (num_sides_x * 2)
        shift_y = range[1] / (num_sides_y * 2)

        ycoords, xcoords = np.meshgrid(
                np.linspace(0, range[1], num_sides_y, endpoint=False),
                np.linspace(0, range[0], num_sides_x, endpoint=False))
        points = np.stack(
                [xcoords.ravel() + shift_x, ycoords.ravel() + shift_y], 0)
        niche = kde(points).reshape([num_sides_x, num_sides_y])
        return niche

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            include=None,
            **kwargs):
        import matplotlib.pyplot as plt

        if include is None:
            include = [
                    'niche_boundary',
                    'niche',
                    'rectangle']
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax = super(Site, self).plot(ax=ax, include=include, **kwargs)

        if 'points' in include:
            X, Y = self.points.T
            ax.scatter(X, Y, label='KDE Points')

        return ax

    @classmethod
    def make_random(
            cls,
            niche_size,
            resolution=None,
            range=None,
            min_clusters=None,
            max_clusters=None,
            min_cluster_points=None,
            max_cluster_points=None):

        if resolution is None:
            resolution = GLOBAL_CONSTANTS['resolution']
        if range is None:
            range = GLOBAL_CONSTANTS['range']
        if min_clusters is None:
            min_clusters = GLOBAL_CONSTANTS['min_clusters']
        if max_clusters is None:
            max_clusters = GLOBAL_CONSTANTS['max_clusters']
        if min_cluster_points is None:
            min_cluster_points = GLOBAL_CONSTANTS['min_cluster_points']
        if max_cluster_points is None:
            max_cluster_points = GLOBAL_CONSTANTS['max_cluster_points']

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)

        points = _make_random_points(
            range, min_clusters, max_clusters, min_cluster_points,
            max_cluster_points)

        bandwidth = _select_bandwidth(range, points, niche_size, resolution)
        return cls(range, points, resolution, bandwidth)


def _make_random_points(range, min_clusters, max_clusters, min_cluster_points,
                        max_cluster_points):
    n_clusters = np.random.randint(min_clusters, max_clusters)

    cluster_centers_x = np.random.uniform(0, range[0], size=[n_clusters])
    cluster_centers_y = np.random.uniform(0, range[1], size=[n_clusters])
    cluster_centers = np.stack([cluster_centers_x, cluster_centers_y], -1)

    points = []
    for k in xrange(n_clusters):
        n_neighbors = np.random.randint(
            min_cluster_points, max_cluster_points)
        centered_points = np.random.normal(size=[n_neighbors, 2])
        variances = np.random.normal(size=[2, 2])
        sheared_points = np.tensordot(
            centered_points, variances, (1, 1))
        shifted_points = sheared_points + cluster_centers[k]
        points.append(shifted_points)

    points = np.concatenate(points, 0)
    return points


def _select_bandwidth(range, points, niche_size, resolution):
    max_iters = GLOBAL_CONSTANTS['max_iters']
    epsilon = GLOBAL_CONSTANTS['bandwidth_epsilon']

    max_bw = range.max() / 2
    min_bw = 0.01

    mid_bw = (max_bw + min_bw) / 2

    kde = gaussian_kde(points.T, mid_bw)

    counter = 0
    while True:
        niche = Site.make_niche_from_kde(kde, range, resolution=resolution)
        calculated_niche = Site.get_niche_size(niche)

        err = abs(calculated_niche - niche_size)
        if err < epsilon:
            break

        elif calculated_niche < niche_size:
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

    return (min_bw + max_bw) / 2
