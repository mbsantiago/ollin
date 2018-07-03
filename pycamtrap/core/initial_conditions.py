from __future__ import division

import numpy as np  # pylint: disable=import-error
from scipy.stats import gaussian_kde  # pylint: disable=import-error

from utils import (home_range_to_velocity,
                   home_range_resolution,
                   occupancy_resolution)
from constants import handle_global_constants

from home_range import _make_grid as hr_grid
from occupancy import _make_grid as oc_grid


class InitialCondition(object):
    def __init__(
            self,
            occupancy,
            home_range,
            range=None,
            kde_points=None,
            parameters=None):

        parameters = handle_global_constants(parameters)
        self.parameters = parameters

        if range is None:
            range = parameters['range']

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)
        self.range = range.astype(np.float64)

        self.occupancy = occupancy
        self.home_range = home_range

        self.home_range_resolution = home_range_resolution(
            self.home_range, parameters=parameters)

        if kde_points is None:
            kde_points, n_clusters = self.make_cluster_points()
            self.n_clusters = n_clusters
        self.kde_points = kde_points
        self.kde, self.kde_approximation = self.make_kde()

    def make_cluster_points(self):
        return make_cluster_points(self.range, parameters=self.parameters)

    def make_kde(self):
        kde = make_kde(
            self.kde_points,
            self.range,
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
            time_steps=None,
            range=None,
            home_range=None,
            occupancy=None,
            parameters=None,
            center=True):

        if parameters is None:
            parameters = {}
        parameters = handle_global_constants(parameters)

        points = np.array(points)

        if points.ndim == 2:
            points = np.array([points])

        if center:
            points = points - points.min(axis=(0, 1))

        if range is None:
            minx, miny = np.min(points, axis=(0, 1))
            maxx, maxy = np.max(points, axis=(0, 1))

            range = (maxx - minx, maxy-miny)

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)
        range = range.astype(np.float64)

        if home_range is None:
            # Calculation of mean velocity per day
            if time_steps is None:
                time_steps = np.ones(points.shape[1] - 1)

            steps = points[:, 1:, :] - points[:, :-1, :]
            velocities = steps / time_steps[:, None]

            magnitudes = np.sqrt(np.sum(velocities ** 2, axis=-1))
            velocity = magnitudes.mean()
            resolution = home_range_resolution(
                velocity, parameters=parameters)

            home_range_grid = hr_grid(points, range, resolution)
            areas = np.sum(home_range_grid, axis=(1, 2))
            home_range = np.mean(areas)

        if occupancy is None:
            steps = points.shape[1]
            num = points.shape[0]

            resolution = occupancy_resolution(
                home_range, parameters=parameters)
            occupancy_grid = oc_grid(
                steps, points, range, num, resolution, 1)

            occupation_nums = np.sum(occupancy_grid, axis=1)
            occupancy = np.mean(occupation_nums / float(steps))

        kde_points = points.reshape([-1, 2])

        init = cls(
            occupancy,
            home_range,
            range=range,
            kde_points=kde_points,
            parameters=parameters)
        return init

    def plot(self, include=None, ax=None, niche_cmap='Reds'):
        import matplotlib.pyplot as plt  # pylint: disable=import-error

        if include is None:
            include = [
                    'heatmap',
                    'niche',
                    'rectangle']

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if 'rectangle' in include:
            rect = plt.Rectangle(
                [0, 0], self.range[0], self.range[1], fill=False)
            ax.add_patch(rect)

        if 'heatmap' in include or 'niche' in include:
            heatmap = self.kde_approximation
            sizex = heatmap.shape[0]
            sizey = heatmap.shape[1]
            Xrange, Yrange = np.meshgrid(
                    np.linspace(0, self.range[0], sizex),
                    np.linspace(0, self.range[1], sizey))

        if 'heatmap' in include:
            ax.pcolormesh(Xrange, Yrange, heatmap.T, cmap=niche_cmap)

        if 'niche' in include:
            zone = occupation_space_from_approximation(heatmap)
            ax.contour(Xrange, Yrange, zone.T, levels=0.5)

        if 'kde_points' in include:
            X, Y = self.kde_points.T
            ax.scatter(X, Y, label='KDE Points')

        ax.set_xticks(np.linspace(0, self.range[0], 2))
        ax.set_yticks(np.linspace(0, self.range[1], 2))

        ax.set_xlim(0, self.range[0])
        ax.set_ylim(0, self.range[1])

        return ax


def make_cluster_points(range, parameters=None):
    min_clusters = parameters['min_clusters']
    max_clusters = parameters['max_clusters']
    min_neighbors = parameters['min_neighbors']
    max_neighbors = parameters['max_neighbors']

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
            centered_points, variances, (1, 1))
        shifted_points = sheared_points + cluster_centers[k]
        points.append(shifted_points)

    points = np.concatenate(points, 0)
    return points, n_clusters


def make_density_approximation(density, range, resolution=1.0):
    num_sides_x = int(np.ceil(range[0] / float(resolution)))
    num_sides_y = int(np.ceil(range[1] / float(resolution)))

    shift_x = range[0] / (num_sides_x * 2)
    shift_y = range[1] / (num_sides_y * 2)

    ycoords, xcoords = np.meshgrid(
        np.linspace(0, range[1], num_sides_y, endpoint=False),
        np.linspace(0, range[0], num_sides_x, endpoint=False))
    points = np.stack(
        [xcoords.ravel() + shift_x, ycoords.ravel() + shift_y], 0)
    values = density(points).reshape([num_sides_x, num_sides_y])
    return values


def occupation_space_from_approximation(aprox):
    return (aprox >= .25 * aprox.mean())


def make_kde(
        points,
        range,
        t_occupancy,
        resolution=1.0,
        epsilon=0.05,
        parameters=None):
    max_iters = parameters['max_iters']

    max_bw = range.max() / 2
    min_bw = 0.01

    mid_bw = (max_bw + min_bw) / 2

    kde = gaussian_kde(points.T, mid_bw)

    counter = 0
    while True:
        occ, kde_approx = calculate_occupancy(
            kde, range, resolution=resolution)

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
