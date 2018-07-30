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
be exploited to guide individuals in their movements.

Sites can be created randomly by placing a random number of cluster points
in range and making a Gaussian kernel density estimation, or by specifying
points at which niche values are known to be high and extrapolating by some
kernel density estimation. This data could possibly arise from ecological and
climatic variables, real telemetry data, or presence/absence data from camera
traps studies.
"""
from __future__ import division

from abc import abstractmethod
import numpy as np
from six.moves import xrange
from scipy.stats import gaussian_kde

from .constants import GLOBAL_CONSTANTS


class BaseSite(object):
    """Base class for all types of site.

    Attributes
    ----------
    range : array
        Array of shape [2], specifying the dimensions of site (in Km).
    niche : array
        Matrix representing the values of adequacy at different points in site.
    niche_size : float
        Proportion of total area adequate for species. This are points at which
        niche value is above some threshold.
    resolution : float
        Spatial resolution (in Km) of niche array. If ``range = (x, y)`` and
        ``niche.shape = [n, m]`` then ``resolution = (x/n + y/m)/2``.

    """

    def __init__(
            self,
            range,
            niche):
        """Construct BaseSite object.

        Arguments
        ---------
        range : int or float or tuple or list or array
            Dimensions of site in Km. If int or float, it will be assumed that
            site is a square.
        niche : array
            Matrix representing the values of adequacy at different points in
            site.

        """
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
    def get_true_niche(niche, threshold=0.25):
        """Select cells with good level of niche adequacy."""
        true_niche = niche >= threshold
        return true_niche

    @staticmethod
    def get_niche_size(niche):
        """Calculate proportion of area of adequate space."""
        true_niche = BaseSite.get_true_niche(niche)
        return true_niche.mean()

    @staticmethod
    def get_niche_resolution(niche, range):
        """Get spatial resolution used in niche array."""
        x, y = range
        n, m = niche.shape
        xres = x / n
        yres = y / m
        return (xres + yres)/2

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            include=None,
            niche_cmap='Reds',
            niche_alpha=1.0,
            boundary_color='black',
            **kwargs):
        """Plot BaseSite information.

        Site plotting adds the following optional components to the
        plot:

        1. "rectangle":
            If present in include list, axes will be fitted to Site's range and
            no ticks will be placed.

        2. "niche":
            If present in include list, a heatmap plot of niche's adequacy
            values will be placed within range. A colormap will be used to
            translate adequacy values to colors.

        2. "niche_boundary":
            If present in include list, the boundary defining the true niche
            will be plotted. The true niche is defined as those cells where
            adequacy value is higher than some threshold, see
            :py:meth:`BaseSite.get_true_niche`.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot site information.
        figsize : list or tuple, optional
            Size of figure to create if no axes object was given. Defaults to
            (10, 10).
        include : list or tuple, optional
            List of components to add to the plot.
        niche_cmap : str, optional
            Colormap to use to codify niche adequacy level to color. Defaults
            to 'Reds'.
        niche_alpha: float, optional
            Alpha value of niche cell colors.
        boundary_color: str, optional
            Color of boundary of true niche.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Return axes for further plotting.

        """
        import matplotlib.pyplot as plt

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
                ax.pcolormesh(
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
        """Sample n random points from site."""
        pass


class Site(BaseSite):
    """Site with niche from gaussian kernel density estimation.

    A simple way of obtaining an estimate of niche adequacy value, or for
    random niche creation, is to select points in space which are
    known to be adequate for the target species. Interpolation to unknown
    points in space can be accomplished with a gaussian kernel density
    estimation. Selecting different bandwidths for the kernel density
    estimation will result in tighter or more diffuse niche values.

    Attributes
    ----------
    points : array
        Array of shape [num_points, 2] containing the coordinates of the points
        used for the kernel density estimation.
    kde_bandwidth : float
        Bandwidth used in the gaussian kernel density estimation.
    kde : :py:obj:`scipy.stats.gaussian_kde`
        Density estimation object.
    range : array
        Array of shape [2], specifying the dimensions of site (in Km).
    niche : array
        Matrix representing the values of adequacy at different points in site.
    niche_size : float
        Proportion of total area adequate for species. This are points at which
        niche value is above some threshold.
    resolution : float
        Spatial resolution used for niche array construction, in Km.

    """

    def __init__(
            self,
            range,
            points,
            resolution=0.1,
            kde_bandwidth=0.3,
            max_niche_value=1):
        """Construct site object.

        Arguments
        ---------
        range : int or float or tuple or list or array
            Dimensions of site in Km. If int or float, it will be assumed that
            site is a square.
        points : array
            Array of shape [num_points, 2] with coordinates of points to use
            for the kernel density estimation.
        resolution : float
            Spatial resolution used for niche array construction, in Km.
        kde_bandwidth : float
            Bandwidth to use in kernel density estimation.
        resolution : float
            Spatial resolution used for niche array construction, in Km.
        kde_bandwidth : float
            Bandwidth to use in kernel density estimation.
        max_niche_value : float, optional
            After niche construction, niche array will be scaled so that its
            maximum value is this.

        """
        self.points = points
        self.kde_bandwidth = kde_bandwidth

        niche, kde = self.make_niche(points, range, kde_bandwidth, resolution)
        self.kde = kde

        niche = max_niche_value * niche / niche.max()
        super(Site, self).__init__(range, niche)

    def sample(self, num):
        """Use kernel density estimation to sample random points form site."""
        points = self.kde.resample(num).T
        points = np.maximum(
            np.minimum(points, self.range),
            [0, 0])
        return points

    @staticmethod
    def make_niche(points, range, kde_bandwidth, resolution=1.0):
        """Make niche array from points."""
        kde = gaussian_kde(points.T, kde_bandwidth)
        niche = Site.make_niche_from_kde(kde, range, resolution=resolution)
        return niche, kde

    @staticmethod
    def make_niche_from_kde(kde, range, resolution=1.0):
        """Make niche array from kernel density estimation."""
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
            points_color='red',
            **kwargs):
        """Plot Site information.

        Site plotting adds the following optional components to the
        plot:

        1. "points":
            If present in include list, points used for kernel density
            estimation will be show in plot.

        All other components in include list will be passed to BaseSite
        plotting method. See :py:meth:`BaseSite.plot` to see all components
        defined at that level.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot site information.
        figsize : list or tuple, optional
            Size of figure to create if no axes object was given. Defaults to
            (10, 10).
        include : list or tuple, optional
            List of components to add to the plot. Components list will be
            passed to the BaseSite plotting method to add the corresponding
            components.
        points_color : str, optional
            Color of points used for kernel density estimation. Defaults to
            'red'.
        **kwargs : dict, optional
            All other keyworded arguments will be passed to BaseSite plotting
            method.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Return axes for further plotting.

        """
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
            max_cluster_points=None,
            max_niche_value=1):
        """Make random site.

        Process for random site creation follows the next steps:

        1. Select a random number of clusters. The number is selected within
           the [min_clusters, max_clusters] range.
        2. For each cluster select a central point randomly, using a uniform
           distribution, within site's range.
        3. For each cluster, select a random number of cluster points, within
           the range [min_cluster_points, max_cluster_points], around the
           cluster center, using a gaussian distribution with random covariance
           matrix.
        4. Collect all generated points for use in kernel density estimation.
        5. Select kernel density estimation bandwidth so that niche
           size (see :py:meth:`BaseSite.get_niche_size`) is recovered.

        Arguments
        ---------
        niche_size : float
            Number in [0, 1] range representing the desired proportion of
            adequate niche space to total area.
        resolution : float, optional
            Spatial resolution to use for niche creation. If none is given it
            will be taken from the global constants. See
            :py:const:`ollin.core.constants.GLOBAL_CONSTANTS`.
        range : int or float or list or tuple or array, optional
            Size of created site. If int or float it will be assumed that site
            is square. If none is given it will be taken from the global
            constants.
        min_clusters : int, optional
            Minimum number of clusters used in random niche creation. If none
            is given it will be taken from the global constants.
        max_clusters : int, optional
            Maximum number of clusters used in random niche creation. If none
            is given it will be taken from the global constants.
        min_cluster_points : int, optional
            Minimum number points per cluster used in random niche creation. If
            none is given it will be taken from the global constants.
        max_cluster_points : int, optional
            Maximum number points per cluster used in random niche creation. If
            none is given it will be taken from the global constants.
        max_niche_value : float, optional
            Number in [0, 1] range. Final niche value will have this number as
            a maximum value.

        """
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
        site = cls(
            range,
            points,
            resolution=resolution,
            kde_bandwidth=bandwidth,
            max_niche_value=max_niche_value)
        return site


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
        niche = niche / niche.max()
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
