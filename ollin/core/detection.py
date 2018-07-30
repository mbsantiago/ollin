"""Module for generation of camera configuration and detection data.

Cameras can be placed in a virtual world through a
:py:class:`CameraConfiguration`. Given a camera configuration and movement
data, a detection matrix and further detection information can be obtained
through the :py:class:`Detection` class.

Occupancy and other state variables can then be estimated with such detection
data.
"""
from six.moves import xrange
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

from .constants import GLOBAL_CONSTANTS
from ..estimation.estimation import get_estimation_model


class CameraConfiguration(object):
    """Camera configuration class holding camera positions and directions.

    Attributes
    ----------
    positions : array
        Array of shape [num_cams, 2] to indicate coordinates of each
        camera.
    directions : array
        Array of shape [num_cams, 2] holding a vector of camera direction
        for each camera.
    cone_angle : float
        Viewing angle of cameras in degrees.
    cone_range : float
        Distance to camera at which detection is posible.
    range : array
        Array of shape [2] specifying the dimensions of the virtual world.
    site: :py:obj:`ollin.Site`
        Site object holding information about the virtual world.
    num_cams : int
        Number of cameras.

    """

    def __init__(
            self,
            positions,
            directions,
            site,
            cone_range=None,
            cone_angle=None):
        """Build a camera configuration object.

        Arguments
        ---------
        positions : list or tuple or array
            Array of shape [num_cams, 2] of camera positions.
        directions : list or tuple or array
            Array of shape [num_cams, 2] of camera pointing directions.
        site : :py:obj:`ollin.core.sites.Site`
            Site in which to place cameras.
        cone_range : float, optional
            Distance to camera at which detection is possible. If not provided
            it will be extracted from global constants (see
            :py:const:`ollin.core.constants.GLOBAL_CONSTANTS`).
        cone_angle : float, optional
            Viewing angle of camera in degrees. Default behaviour is as with
            cone_range.

        """
        self.positions = np.array(positions)
        self.directions = np.array(directions)

        self.range = site.range.astype(np.float64)
        self.site = site

        if cone_angle is None:
            cone_angle = GLOBAL_CONSTANTS['cone_angle']
        self.cone_angle = cone_angle

        if cone_range is None:
            cone_range = GLOBAL_CONSTANTS['cone_range']
        self.cone_range = cone_range

        self.num_cams = len(positions)

    def detect(self, mov):
        """Use camera configuration to detect movement history.

        Arguments
        ---------
        mov : :py:obj:`ollin.core.movement.Movement`
            Movement data object to be detected by the camera configuration.

        Returns
        -------
        data : :py:obj:`MovementDetection`
            Camera detection information.

        """
        data = MovementDetection(mov, self)
        return data

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            include=None,
            cone_length=0.4,
            camera_alpha=0.3,
            camera_color=None,
            **kwargs):
        """Draw camera positions and orientations.

        This method will make a plot representing the camera positions and
        orientations in the virtual world. To help visualize the cameras
        corresponding territory Voronoi cells of camera points will be plotted.

        Camera configuration plot adds two components:
            1. "cameras":
                If present in include list, camera position with cone of
                detection will be added to the plot.
            2. "camera_voronoi":
                If present in include list, Voronoi cells for each camera will
                be added to the plot.

        All other components in the include list will be handed down to the
        Site's plotting method. See
        :py:meth:`ollin.core.sites.Site.plot` to
        consult all plotting components defined at that level.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes in which to plot camera info. New axes will be created if none
            are provided.
        figsize : tuple or list, optional
            Size of figure, if ax is not provided. See figsize argument in
            :py:func:`matplotlib.pyplot.figure`.
        include: list or tuple, optional
            List of components to plot. Components list will be passed first
            to the Site object to add the corresponding
            components. Then components corresponding to CameraConfiguration
            included in the list will be plotted.
        cone_length : float, optional
            Length of camera cone for visualization purposes. Defaults to 0.4
            km.
        camera_alpha : float, optional
            Alpha value for camera cones.
        camera_color : str, optional
            Color for camera position points.
        kwargs : dict
            Other keyworded arguments will be passed to the CameraConfiguration
            plot method.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Axes of plot for further processing.

        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        from matplotlib.collections import PatchCollection

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if include is None:
            include = [
                'rectangle',
                'cameras',
                'camera_voronoi']

        ax = self.site.plot(ax=ax, include=include, **kwargs)

        if 'camera_voronoi' in include:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            vor = Voronoi(self.positions)
            voronoi_plot_2d(
                vor, show_points=False, show_vertices=False, ax=ax)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        if 'cameras' in include:
            if cone_length is None:
                cone_length = self.cone_range

            c_angle = self.cone_angle / 2.0
            patches = []
            for pos, angle in zip(self.positions, self.directions):
                ang = 180 * np.angle(angle[0] + 1j * angle[1]) / np.pi
                wedge = Wedge(pos, cone_length, ang - c_angle, ang + c_angle)
                patches.append(wedge)
            collection = PatchCollection(
                patches, cmap=plt.cm.hsv, alpha=camera_alpha)
            ax.add_collection(collection)

            xcoord, ycoord = self.positions.T
            ax.scatter(xcoord, ycoord, color=camera_color, s=4)

        return ax

    @classmethod
    def make_random(
            cls,
            num,
            site,
            min_distance=None,
            cone_range=None,
            cone_angle=None):
        """Place cameras randomly in range.

        Will create a number of cameras placed at random with random
        directions. If min_distance option is passed, then camera positions
        will be chosen so that any two are not closer that min_distance.

        Arguments
        ---------
        num : int
            Number of cameras to place.
        site : :py:obj:`ollin.core.sites.Site`
            Site in which to place cameras.
        min_distance : float, optional
            Minimum distance in between cameras.
        cone_range : float, optional
            Distance to camera at which detection is possible. If not provided
            it will be extracted from the global constants, see
            :py:const:`ollin.core.constants.GLOBAL_CONSTANTS`.
        cone_angle : float, optional
            Viewing angle of camera in radians. Default behaviour is as with
            cone_range.

        Returns
        -------
        camera : :py:obj:`CameraConfiguration`
            Camera configuration object with random positions and directions.

        """
        range = site.range.astype(np.float64)

        if min_distance is None:
            positions_x = np.random.uniform(0, range[0], size=(num))
            positions_y = np.random.uniform(0, range[1], size=(num))
            positions = np.stack([positions_x, positions_y], -1)
        else:
            positions = make_random_camera_positions(
                num, range, min_distance=min_distance)
        angles = make_random_directions(num)
        cam = cls(
            positions,
            angles,
            site,
            cone_range=cone_range,
            cone_angle=cone_angle)
        return cam

    @classmethod
    def make_grid(
            cls,
            distance,
            site,
            cone_range=None,
            cone_angle=None):
        """Place grid of cameras in virtual world.

        Place cameras in a square grid configuration in range of virtual world,
        parallel to the x and y axis, separated by some distance. Camera
        directions will be random.

        Arguments
        ---------
        distance : float
            Distance between adjacent cameras in grid.
        site : :py:obj:`ollin.core.sites.Site`
            Site in which to place cameras.
        cone_range : float, optional
            Distance to camera at which detection is possible. Default
            behaviour is as with range.
        cone_angle : float, optional
            Viewing angle of camera in radians. Default behaviour is as with
            range.

        Returns
        -------
        camera : :py:obj:`CameraConfiguration`
            Camera configuration object in square grid configuration.

        """
        range = site.range

        num_x = int(range[0] / distance)
        num_y = int(range[1] / distance)

        shift_x = range[0] / num_x
        shift_y = range[1] / num_y

        points_x = np.linspace(0, range[0], num_x, endpoint=False)
        points_y = np.linspace(0, range[1], num_y, endpoint=False)

        X, Y = np.meshgrid(points_x, points_y)
        positions = np.stack((X, Y), -1) + (np.array([shift_x, shift_y]) / 2)
        positions = positions.reshape([-1, 2])
        num = positions.size / 2
        angles = make_random_directions(num)
        cam = cls(
            positions,
            angles,
            site,
            cone_angle=cone_angle,
            cone_range=cone_range)
        return cam


def make_random_camera_positions(num, range, min_distance):
    """Create and return n random points in range separated by min distance.

    Arguments
    ---------
    num : int
        Number of points to create.
    range : tuple or list or array
        Shape of range. range = (a, b) means points will be taken from a a x b
        square.
    min_distance : float
        Distance of minimum separation between points.

    Returns
    -------
    points : array
        Array of shape [num, 2] holding the (x, y) coordinates of points.

    Raises
    ------
    RuntimeError
        If too many cameras are being placed so that minimum distance
        restriction has to be broken.

    """
    random_points_x = np.random.uniform(range[0]/10.0, size=[10, 10, 10])
    random_points_y = np.random.uniform(range[0]/10.0, size=[10, 10, 10])
    random_points = np.stack([random_points_x, random_points_y], -1)
    shift_x = np.linspace(0, range[0], 10, endpoint=False)
    shift_y = np.linspace(0, range[1], 10, endpoint=False)
    shifts = np.stack(np.meshgrid(shift_x, shift_y), -1)
    points = random_points + shifts[:, :, None, :]

    points = points.reshape([-1, 2])
    np.random.shuffle(points)

    selection = [points[0]]
    for i in xrange(num - 1):
        selected = False
        for point in points[1:]:
            is_far = True
            for other_point in selection:
                distance = np.sqrt(
                        (point[0] - other_point[0])**2 +
                        (point[1] - other_point[1])**2)
                if distance <= min_distance:
                    is_far = False
                    break
            if is_far:
                selected = True
                selection.append(point)
                break
        if not selected:
            raise RuntimeError('Cameras dont fit.')

    return np.array(selection)


def make_random_directions(num):
    """Create and return n random direction vectors."""
    angles = np.random.uniform(0, 2*np.pi, size=[num])
    directions = np.stack([np.cos(angles), np.sin(angles)], -1)
    return directions


class Detection(object):
    """Class holding camera detection information.

    Cameras left at site (virtual or real) will make detections at diferent
    steps in time. Which cameras detected when can be stored in a binary
    matrix. This matrix can then be used for estimation of state variables.

    Attributes
    ----------
    camera_config : :py:obj:`CameraConfiguration`
        The camera configuration for the detection data.
    range : array
        Array of shape [2] that holds the dimensions of site.
    detections : array
        Array of shape [steps, num_cams] containing the
        detection information. If::

            detections[j, i] = 1

        then the i-th camera had an detection event at the j-th time step.
    detection_nums : array
        Array of shape [num_cams] with the total number of
        detections for each camera.

    """

    def __init__(self, cam, detections):
        """Construct detection data object.

        Arguments
        ---------
        cam : :py:obj:`CameraConfiguration`
            Camera configuration.
        detections : array
            Array of shape [num_cams, steps] holding the detection
            information.

        """
        self.camera_configuration = cam
        self.range = cam.site.range

        msg = 'Detection array shape implies a different number of cameras'
        msg += ' to number reported from camera configuration'
        assert cam.num_cams == detections.shape[-1], msg

        self.detections = detections
        self.detection_nums = self.detections.sum(axis=0)

    def estimate_occupancy(
            self,
            model='single_species',
            method='MAP',
            priors=None):
        """Estimate occupancy and detectability from detection data.

        Use one of the estimation methods to estimate occupancy and
        detectability, see (:py:mod:`ollin.estimation`)

        Arguments
        ---------
            type : str
                Name of estimation method to use. See
                :py:mod:`ollin.estimation` documentation for a full list.
        Returns
        -------
            estimate : :py:obj:`ollin.estimation.Estimate`
                Estimate object containing estimation information.

        """
        model = get_estimation_model('occupancy', model)
        estimate = model.estimate(self, method=method, priors=priors)
        return estimate

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            include=None,
            detection_cmap='Purples',
            detection_alpha=0.2,
            **kwargs):
        """Plot camera detection data.

        Plots number of detections per camera by coloring the correponding
        Voronoi cell. The number of detections are transformed to [0, 1] scale
        and mapped to a color using a colormap.

        Detection plotting adds the following optional components to the plot:
            1. "detection:
                If present in include list Voronoi regions with color fill,
                encoding the corresponding detection numbers, will be added to
                the plot.
            2. "detection_colorbar":
                If present in include list a colorbar representing the mapping
                between detection numbers and colors will be added to the plot.

        All other components in the include list will be handed down to the
        CameraConfiguration plotting method. See
        :py:meth:`CameraConfiguration.plot` to see all plotting components
        defined there.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot detection information.
        figsize: list or tuple, optional
            Size of figure to create if no axes object was given.
        include: list or tuple, optional
            List of components to plot. Components list will be passed first
            to the Camera Configuration object to add the corresponding
            components. Then components corresponding to CameraConfiguration
            included in the list will be plotted.
        detection_cmap : str, optional
            Colormap with which to encode detection numbers. See
            :py:mod:`matplotlib.cm` to see all options. Defaults to 'Purples'.
        detection_alpha : float, optional
            Alpha value of Voronoi region's color fill.
        kwargs : dict, optional
            Any additional keyworded arguments will be passed to the Camera
            Configuration plot method.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Plot axes for further plotting.

        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if include is None:
            include = [
                'rectangle',
                'camera',
                'camera_voronoi',
                'detection',
                'detection_colorbar']

        ax = self.camera_configuration.plot(ax=ax, include=include, **kwargs)

        if 'detection' in include:
            vor = Voronoi(self.camera_configuration.positions)
            cmap = plt.get_cmap(detection_cmap)
            max_num = self.detection_nums.max()
            regions, vertices = voronoi_finite_polygons_2d(vor)
            nums = self.detection_nums
            for reg, num in zip(regions, nums):
                polygon = vertices[reg]
                X, Y = zip(*polygon)
                color = cmap(num / float(max_num))
                ax.fill(X, Y, color=color, alpha=detection_alpha)

            if 'detection_colorbar' in include:
                norm = Normalize(vmin=0, vmax=max_num)
                mappable = ScalarMappable(norm, cmap)
                mappable.set_array(self.detection_nums)
                plt.colorbar(mappable, ax=ax)

        return ax


class MovementDetection(Detection):
    """Class holding detection data arising from movement data.

    From camera placement and movement data, camera detection data can be
    calculated and collected into an array of shape [num_individuals,
    time_steps, num_cameras]. Here::

        array[j, i, k] = 1

    indicates that at the i-th step the j-th
    individual was detected by the k-th camera. Hence more detailed
    analysis is posible from such data.

    Attributes
    ----------
    camera_config : :py:obj:`CameraConfiguration`
        The camera configuration for the detection data.
    range : array
        Array of shape [2] that holds the dimensions of site.
    detections : array
        Array of shape [steps, num_cam] containing the detection information.
        Here::

            detections[j, i] = 1

        means that the i-th camera had an detection event at the j-th time
        step.
    detection_nums : array
        Array of shape [num_cams] with the total number of detections for each
        camera.
    movement : :py:obj:`ollin.core.movement.Movement`
        Movement data being detected.
    grid : array
        Array of shape [num_individuals, time_steps, num_cameras] holding all
        detection data.

    """

    def __init__(self, mov, cam):
        """Construct MovementDetection object.

        Arguments
        ---------
        mov : :py:obj:`ollin.core.movement.Movement`
            Movement data being detected.
        cam : :py:obj:`CameraConfiguration`
            Cameras used for detection.

        """
        msg = "Camera range and movement range do not coincide"
        assert (cam.range == mov.site.range).all(), msg

        self.movement = mov
        self.grid = make_detection_data(mov, cam)
        detections = np.amax(self.grid, axis=0)

        super(MovementDetection, self).__init__(cam, detections)

    def plot(self, ax=None, figsize=(10, 10), include=None, **kwargs):
        """Plot camera detection data.

        Plots number of detections per camera by coloring the correponding
        Voronoi cell. The number of detections are transformed to [0, 1] scale
        and mapped to a color using a colormap.

        Detection plotting adds the following optional components to the plot:
            1. "detection:
                If present in include list Voronoi regions with color fill,
                encoding the corresponding detection numbers, will be added to
                the plot.
            2. "detection_colorbar":
                If present in include list a colorbar representing the mapping
                between detection numbers and colors will be added to the plot.

        All other components in the include list will be passed down to the
        Movement plotting method. See
        :py:meth:`ollin.core.movement.Movement.plot` for all plot
        components defined at that level.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot detection information.
        figsize: list or tuple, optional
            Size of figure to create if no axes object was given.
        include: list or tuple, optional
            List of components to plot. Components list will be passed first
            to the Movement Data object to add the corresponding
            components. Then components corresponding to Movement
            included in the list will be plotted.
        kwargs: dict, optional
            All other keyworded arguments will be passed to Detection and
            Movement plotting methods.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Returns axes for further plotting.

        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xticks((0, self.range[0]))
            ax.set_yticks((0, self.range[1]))

        if include is None:
            include = [
                'rectangle',
                'camera',
                'camera_voronoi',
                'detection',
                'detection_colorbar']

        ax = self.movement.plot(ax=ax, include=include, **kwargs)
        ax = super(MovementDetection, self).plot(
            ax=ax, include=include, **kwargs)
        return ax


def make_detection_data(movement, camera_config):
    """Generate and return detection data from movement and camera data.

    Movement data is held in the
    :py:class:`ollin.core.movement.Movement` object and is represented
    by an array of shape [num_individuals, time_steps, 2], which contains the
    full history of movement along the simulated time.

    Use movement history, camera placement and directions to calculate an array
    of shape [num_individuals, time_steps, num_cams] where::

        array[j, k, i] = 1

    implies that the j-th individual was within the detection cone of the i-th
    camera at the k-th time step.

    Arguments
    ---------
    movement : :py:obj:`ollin.core.movement.Movement`
        Movement of individuals to detect.
    camera_config : :py:obj:`CameraConfiguration`
        Camera position and directions to use for detection

    Returns
    -------
    grid : array
        Array of shape [num_individuals, time_steps, num_cameras].

    """
    camera_position = camera_config.positions
    camera_direction = camera_config.directions
    camera_direction = camera_direction[:, 0] + 1j * camera_direction[:, 1]

    movement_data = movement.data
    num, steps, _ = movement_data.shape
    cone_range = camera_config.cone_range
    cone_angle = np.pi * camera_config.cone_angle / 360.0

    relative_pos = (movement_data[:, :, None, :] -
                    camera_position[None, None, :, :])
    relative_pos = relative_pos[:, :, :, 0] + 1j * relative_pos[:, :, :, 1]
    norm = np.abs(relative_pos)
    closeness = np.less(norm, cone_range)

    angles = np.abs(np.angle(relative_pos / camera_direction, deg=1))
    is_in_angle = np.less(angles, cone_angle, where=closeness)

    detected = closeness * is_in_angle
    return detected


def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

    Arguments
    ---------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
