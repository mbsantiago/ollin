import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

from constants import handle_parameters
from pycamtrap.estimation import make_estimate


class CameraConfiguration(object):
    def __init__(self, positions, directions, range=None, parameters=None):

        if parameters is None:
            parameters = {}
        parameters = handle_parameters(parameters)

        self.positions = positions
        self.directions = directions

        if range is None:
            range = parameters['RANGE']

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)
        self.range = range.astype(np.float64)

        self.cone_angle = parameters['CONE_ANGLE']
        self.cone_range = parameters['CONE_RANGE']
        self.num_cams = len(positions)

    def plot(
            self,
            ax=None,
            cone_length=None,
            show_cones=True,
            vor=None,
            alpha=0.3):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        from matplotlib.collections import PatchCollection

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks((0, self.range[0]))
            ax.set_yticks((0, self.range[1]))

        if vor is None:
            vor = Voronoi(self.positions)
        voronoi_plot_2d(vor, show_vertices=False, ax=ax)
        ax.set_xlim(0, self.range[0])
        ax.set_ylim(0, self.range[1])

        if show_cones:
            if cone_length is None:
                cone_length = self.cone_range

            c_angle = self.cone_angle / 2.0
            patches = []
            for pos, angle in zip(self.positions, self.directions):
                ang = 180 * np.angle(angle[0] + 1j * angle[1]) / np.pi
                wedge = Wedge(pos, cone_length, ang - c_angle, ang + c_angle)
                patches.append(wedge)
            collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=alpha)
            ax.add_collection(collection)
        return ax

    @classmethod
    def make_random(cls, num, range=None, min_distance=None, parameters=None):
        if parameters is None:
            parameters = {}
        parameters = handle_parameters(parameters)

        if range is None:
            range = parameters['RANGE']

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)
        range = range.astype(np.float64)

        if min_distance is None:
            positions_x = np.random.uniform(0, range[0], size=(num))
            positions_y = np.random.uniform(0, range[1], size=(num))
            positions = np.stack([positions_x, positions_y], -1)
        else:
            positions = make_random_camera_positions(
                num, range, min_distance=min_distance)
        angles = make_random_directions(num)
        return cls(positions, angles, range=range, parameters=parameters)

    @classmethod
    def make_grid(cls, distance, range=None, parameters=None):
        if parameters is None:
            parameters = {}
        parameters = handle_parameters(parameters)

        if range is None:
            range = parameters['RANGE']

        if isinstance(range, (int, float)):
            range = np.array([range, range])
        elif isinstance(range, (tuple, list)):
            if len(range) == 1:
                range = [range[0], range[0]]
            range = np.array(range)
        range = range.astype(np.float64)

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
        return cls(positions, angles, range=range, parameters=parameters)


def make_random_camera_positions(num, range, min_distance=1.0):
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
            raise Exception('No funciono')

    return np.array(selection)


def make_random_directions(num):
    angles = np.random.uniform(0, 2*np.pi, size=[num])
    directions = np.stack([np.cos(angles), np.sin(angles)], -1)
    return directions


class Detection(object):
    def __init__(self, mov, cam):
        self.movement_data = mov
        self.camera_config = cam

        self.steps = mov.steps
        self.num_experiments = mov.num_experiments
        self.range = mov.range
        self.grid = make_detection_data(mov, cam)
        self.detections = np.amax(self.grid, axis=1)
        self.detection_nums = self.detections.sum(axis=1)

    def estimate(self, type='stan'):
        return make_estimate(self, type=type)

    def plot(
            self,
            ax=None,
            plot_cameras=True,
            cone_length=None,
            cmap='Purples',
            colorbar=True,
            movement=False,
            alpha=0.2,
            experiment_number=0):
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        vor = Voronoi(self.camera_config.positions)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks((0, self.range[0]))
            ax.set_yticks((0, self.range[1]))

        cmap = plt.get_cmap(cmap)
        max_num = self.detection_nums.max()
        regions, vertices = voronoi_finite_polygons_2d(vor)
        if experiment_number == 'mean':
            nums = self.detection_nums.mean(axis=0)
        else:
            nums = self.detection_nums[experiment_number]
        for reg, num in zip(regions, nums):
            polygon = vertices[reg]
            X, Y = zip(*polygon)
            color = cmap(num / float(max_num))
            ax.fill(X, Y, color=color, alpha=alpha)

        if movement:
            self.movement_data.plot(ax=ax)

        if colorbar:
            norm = Normalize(vmin=0, vmax=max_num)
            mappable = ScalarMappable(norm, cmap)
            mappable.set_array(self.detection_nums)
            plt.colorbar(mappable, ax=ax)

        if plot_cameras:
            self.camera_config.plot(
                ax=ax,
                vor=vor,
                alpha=alpha,
                cone_length=cone_length)

        return ax


def make_detection_data(movement_data, camera_config):
    camera_position = camera_config.positions
    camera_direction = camera_config.directions
    camera_direction = camera_direction[:, 0] + 1j * camera_direction[:, 1]
    num_cameras = len(camera_position)

    movement_data = movement_data.data
    num_experiments, num, steps, _ = movement_data.shape
    cone_range = camera_config.cone_range
    cone_angle = camera_config.cone_angle

    grid = np.zeros((num_experiments, num, steps, num_cameras))

    for k in xrange(num_experiments):
        species_movement = movement_data[k, :, :, :]

        relative_pos = (species_movement[:, :, None, :] -
                        camera_position[None, None, :, :])
        relative_pos = relative_pos[:, :, :, 0] + 1j * relative_pos[:, :, :, 1]
        norm = np.abs(relative_pos)
        closeness = np.less(norm, cone_range)

        angles = np.abs(np.angle(relative_pos / camera_direction, deg=1))
        is_in_angle = np.less(angles, cone_angle / 2.0, where=closeness)

        detected = closeness * is_in_angle
        grid[k, :, :, :] = detected
    return grid


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
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
