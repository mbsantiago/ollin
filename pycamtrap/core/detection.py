import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

from constants import RANGE, CONE_RANGE, CONE_ANGLE


class CameraConfiguration(object):
    def __init__(self, positions, directions, range=RANGE, cone_range=CONE_RANGE, cone_angle=CONE_ANGLE):
        self.positions = positions
        self.directions = directions
        self.range = range
        self.cone_angle = cone_angle
        self.cone_range = cone_range
        self.num_cams = len(positions)

    def plot(self, ax=None, cone_length=None, show_cones=True, vor=None, alpha=0.3):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        from matplotlib.collections import PatchCollection
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks((0, self.range))
            ax.set_yticks((0, self.range))

        if vor is None:
            vor = Voronoi(self.positions)
        voronoi_plot_2d(vor, show_vertices=False, ax=ax)
        ax.set_xlim(0, self.range)
        ax.set_ylim(0, self.range)

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
    def make_random(cls, num, range=RANGE, min_distance=None, **kwargs):
        if min_distance is None:
            positions = np.random.uniform(0, range, size=(num, 2))
        else:
            positions = make_random_camera_positions(
                num, range=range, min_distance=min_distance)
        angles = make_random_directions(num)
        return cls(positions, angles, range=range, **kwargs)

    @classmethod
    def make_grid(cls, num, range=RANGE, **kwargs):
        dx = range / float(num)
        points = np.linspace(0, range, num, endpoint=False)
        X, Y = np.meshgrid(points, points)
        positions = np.stack((X, Y), -1) + (dx / 2)
        positions = positions.reshape([-1, 2])
        angles = make_random_directions(num**2)
        return cls(positions, angles, **kwargs)


def make_random_camera_positions(num, range=RANGE, min_distance=1.0):
    random_points = np.random.uniform(range/10.0, size=[10, 10, 10, 2])
    shift = np.linspace(0, range, 10, endpoint=False)
    shifts = np.stack(np.meshgrid(shift, shift), -1)
    points = random_points + shifts[:, :, None, :]

    points = points.reshape([-1, 2])
    np.random.shuffle(points)

    selection = [points[0]]
    for i in xrange(num - 1):
        selected = False
        for point in points[1:]:
            is_far = True
            for other_point in selection:
                distance = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
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
        self.range = mov.range
        self.grid = make_detection_data(mov, cam)
        self.detections = np.amax(self.grid, axis=0)
        self.detection_nums = self.detections.sum(axis=0)
        self.total_detections = self.detection_nums.sum()

    def plot(self, ax=None, plot_cameras=True, cmap='Purples', colorbar=True, movement=False, alpha=0.2):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        vor = Voronoi(self.camera_config.positions)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks((0, self.range))
            ax.set_yticks((0, self.range))

        cmap = plt.get_cmap(cmap)
        max_num = self.detection_nums.max()
        regions, vertices = voronoi_finite_polygons_2d(vor)
        for reg, num in zip(regions, self.detection_nums):
            polygon = vertices[reg]
            X, Y = zip(*polygon)
            color = cmap(num / float(max_num))
            ax.fill(X, Y, color=color, alpha=alpha)

        if movement:
            self.movement_data.plot(ax=ax)

        if colorbar:
            fig = plt.gcf()
            width, height = fig.get_size_inches()
            ax1 = fig.add_axes([1, .1, 0.1, .8])
            cb = mpl.colorbar.ColorbarBase(
                ax1, cmap=cmap,
                norm=mpl.colors.Normalize(vmin=0, vmax=max_num),
                orientation='vertical',
                alpha=alpha)

        if plot_cameras:
            self.camera_config.plot(ax=ax, vor=vor, alpha=alpha)


def make_detection_data(movement_data, camera_config):
    camera_position = camera_config.positions
    camera_direction = camera_config.directions
    species_movement = movement_data.data
    cone_range = camera_config.cone_range
    cone_angle = camera_config.cone_angle

    relative_pos = species_movement[:, :, None, :] - camera_position[None, None, :, :]
    norm = np.sqrt(np.sum(relative_pos**2, -1))
    closeness = np.less(norm, cone_range)

    direction = np.divide(relative_pos, norm[..., None], where=closeness[..., None])
    angles = np.arccos(np.sum(camera_direction * direction, -1), where=closeness)

    angle = np.pi * (cone_angle / 360.0)
    is_in_angle = np.less(angles, angle, where=closeness)

    detected = closeness * is_in_angle
    return detected


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

            t = vor.points[p2] - vor.points[p1] # tangent
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
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
