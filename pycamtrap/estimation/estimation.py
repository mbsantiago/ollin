from abc import abstractmethod

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import box, Polygon

from stanmodels import estimate as stanestimate


class Estimate(object):
    def __init__(self, occupancy, detectability):
        self.occupancy = occupancy
        self.detectability = detectability

    def __repr__(self):
        msg = 'Estimate: Occupancy = {:1.3f}, Detectability = {:1.3f}'
        msg = msg.format(self.occupancy, self.detectability)
        return msg


class EstimationModel(object):
    def __init__(self, detection):
        self.detection = detection
        self.num_experiments = detection.num_experiments

        self.estimations = [
            self.estimate(detection.detections[k])
            for k in xrange(self.num_experiments)]

        self.occupancies = [
            est.occupancy for est in self.estimations]
        self.detectabilities = [
            est.detectability for est in self.estimations]

    @abstractmethod
    def estimate(self, detection_data):
        pass

    def __repr__(self):
        msg = ''
        for est in self.estimations:
            msg += str(est) + '\n'
        return msg


class StanModel(EstimationModel):
    def __init__(
            self,
            detection,
            model=None,
            method='MAP',
            priors_parameters=None):
        self.method = method
        self.model = model
        self.priors_parameters = priors_parameters
        super(StanModel, self).__init__(detection)

    def estimate(self, detection_data):
        occ, det = stanestimate(
            detection_data,
            method=self.method,
            model=self.model,
            priors_parameters=self.priors_parameters)
        return Estimate(occ, det)


class AreasModel(EstimationModel):
    def __init__(self, detection):
        self.area_ratios = self.calculate_area_ratios(detection.camera_config)
        self.steps = detection.steps
        super(AreasModel, self).__init__(detection)

    def calculate_area_ratios(self, camera):
        vor = Voronoi(camera.positions)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        voronoi_areas = []
        range_box = box(0, 0, camera.range[0], camera.range[1])
        for reg in regions:
            polygon = Polygon(vertices[reg])
            area = polygon.intersection(range_box).area
            voronoi_areas.append(area)
        voronoi_areas = np.array(voronoi_areas)

        camera_area = np.pi * (camera.cone_angle) * (camera.cone_range)**2 / 360.0

        return voronoi_areas / camera_area

    def estimate(self, detection_data):
        estimated_detection_nums = np.minimum(
            detection_data.sum(axis=0) * self.area_ratios,
            self.steps)
        # estimated_detection_nums = det.detection_nums * area_ratio
        occupancy = np.mean(estimated_detection_nums / float(self.steps))
        return Estimate(occupancy, 0)


def make_estimate(det, type='stan', **kwargs):
    if type == 'areas':
        return AreasModel(det, **kwargs)
    elif type == 'stan':
        return StanModel(det, **kwargs)


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
