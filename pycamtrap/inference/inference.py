import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import box, Polygon

from occupation import estimate

def make_inference(det, type):
    if type == 'areas':
        occupancy = infer_areas(det)

    elif type == 'MacKenzie':
        occupancy = mackenzie(det)

    return occupancy


def infer_areas(det):
    vor = Voronoi(det.camera_config.positions)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    voronoi_areas = []
    range_box = box(0, 0, det.range[0], det.range[1])
    for reg in regions:
        polygon = Polygon(vertices[reg])
        area = polygon.intersection(range_box).area
        voronoi_areas.append(area)
    voronoi_areas = np.array(voronoi_areas)

    cam = det.camera_config
    camera_area = np.pi * (cam.cone_angle) * (cam.cone_range)**2 / 360.0

    area_ratio = voronoi_areas / camera_area

    estimated_detection_nums = np.minimum(det.detection_nums * area_ratio, det.steps)
    # estimated_detection_nums = det.detection_nums * area_ratio
    occupancy = np.mean(estimated_detection_nums / float(det.steps))
    return occupancy


def mackenzie(det):
    occ, det = estimate(det)
    return occ


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
