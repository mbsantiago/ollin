from __future__ import division

from ..estimation import EstimationModel
from .occupancy_estimation import OccupancyEstimate


class Model(EstimationModel):
    name = 'Naive Occupancy Estimator'

    def estimate(self, detection, **kwargs):
        steps, cams = detection.detections.shape
        nums = detection.detection_nums

        cams_w_detection = (nums > 0).sum()

        occupancy = cams_w_detection / cams
        detectability = (nums / steps).sum() / cams_w_detection

        est = OccupancyEstimate(
            occupancy, self, detection, detectability=detectability)
        return est
