from __future__ import division

from ..estimation import EstimationModel
from .base import OccupancyEstimate


class Model(EstimationModel):
    r"""Naive Occupancy Estimator.

    The naive model uses the percentage of cameras with some detection to
    estimate occupancy. Detectability is then defined as the mean proportion
    of detection at cameras with some detection.

    Hence if detection data is:

    .. math::

        D = \left(d_{ij}\right)_{\substack{1 \leq i \leq N \\ 1 \leq j \leq M}}

    Where:

    .. math::

        d_{ij} = \begin{cases}
            1 & \text{Camera } i \text{ made a detection at step } j \\
            0 & \text{Otherwise}
        \end{cases}

    Then, if :math:`\hat{o_{}}` and :math:`\hat{d_{}}` is detectability:

    .. math::

        \begin{align}
            \hat{o_i} & = \max\{d_{ij} \mid 1 \leq j \leq M\} \\
            \hat{o_{}} & = \frac{1}{N} \sum_{i = 1}^N \hat{d_i}
        \end{align}

    And

    .. math::

        \begin{align}
            \hat{d_i} &= \frac{1}{M} \sum_{j = 1}^M d_{ij} \\
            \hat{d_{}} &= \frac{\sum_{i=1}^N \hat{d_i}}{\sum_{i=1}^N \hat{o_i}}
        \end{align}

    """

    name = 'Naive Occupancy Estimator'

    def estimate(self, detection, **kwargs):
        """Make estimation using detection data."""
        steps, cams = detection.detections.shape
        nums = detection.detection_nums

        cams_w_detection = (nums > 0).sum()

        occupancy = cams_w_detection / cams
        detectability = (nums / steps).sum() / cams_w_detection

        est = OccupancyEstimate(
            occupancy, self, detection, detectability=detectability)
        return est
