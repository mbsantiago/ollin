"""Occupancy estimate definition."""
from ..estimation import Estimate


class OccupancyEstimate(Estimate):
    """Occupancy estimate container.

    Extension of :py:class:`ollin.estimation.estimation.Estimate` class for
    occupancy estimates.

    Attributes
    ----------
    occupancy : float
        Estimated occupancy.
    detectability : float or None
        Estimated detectability. Some models do not provide an estimated
        detectability.
    model : :py:obj:`ollin.estimation.estimation.EstimationModel`
        Model used to make estimate.
    data : :py:obj:`ollin.core.detection.Detection`
        Data used to make detection.

    """

    __slots__ = [
        'occupancy', 'model', 'data', 'detectability']

    def __init__(self, occupancy, model, data, detectability=None):
        """Construct Occupancy Estimate object.

        Arguments
        ---------
        occupancy : float
            Estimated occupancy.
        model : :py:obj:`ollin.estimation.estimation.EstimationModel`
            Model used for estimation.
        data : :py:obj:`ollin.core.detection.Detection`
            Detection data used for estimation.
        detectability : float, optional
            Estimated detectability.

        """
        self.occupancy = occupancy
        self.model = model
        self.data = data
        self.detectability = detectability

    def __str__(self):
        """Representation of Occupancy estimate."""
        msg = 'Occupancy estimation done with {} model.\n'
        msg += '\tOccupancy: {}'.format(self.occupancy)
        if self.detectability is not None:
            msg += '\n\tDetectability: {}'.format(self.detectability)
        return msg.format(self.model.name)
