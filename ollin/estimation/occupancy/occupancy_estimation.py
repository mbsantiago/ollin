from ..estimation import Estimate


class OccupancyEstimate(Estimate):
    __slots__ = [
        'occupancy', 'model', 'data', 'detectability']

    def __init__(self, occupancy, model, data, detectability=None):
        self.occupancy = occupancy
        self.model = model
        self.data = data
        self.detectability = detectability

    def __str__(self):
        msg = 'Occupancy estimation done with {} model.\n'
        msg += '\tOccupancy: {}'.format(self.occupancy)
        if self.detectability is not None:
            msg += '\n\tDetectability: {}'.format(self.detectability)
        return msg.format(self.model.name)
