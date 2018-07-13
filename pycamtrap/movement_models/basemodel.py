from abc import abstractmethod

from six import iteritems

from ..core.constants import (GLOBAL_CONSTANTS,
                              MOVEMENT_PARAMETERS)


class MovementModel(object):
    name = None
    default_parameters = {}

    def handle_parameters(self, params):
        if params is None:
            params = {}

        parameters = MOVEMENT_PARAMETERS.copy()
        parameters.update(GLOBAL_CONSTANTS)

        for key, value in iteritems(self.default_parameters):
            if isinstance(value, dict):
                try:
                    parameters[key].update(value)
                except KeyError:
                    parameters[key] = value
            else:
                parameters[key] = value

        for key, value in iteritems(params):
            if isinstance(value, dict):
                try:
                    parameters[key].update(value)
                except KeyError:
                    parameters[key] = value
            else:
                parameters[key] = value

        return parameters

    def __init__(self, parameters):
        self.parameters = self.handle_parameters(parameters)

    @abstractmethod
    def generate_movement(
            self,
            initial_position,
            initial_conditions,
            days,
            velocity):
        pass
