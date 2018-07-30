"""Base class for all movement models."""

from abc import abstractmethod
from six import iteritems

from ..core.constants import MOVEMENT_PARAMETERS


class MovementModel(object):
    """Base class for all movement models

    Any movement model must subclass this class and implement the generate
    movement method. Also they must provide a global attribute defining the
    default parameters for any instance of the movement model.

    Attributes
    ----------
    name : str
        Name of model. Mainly used in plotting.
    default_parameters : dict
        Dictionary holding default values of any parameters associated with the
        movement model. There are some required parameters for every
        movement model. If not provided they will default to those in
        :py:const:`ollin.core.constants.MOVEMENT_PARAMETERS`.
    """
    name = None
    default_parameters = {}

    def handle_parameters(self, params):
        """Return parameter dictionary values with missing default values.

        This function searchs for all required parameters in the dictionary
        passed as argument. If not found it will default to the value specified
        in the default_parameters dictionary of the movement model class. If
        not found it will default to the one specified in the constants module.
        (see :py:const:`pycamptra.core.constants.MOVEMENT_PARAMETERS`)
        """
        if params is None:
            params = {}

        parameters = MOVEMENT_PARAMETERS.copy()

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

    def __init__(self, parameters=None):
        """Construct a movement model instance with the given parameters"""
        if parameters is None:
            parameters = {}
        self.parameters = self.handle_parameters(parameters)

    @abstractmethod
    def generate_movement(
            self,
            initial_position,
            site,
            steps,
            velocity):
        """Generate simulated movement from initial positions and conditions.

        This is an abstract method that must be implemented in any subclass.

        Arguments
        ---------
        initial_position : array
            Array of initial positions of shape [num, 2] to specify coordinates
            of individuals to be simulated.
        site: :py:obj:`ollin.Site`
            Site in which to simulate movement.
        steps : int
            Number of steps to simulate.
        velocity : int
            Mean velocity of individuals.

        Returns
        -------
        array : array
            Array of shape [num, steps, 2], so if (x, y) = Array[i, j, :] then
            x and y are the coordinates of the i-th individual at step j in the
            simulation. Number of steps is determined by number of days and
            steps per day parameter.
        """
        pass
