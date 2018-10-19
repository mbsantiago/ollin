"""Module defining basic estimation interface and library tools.

Passive detector data can be used to estimate ecological state variables. This
module holds libraries of models used for estimation of these state variables.

Currently only occupancy estimation models are available, but we expect to add
more state variables in the near future.

Notes
-----
All estimation models must inherit from :py:class:`EstimationModel` metaclass.
Hence any estimation model must implement its
:py:meth:`EstimationModel.estimate` method. Estimation models must return an
:py:class:`Estimate` object. Estimate objects should specialize to whatever is
being estimated, see
:py:class:`.OccupancyEstimate` for an example.

New estimation models must be placed within their corresponding directories.
For example, any new occupancy estimation model should be placed in::

    ollin/estimation/occupancy/

Libraries of estimation models for other state variables can also be built
following the same structure, say::

    ollin/estimation/abundance/

could contain abundance estimation models.

Every estimation model must be contained in a new file, in its corresponding
directory, so that :py:func:`get_estimation_model` and
:py:func:`get_estimation_model_list` tools function correctly. These tools will
look for files in these directories automatically and no further integration is
needed.

"""
from abc import abstractmethod


class Estimate(object):
    """Basic estimate class.

    Any estimate result must hold, apart from the estimate, the original data
    and the estimation model used.

    This class is meant to serve as an interface for specialized estimates of
    different state variables. Occupancy, abundance and other state variables
    must have specialized Estimate classes.

    Attributes
    ----------
    model : :py:obj:`EstimationModel`
        Model used for estimation.
    data : :py:obj:`.Detection`
        Detection data used for estimation.

    """

    def __init__(self, model, data):
        """Construct Estimate object.

        Arguments
        ---------
        model : :py:obj:`EstimationModel`
            Model used for estimation.
        data : :py:obj:`.Detection`
            Detection data used for estimation.

        """
        self.model = model
        self.data = data


class EstimationModel(object):
    """Basic estimation model class.

    Any estimation model must subclass this class and define an estimate
    method.

    Attributes
    ----------
    name : str
        Name of estimation model.

    """

    name = None

    @abstractmethod
    def estimate(self, detection):
        """Make estimate using detection data.

        Abstract method that must be overwritten by any estimation model. Its
        argument must be a :py:obj:`.Detection` object, and
        it must return a :py:obj:`Estimate`.
        """
        pass
