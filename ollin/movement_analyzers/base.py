"""Interface for all movement analyzers"""

from abc import abstractmethod, ABCMeta
import six


@six.add_metaclass(ABCMeta)
class MovementAnalyzer():
    """Base class for all movement analyzers.

    All movement analyzers must subclass this class and implement the analyze
    and plot methods. A name for the analyzer must also be provided.

    Attributes
    ----------
    name : str
        Name of analyzer.
    movement : :py:obj:`.Movement`
        Reference to the movement data being analyzed.
    results : :py:obj:`array`
        An array holding the results of the analysis.

    """

    @property
    @abstractmethod
    def name(self):
        pass

    def __init__(self, movement):
        """Construct a movement analyzer.

        Arguments
        ---------
        movement: :py:obj:`.Movement`
            Movement data to analyze.

        """
        self.movement = movement
        self.results = self.analyze(movement)

    @abstractmethod
    def analyze(self, movement):
        """Analyze movement data.

        This is an abstract method that must be overwritten by any
        implementation.

        Arguments
        ---------
        movement : :py:obj:`.Movement`

        Returns
        -------
        results : :py:obj:`array`
            Array containing the results of the analysis.

        """
        pass

    @abstractmethod
    def plot(self):
        """Visualize analysis results.

        This is an abstract method that must be overwritten by any
        implementation.

        """
        pass
