from importlib import import_module
import os
import glob

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
from .estimation import EstimationModel


@lru_cache()
def get_estimation_model(variable, name):
    """Load and return an estimation model by name.

    Arguments
    ---------
    variable : str
        Name of state variable to estimate.
    name : str
        Name of estimation model.

    Returns
    -------
    model : :py:obj:`EstimationModel`

    Raises
    ------
    Exception
        If no estimation model of the given name was found or some error
        occurred when loading.

    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        variable,
        name + '.py')
    if os.path.exists(path):
        try:
            model = import_module(
                'ollin.estimation.{}.{}'.format(variable, name)).Model()
            return model
        except Exception as e:
            print('Unexpected exception occurred while loading model.')
            raise e


def get_estimation_model_list(variable):
    """Print all estimation model names for state variable.

    Search for all estimation models in the state variable library and print
    for consultation.

    Arguments
    ---------
    variable : str
        Name of state variable.

    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), variable)
    python_files = [
        os.path.basename(module)[:-3]
        for module in glob.glob(os.path.join(path, '*.py'))]
    estimation_models = [
        module for module in python_files
        if (
            (module != '__init__') and
            (module != 'base'))]

    return estimation_models
