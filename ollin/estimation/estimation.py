"""
"""
from abc import abstractmethod
from importlib import import_module
import os
import glob

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


class Estimate(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data


class EstimationModel(object):
    name = None

    @abstractmethod
    def estimate(self, detection):
        pass


@lru_cache()
def get_estimation_model(what, name):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        what,
        name + '.py')
    if os.path.exists(path):
        try:
            model = import_module(
                'ollin.estimation.{}.{}'.format(what, name)).Model()
            return model
        except Exception as e:
            print('Unexpected exception occurred while loading model.')
            raise e


def get_estimation_model_list(what):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), what)
    python_files = [
        os.path.basename(module)[:-3]
        for module in glob.glob(os.path.join(path, '*.py'))]
    movement_models = [
            module for module in python_files
            if (
                (module != '__init__') and
                (module != '{}_estimation'.format(what)))]

    print('{} Estimation Model Library:'.format(what.title()))
    for num, mov in enumerate(movement_models):
        print('\t{}.- {}'.format(num + 1, mov))
