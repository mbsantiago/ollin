from importlib import import_module
import os
import glob

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


@lru_cache()
def load_movement_model(model):
    model_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(
            model_path, 'movement_models', '{}.py'.format(model))

    if os.path.exists(model_path):
        try:
            cls = import_module(
                    'ollin.movement_models.{}'.format(model)).Model
            return cls
        except Exception as e:
            print('Unexpected exception occured while loading model file')
            raise e


def get_movement_model(model, parameters=None):
    cls = load_movement_model(model)
    return cls(parameters)


def get_movement_model_list():
    """Return all movement models in library."""
    path = os.path.dirname(os.path.abspath(__file__))
    python_files = [
        os.path.basename(module)[:-3]
        for module in glob.glob(os.path.join(path, '*.py'))]
    movement_models = [
        module for module in python_files
        if (module != '__init__') and (module != 'base')]

    return movement_models
