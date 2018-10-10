from importlib import import_module
import os
import glob

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


@lru_cache()
def get_movement_analyzer(analyzer):
    analyzer_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    analyzer_path = os.path.join(
            analyzer_path, 'movement_analyzers', '{}.py'.format(analyzer))

    if os.path.exists(analyzer_path):
        try:
            cls = import_module(
                    'ollin.movement_analyzers.{}'.format(analyzer)).Analyzer
            return cls
        except Exception as e:
            print('Unexpected exception occured while loading model file')
            raise e

    else:
        msg = 'Movement analyzer {} not implemented'.format(analyzer)
        raise NotImplementedError(msg)


def get_movement_analyzer_list():
    """Return all movement analyzers in library."""
    path = os.path.dirname(os.path.abspath(__file__))
    python_files = [
            os.path.basename(module)[:-3]
            for module in glob.glob(os.path.join(path, '*.py'))]
    movement_analyzers = [
            module for module in python_files
            if (module != '__init__') and (module != 'base')]

    return movement_analyzers
