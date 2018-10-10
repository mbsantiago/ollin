import os
import importlib
import sys
import logging

sys.path.append('../')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

BLACKLIST = [
    'base.py',
    '__init__.py',
    '__pycache__']


MODEL_TEMPLATE = '''
{title}
{underscore}

.. automodule:: {module}
  :members:

'''

MAIN_TEMPLATE = '''
.. _{ref}:

{header}

.. toctree::
{toctree}
'''


def make_library(module, klass='Model'):
    path = os.path.join('..', module.replace('.', '/'))
    name = module.split('.')[-1]

    # Select files in directory not in blacklist
    model_files = filter(
        lambda path: (
            (path not in BLACKLIST) and
            ('.pyc' not in path) and
            ('.py' in path)),
        os.listdir(path))

    # Check if header exists
    header_file = os.path.join(
        'headers',
        name + '.rst')

    if not os.path.exists(header_file):
        raise IOError('No header file for {}'.format(name))

    # Make library directory
    lib_dir = os.path.join('libraries', name)

    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)

    # Add path to sys.path to load files
    sys.path.insert(0, path)

    # Add a file for each model
    rst_files = []
    for model_file in model_files:
        py_name = os.path.splitext(model_file)[0]
        module_name = '{}.{}'.format(module, py_name)
        logging.info('Building file for %s, module: %s', model_file, module_name)
        model = getattr(importlib.import_module(module_name), klass)
        model_name = model.name

        filename = py_name + '.rst'
        filepath = os.path.join(lib_dir, filename)
        rst_files.append(os.path.join(name, py_name))

        underscore = '^' * len(model_name)
        contents = MODEL_TEMPLATE.format(
            title=model_name,
            underscore=underscore,
            module=module_name).strip()

        with open(filepath, 'w') as rst_file:
            rst_file.write(contents)

    # Remove path from sys.path
    sys.path.pop(0)

    # Make main page for library
    basefile = lib_dir + '.rst'

    # Generate toctree
    toctree = '\n'.join(map(lambda path: '  ' + path, rst_files))

    # Make label for reference
    ref = name.replace('_', '-') + '-library'

    # Load header
    with open(header_file, 'r') as rst_file:
        header = rst_file.read()

    logger.info('Writing to file %s', basefile)
    with open(basefile, 'w') as mainfile:
        content = MAIN_TEMPLATE.format(
            header=header,
            toctree=toctree,
            ref=ref).strip()
        mainfile.write(content)


if __name__ == '__main__':
    make_library('ollin.movement_models')
    make_library('ollin.estimation.occupancy')
    make_library('ollin.movement_analyzers', klass='Analyzer')
