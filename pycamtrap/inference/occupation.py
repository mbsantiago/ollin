from __future__ import print_function

import os
import pickle
import pystan


MODEL_PATHS = os.path.dirname(os.path.realpath(__file__))


def compile_and_save(model_name):
    full_path = os.path.join(MODEL_PATHS, 'models', model_name + '.stan')
    stan_model = pystan.StanModel(file=full_path)
    compiled_path = os.path.join(
        MODEL_PATHS, 'compiled_models', model_name + '.pkl')
    with open(compiled_path, 'wb') as pkl_file:
        pickle.dump(stan_model, pkl_file)
    return stan_model


def load_model(name):
    compiled_path = os.path.join(
        MODEL_PATHS, 'compiled_models', name + '.pkl')
    if not os.path.exists(compiled_path):
        print('Compiled model not found... Trying to compile.')
        stan_model = compile_and_save(name)
    else:
        with open(compiled_path, 'rb') as pkl_file:
            stan_model = pickle.load(pkl_file)
    return stan_model


DEFAULT_MODEL = load_model('single_species_model')


def estimate(detection_data, method='MAP', model=None, priors_parameters=None):
    global DEFAULT_MODEL
    if model is not None:
        DEFAULT_MODEL = load_model(model)

    if priors_parameters is None:
        priors_parameters = {}

    data = {
        'Days': detection_data.steps,
        'Cams': detection_data.camera_config.num_cams,
        'Det': detection_data.detections.T.astype(int),
        'alpha_oc': priors_parameters.get('alpha_oc', 1),
        'beta_oc': priors_parameters.get('beta_oc', 1),
        'alpha_det': priors_parameters.get('alpha_det', 1),
        'beta_det': priors_parameters.get('beta_det', 1),
    }
    if method == 'MAP':
        op = DEFAULT_MODEL.optimizing(data=data)
        occ = op['occupancy']
        det = op['detectability']
    elif method == 'mean':
        pass
    else:
        msg = 'Estimation method {} not implemented.'.format(method)
        raise NotImplementedError(msg)

    return occ, det
