from __future__ import print_function

from abc import abstractmethod
import os
import pickle

from .estimation import EstimationModel


COMPILED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'compiled_models')


class StanModel(EstimationModel):
    name = None
    stancode = None

    def __init__(self):
        self.stanmodel = self.load_model()

    @abstractmethod
    def prepare_data(self, detection, priors):
        pass

    @abstractmethod
    def estimate(self, detection, method='MAP', priors=None):
        if priors is None:
            priors = {}

        model = self.stanmodel
        data = self.prepare_data(detection, priors)

        if method == 'MAP':
            op = model.optimizing(data=data)
        elif method == 'mean':
            op = model.sampling(data=data)
        else:
            msg = 'Estimation method {} not implemented.'.format(method)
            raise NotImplementedError(msg)
        return op

    def load_model(self):

        if not os.path.exists(COMPILED_PATH):
            os.makedirs(COMPILED_PATH)

        path = os.path.join(
            COMPILED_PATH, self.name.replace(' ', '_') + '.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as stanfile:
                stan_model = pickle.load(stanfile)
        else:
            stan_model = self.compile_and_save()
        return stan_model

    def compile_and_save(self):
        import pystan

        stan_model = pystan.StanModel(model_code=self.stancode)

        path = os.path.join(
            COMPILED_PATH, self.name.replace(' ', '_') + '.pkl')
        with open(path, 'wb') as stanfile:
            pickle.dump(stan_model, stanfile)

        return stan_model
