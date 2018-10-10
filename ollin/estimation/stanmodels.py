"""Module defining estimation models using PyStan for estimation.

:py:mod:`PyStan` is a python package for Bayesian inference with an efficient
Hamiltonian Monte Carlo sampler under the hood (see pystan documentation
http://pystan.readthedocs.io/en/latest/). The Monte Carlo sampler can be used
for MAP (maximum *a posteriori*) estimation, or full Bayesian inference.

Models are defined in a specialized statistical programming language and
compiled to an computationally fast and efficient sampler.

This module defines a class that interfaces all models that use Stan for
inference.

Warning
-----
Since compilation is necessary, the first run of a Stan Estimation model will
seem very slow. Compiled versions are stored so a second use of the estimation
model will not incur in such expensive overhead.

Compilation is hardware and python version dependant, so no pre-compiled models
are shipped with a standard ollin installation.

Attributes
---------
COMPILED_PATH : str
    Path of all compiled models.

"""
from __future__ import print_function

from abc import abstractmethod
import os
import pickle

from .estimation import EstimationModel


COMPILED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'compiled_models')


class StanModel(EstimationModel):
    """Class interface for estimation models that use PyStan.

    All estimation models that use PyStan must inherit from this class and
    implement :py:meth:`estimate` and :py:meth:`prepare_data methods`.

    Attributes
    ----------
    name : str
        Name of model.
    stancode : str
        String containing the code in Stan language that defines the
        statistical model to use.
    stanmodel : :py:obj:`pystan.StanModel`
        Compiled stanmodel object to use in inference.

    """

    name = None
    stancode = None

    def __init__(self):
        """Construct a StanModel object."""
        self.stanmodel = self.load_model()

    @abstractmethod
    def prepare_data(self, detection, priors):
        """Prepare the data to feed to the Stan model.

        See the :py:meth:`pystan.StanModel.sampling` documentation for further
        information
        (http://pystan.readthedocs.io/en/latest/api.html#pystan.StanModel).

        Arguments
        ---------
        detection : :py:obj:`.Detection`
            Detection data.
        priors : dict
            Any priors parameters information should be stored here.

        Returns
        -------
        data : dict
            All inputs for variables defined in the Stan model must be
            contained in this dictionary.

        """
        pass

    @abstractmethod
    def estimate(self, detection, method='MAP', priors=None):
        """Estimate using detection data and stan model.

        Detection data is prepared using :py:meth:`prepare_data` method
        and fed to pystan model. The model then samples from the posterior
        distribution (if method == 'sample') or optimizes for the parameters in
        the posterior distribution (if method == 'MAP').

        Any estimation model that inherits from this class must extend this
        method to extract the relevant information from the stanmodel output.

        Arguments
        ---------
        detection : :py:obj:`ollin.core.detection.Detection`
            Detection data from which to make estimate.
        method : {'MAP', 'sample'}, optional
            Method for inference. If 'MAP', Stan will try to find the
            parameters at which likelihood of the posterior distribution is a
            maximum. If 'sample', Stan will run the Hamiltonian Monte Carlo
            sampler to return a large sample of the posterior distribution.
            Defaults to 'MAP'.
        priors : dict, optional
            Dictionary holding all information of priors parameters.

        Returns
        -------
        result : dict or :py:obj:`pystan.StanFit4Model`
            If method = 'MAP' it will return a dictionary with the
            parameter values at which a maximum (local) of the posterior
            likelihood was found. If method = 'sample' it will a
            :py:obj:`pystan.StanFit4Model` object that contains all information
            of the sampling. See
            http://pystan.readthedocs.io/en/latest/api.html#pystan.StanModel.sampling.

        """
        if priors is None:
            priors = {}

        model = self.stanmodel
        data = self.prepare_data(detection, priors)

        if method == 'MAP':
            result = model.optimizing(data=data)
        elif method == 'sample':
            result = model.sampling(data=data)
        else:
            raise ValueError('Method must be "MAP" or "sample".')
        return result

    def load_model(self):
        """Load and return compiled model.

        If no compiled version is found it will compile the model and save it.

        """
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
        """Compile Stan code and save in compiled model directory."""
        import pystan

        stan_model = pystan.StanModel(model_code=self.stancode)

        path = os.path.join(
            COMPILED_PATH, self.name.replace(' ', '_') + '.pkl')
        with open(path, 'wb') as stanfile:
            pickle.dump(stan_model, stanfile)

        return stan_model
