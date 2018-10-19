Add new Estimation Model
------------------------

Making a new estimation model involves the following steps:

Make the code
^^^^^^^^^^^^^

Make a new file with an abbreviation of the estimation model's name.

Any estimation model must be a new class called ``Model`` that inherits from
:py:class:`.EstimationModel`. A name for the class must be specified as an
attribute::

  from ollin.estimation import EstimationModel


  class Model(EstimationModel):
    name = 'New Estimation Model'

Estimation is implemented in the estimate method. This method must have as
arguments:

  1. The model instance (self).
  2. The detection data (:py:obj:`.Detection`).

Select the correct return class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At Ollin we try to standarize the way we make estimates of state variables.
Hence we try to enforce that every estimation model returns the correct type of
object. For every state variable we provide a specialized :py:class:`.Estimate`
class for estimation returns. Hence if you are estimating occupancy, we ask
that you return an object of type :py:class:`.OccupancyEstimate`.

For example::

  from ollin.estimation import EstimationModel
  from ollin.estimation.occupancy import OccupancyEstimate

  class Model(EstimationModel):
    name = 'New estimation model'

    class estimate(self, detection):
      mean = detection.detections.mean()
      return OccupancyEstimate(mean, self.model, detection)

Write a statistical model with Stan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For bayesian inference or general statistical modelling we use Stan_. To use
Stan in estimation models please use :py:class:`.StanModel` as interface. Write
your stan code, and put it in the class attribute ``stan_code``::

  from ollin.estimation import StanModel

  stan_code = '''
  data {
    int M;
    int N;
    matrix[N, M] x;
    vector[N] y;
  }
  parameters {
    vector[M] beta;
    float sigma;
  }
  model {
    y ~ normal(x * beta, sigma);
  }
  '''

  class EstimationModelExample(StanModel):
    stan_code=stan_code

To check all requirements of StanModels see the documentation
:py:class:`.StanModel`.

Stan code must be compiled before use, wich might take some time, but compiled
models are stored locally for future use. Ollin handles compilation, saving a
loading of stan models automatically.

.. _stan: http://mc-stan.org/

Add file to estimation library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the estimation model has been writen, fork the repository_ and move the
file to ``ollin/estimation/{variable}/``, where ``{variable}`` stands for the
variable being estimated. Reinstall Ollin using the setup script (see
:any:`installation`) and use the model freely! Don't forget to send the pull
request.

.. _repository: https://github.com/mbsantiago/ollin
