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

  Hoa
