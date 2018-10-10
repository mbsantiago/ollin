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

And it must return an :py:obj:`.Estimate` object containing all relevant
estimate information. Each state variable being estimated has a specialized
Estimate class defining the relevant data to report in an estimation. 
