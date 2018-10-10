"""Helper functions."""
from __future__ import division

import numpy as np


def sigmoid(x):
    r"""Sigmoid function.

    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}

    """
    return 1 / (1 + np.exp(-x))


def logit(y):
    r"""Logit function.

    .. math::

        \mathop{logit}(y) = -log\left(\frac{1 - y}{y}\right)
    """
    return -np.log((1/y) - 1)


def occupancy_resolution(home_range, parameters=None):
    r"""Get discretization resolution from home range.

    Notes
    -----
    This function is used throughout the package to select spatial resolution
    of site discretization for all occupancy calculations. Such resolution is
    dependant on home range information, and possibly some model parameters.

    Currently, the relation assumed is

    .. math::

        resolution = \sqrt{home\_range}

    Arguments
    ---------
    home_range : float
    parameters : dict, optional
        Possible parameters affecting the home_range -> resolution relation.

    Returns
    -------
    resolution: float

    """
    return np.sqrt(home_range)


def home_range_to_velocity(home_range, parameters=None):
    r"""Get mean velocity from home range information.

    Notes
    This function is used throughout the package to translate home range
    information to mean velocity. Each movement model produces a different
    relation between home range and mean velocity, but a single model is
    applied to all movement models. The general relationship assumed is:

    .. math::

        home\_range = \alpha \cdot velocity^{\beta}

    Parameters are then fitted using simulated data.

    Arguments
    ---------
    home_range : float
    parameters : dict
        The parameters dictionary must contain at two keys, "exponent" and
        "alpha".

    Returns
    -------
    velocity : float

    """
    exponent = parameters['exponent']
    alpha = parameters['alpha']
    return np.power(home_range / alpha, 1 / exponent)


def velocity_to_home_range(velocity, parameters=None):
    r"""Get home range from mean velocity information.

    Notes
    -----
    This function is used throughout the package to translate mean velocity
    to home range information. Each movement model produces a different
    relation between home range and mean velocity, but a single model is
    applied to all movement models. The general relationship assumed is:

    .. math::

        home\_range = \alpha \cdot velocity^{\beta}

    Parameters are then fitted using simulated data.

    Arguments
    ---------
    velocity: float
    parameters : dict
        The parameters dictionary must contain at two keys, "exponent" and
        "alpha".

    Returns
    -------
    home_range : float

    """
    exponent = parameters['exponent']
    alpha = parameters['alpha']
    return alpha * velocity**exponent


def occupancy_to_density(
        occupancy,
        home_range_proportion,
        niche_size,
        parameters=None):
    r"""Get estimated density from occupancy information.

    Notes
    -----
    This function is used throughout the package to translate occupancy
    to density information. This relation also depends on the size of
    the home range in relation to the whole site's area, and to the proportion
    of available adequate space, or size of true niche, see
    :py:meth:`.BaseSite.get_niche_size`. Each movement model
    produces a different relation between occupancy and density, but a single
    model (or family of functions) is applied to all movement models. The
    general relationship assumed is:

    .. math::

        logit(o) = \alpha+\beta_1 log(d)+\beta_2 log(ns)+\beta_3 log(hrp)

    where :math:`o, d, ns, hrp` stand for occupancy, density, niche size and
    home range proportion, and :math:`logit(x) = - log(-1 + 1/x)`.
    Equivalently:

    .. math::

        o = \frac{\alpha \cdot d^{\beta_1} \cdot ns^{\beta_2} \cdot
        hrp^{\beta_3}}{1 + \alpha \cdot d^{\beta_1} \cdot
        ns^{\beta_2} \cdot hrp^{\beta_3}}

    Parameters are then fitted using simulated data.

    Arguments
    ---------
    occupancy: float
    home_range_proportion : float
        Number in [0, 1] range representing the proportion of home range area
        to total area.
    niche_size : float
        Proportion of total area that is adequate for species.
    parameters : dict
        The parameters dictionary must contain four keys: "alpha", "hr_exp",
        "density_exp" and "niche_size_exp".

    Returns
    -------
    home_range : float

    """
    alpha = parameters['alpha']
    hr_exp = parameters['hr_exp']
    den_exp = parameters['density_exp']
    nsz_exp = parameters['niche_size_exp']

    density = np.exp(
        (logit(occupancy) -
         alpha -
         np.log(home_range_proportion) * hr_exp -
         np.log(niche_size) * nsz_exp) / den_exp)
    return density


def density_to_occupancy(
        density,
        home_range_proportion,
        niche_size,
        parameters=None):
    r"""Get estimated occupancy from density information.

    Notes
    -----
    This function is used throughout the package to translate density to
    occupancy information. This relation also depends on the size of
    the home range in relation to the whole site's area, and to the proportion
    of available adequate space, or size of true niche, see
    :py:meth:`.BaseSite.get_niche_size`. Each movement model
    produces a different relation between occupancy and density, but a single
    model (or family of functions) is applied to all movement models. The
    general relationship assumed is:

    .. math::

        logit(o) = \alpha+\beta_1 log(d)+\beta_2 log(ns)+\beta_3 log(hrp)

    where :math:`o, d, ns, hrp` stand for occupancy, density, niche size and
    home range proportion, and :math:`logit(x) = - log(-1 + 1/x)`.
    Equivalently:

    .. math::

        o = \frac{\alpha \cdot d^{\beta_1} \cdot ns^{\beta_2} \cdot
                hrp^{\beta_3}}{1 + \alpha \cdot d^{\beta_1} \cdot
                ns^{\beta_2} \cdot hrp^{\beta_3}}

    Parameters are then fitted using simulated data.

    Arguments
    ---------
    density : float
    home_range_proportion : float
        Number in [0, 1] range representing the proportion of home range area
        to total area.
    niche_size : float
        Proportion of total area that is adequate for species.
    parameters : dict
        The parameters dictionary must contain four keys: "alpha", "hr_exp",
        "density_exp" and "niche_size_exp".

    Returns
    -------
    occupancy : float

    """
    alpha = parameters['alpha']
    hr_exp = parameters['hr_exp']
    den_exp = parameters['density_exp']
    nsz_exp = parameters['niche_size_exp']

    occupancy = sigmoid(
        alpha +
        np.log(density) * den_exp +
        np.log(home_range_proportion) * hr_exp +
        np.log(niche_size) * nsz_exp)
    return occupancy


def home_range_resolution(velocity, parameters=None):
    """Get discretization resolution from mean velocity.

    Notes
    -----
    This function is used throughout the package to select spatial resolution
    of site discretization for all home range calculations. Such resolution is
    dependant on mean velocity information, and possibly other model
    parameters.

    Currently, the relation assumed is

    .. math::

        resolution = velocity

    More complicated relations might be used in the future.

    Arguments
    ---------
    velocity : float
    parameters : dict, optional
        Possible parameters affecting the velocity -> resolution relation.

    Returns
    -------
    resolution: float

    """
    return velocity


def velocity_modification(niche_size, parameters):
    """Get velocity modification at niche size.

    Although for many movement models steps sizes are typically drawn
    from a distribution with a known mean, so that mean velocity can
    be controlled, some models include niche information in step size
    selection which could possible perturb mean step size. Therefore velocity
    correction is necessary.

    The correction depends on the overall niche size, which is the proportion
    of total area with an adequate level of niche value (see
    :py:meth:`.BaseSite.get_niche_size`) and also on the
    movement model used.

    Based on simulated data, the correct parameters for velocity correction can
    be obtained. See :py:mod:`.calibration`.

    Arguments
    ---------
    niche_size : float
    parameters : dict
        Dictionary holding the movement model parameters for velocity
        correction.

    """
    alpha = parameters['velocity']['alpha']
    beta = parameters['velocity']['beta']
    return beta + alpha * niche_size
