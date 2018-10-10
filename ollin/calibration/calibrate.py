"""Module for Movement Model calibration.

Every Movement Model is defined in terms of user controllable parameters which
must be calibrated before use. Model calibration refers to the three following
procedures:

    1. Velocity calibration:
        Even though all Movement Models must implement some way of controlling
        agent mean speed, it may be impossible to precisely select the correct
        speed parameter to obtain the desired mean speed, specially for complex
        models with interactions between environment and other individuals.
        Hence a correction to the speed parameter must be calculated to amend
        such shortcomings. Mean speed is calculated in a variety of scenarios
        and a correction factor is calculated by fitting the velocity data of
        the simulated movements to desired outcome.

    2. Home Range calibration.
    3. Occupancy calibration.
        A main feature of :py:mod:`ollin` is its capacity to generate random
        scenarios with a user determined home range and occupancy values. In
        order for this to be possible, the fundamental relations between
        home range and mean velocity, and density and occupancy, must be
        understood. These relations depend on the Movement Model. Home range
        and Occupancy calibration refer to the process in which simulations
        in a variety of scenarios are run and a parametric model of response
        functions are fitted to the data to obtain an approximate functional
        relation between home range and mean velocity, and density and
        occupancy.
"""
import os
import logging

from .home_range import HomeRangeCalibrator
from .occupancy import OccupancyCalibrator
from .velocity import VelocityCalibrator

from ..movement_models.base import MovementModel
from .config import STARTING_PARAMETERS, BASE_CONFIG


logger = logging.getLogger(__name__)


def calibrate(
        model,
        config=None,
        save_fig=False,
        save_path=None,
        plot_style='fivethirtyeight'):
    """Calibrate parameters of movement model.


    Arguments
    ---------
    model : :py:obj:`.MovementModel`
        Movement model instance to calibrate.
    config : dict, optional
        Dictionary holding calibration configuration values. If None is
        given, default configurations will be used. See :py:mod:`.config`
    save_fig : bool, optional
        If True, calibration procedure will save calibration graphs.
    save_path : str, optional
        Path in which to save calibration figures. Must be provided if
        save_fig is True.
    plot_style: str, optional
        Name of pyplot style to use in calibration figures. See
        pyplot_ to see all available options.

    Returns
    ------
    calibrated_model : :py:class:`.MovementModel`
        Same Movement model instance with calibrated parameters.
    calibrated_parameters : dict
        Dictionary holding all calibrated parameters.

    Raises
    ------
    ValueError:
        If no save_path was provided while save_fig is True.


    .. _pyplot: https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html

    """
    # Check if instance of class or class is passed as argument
    if issubclass(model, MovementModel):
        model = model(STARTING_PARAMETERS)
    else:
        params = model.handle_parameters(STARTING_PARAMETERS)
        model.parameters = params

    if (save_fig and save_path is None):
        msg = 'No path was given for calibration figures.'
        raise ValueError(msg)

    # Make sure save path exists
    if save_fig:
        import matplotlib.pyplot as plt
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Handle configs
    if config is None:
        config = {}
    calibration_config = BASE_CONFIG.copy()
    calibration_config.update(config)

    logger.info('Starting Velocity calibration.')
    # Calibrate velocity
    vel_calibrator = VelocityCalibrator(model, calibration_config)
    velocity_parameters = vel_calibrator.fit()
    model.parameters['velocity'] = velocity_parameters

    if save_fig:
        logger.info('Saving velocity calibration plot.')
        with plt.style.context(plot_style):
            ax = vel_calibrator.plot()
            path = os.path.join(save_path, 'velocity_calibration.png')
            ax.get_figure().savefig(path, frameon=True)

    logger.info('Starting Home range calibration.')
    # Calibrate Home Range
    hr_calibrator = HomeRangeCalibrator(model, calibration_config)
    hr_paramters = hr_calibrator.fit()
    model.parameters['home_range'] = hr_paramters

    if save_fig:
        logger.info('Saving home range calibration plot.')
        with plt.style.context(plot_style):
            ax = hr_calibrator.plot()
            path = os.path.join(save_path, 'home_range_calibration.png')
            ax.get_figure().savefig(path, frameon=True)

    logger.info('Starting Occupancy calibration.')
    # Calibrate Occupancy
    oc_calibrator = OccupancyCalibrator(model, calibration_config)
    oc_parameters = oc_calibrator.fit()
    model.parameters['density'] = oc_parameters

    if save_fig:
        logger.info('Saving occupancy calibration plot.')
        with plt.style.context(plot_style):
            ax = oc_calibrator.plot(x_var='density')
            path = os.path.join(
                save_path, 'occupancy_calibration_density.png')
            ax.get_figure().savefig(path, frameon=True)

            ax = oc_calibrator.plot(x_var='home_range')
            path = os.path.join(
                save_path, 'occupancy_calibration_home_range.png')
            ax.get_figure().savefig(path, frameon=True)

            ax = oc_calibrator.plot(x_var='niche_sizes')
            path = os.path.join(
                save_path, 'occupancy_calibration_niche_size.png')
            ax.get_figure().savefig(path, frameon=True)

    return model, model.parameters
