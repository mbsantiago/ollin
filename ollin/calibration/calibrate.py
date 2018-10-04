import os
import numpy as np
import logging

from .home_range import HomeRangeCalibrator
from .occupancy import OccupancyCalibrator
from .velocity import VelocityCalibrator

from ..movement_models.basemodel import MovementModel

logger = logging.getLogger(__name__)

STARTING_PARAMETERS = {
    'velocity': {
        'alpha': 0.0,
        'beta': 1.0},
}

BASE_CONFIG = {
    'num_worlds': 10,
    'trials_per_world': 1000,
    'velocities': np.linspace(0.1, 1.5, 6),
    'niche_sizes': np.linspace(0.2, 0.9, 4),
    'home_ranges': np.linspace(0.1, 3, 6),
    'range': 20,
    'season': 90,
    'max_individuals': 10000,
}


def calibrate(
        model,
        config=None,
        save_fig=True,
        save_path=None,
        plot_style='fivethirtyeight'):
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
    vel_calibrator = VelocityCalibrator(model, **calibration_config)
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
    hr_calibrator = HomeRangeCalibrator(model, **calibration_config)
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
    oc_calibrator = OccupancyCalibrator(model, **calibration_config)
    oc_parameters = oc_calibrator.fit()
    model.parameters['occupancy'] = oc_parameters

    if save_fig:
        logger.info('Saving occupancy calibration plot.')
        with plt.style.context(plot_style):
            ax = oc_calibrator.plot()
            path = os.path.join(save_path, 'occupancy_calibration.png')
            ax.get_figure().savefig(path, frameon=True)

    return model, model.parameters
