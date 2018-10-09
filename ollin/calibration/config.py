"""Module for basic configuration for calibration procedure.

Attributes
----------
STARTING_PARAMETERS : dict
    Initial parameters for models starting the calibration procedure.

BASE_CONFIG : dict
    Configuration for calibration procedure.

    :days: 365

        Number of days to simulate in each scenario.
    :season: 90

        Days to use in occupancy calibration. Occupancy is calculated for this
        number of days.
    :num_worlds: 10

        Number of different sites at which to simulate each configuration.
    :range: (20, 20)

        Size of sites.
    :trials_per_world: 100

        Number of simulations per site.
    :max_individuals: 10000

        Maximum number of simulated individuals.
    :velocities: [0.1, 0.38, 0.66, 0.94, 1.22, 1.5]

        Array of mean velocities to simulate.
    :niche_sizes: [0.2, 0.43, 0.67, 0.9]

        Array of niche sizes to simulate.
    :home_ranges: [0.1, 0.68, 1.26, 1.84, 2.42, 3.0]

        Array of home ranges to simulate.
    :nums: [10, 208, 406, 604, 802, 1000]

        Array of number of individuals to simulate.

"""
import numpy as np


STARTING_PARAMETERS = {
    'velocity': {
        'alpha': 0.0,
        'beta': 1.0},
    }

BASE_CONFIG = {
    'days': 365,
    'season': 90,
    'num_worlds': 10,
    'range': (20, 20),
    'trials_per_world': 100,
    'max_individuals': 10000,
    'velocities': np.linspace(0.1, 1.5, 6).tolist(),
    'niche_sizes': np.linspace(0.2, 0.9, 4).tolist(),
    'home_ranges': np.linspace(0.1, 3, 6).tolist(),
    'nums': np.linspace(10, 1000, 6, dtype=np.int64).tolist(),
}
