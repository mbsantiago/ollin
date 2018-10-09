"""Constants module for default values.

This module holds all default values for simulator configuration.

Attributes
----------
GLOBAL_CONSTANTS : dict
    This dictionary holds all default values unrelated to movement generation,
    such as the site random generator, default camera configuration
    variables, etc.

        :range: 20
        :min_clusters: 2
        :max_clusters: 10
        :min_cluster_points: 1
        :max_cluster_points: 10
        :max_iters: 10
        :bandwidth_epsilon: 0.01
        :resolution: 0.1
        :days: 365
        :cone_range: 0.01
        :cone_angle: 60
        :season: 90

MOVEMENT_PARAMETERS : dict
    This dictionary holds default values for any movement model. When extending
    the :py:class:`.MovementModel` this
    parameters will be added to the model parameters if no replacement is
    provided.

        :hr_days: 365
        :steps_per_day: 4
        :velocity:
            :alpha: 0.0
            :beta: 1.0
        :home_range:
            :alpha: 35.0
            :exponent: 0.54
        :density:
            :alpha: 0.0
            :hr_exp: 1.2
            :density_exp: 1.75
            :niche_size_exp: 1.75
        :movement:
            empty


"""

GLOBAL_CONSTANTS = {
    'range': 20,
    'min_clusters': 2,
    'max_clusters': 10,
    'min_cluster_points': 1,
    'max_cluster_points': 10,
    'max_iters': 10,
    'bandwidth_epsilon': 0.01,
    'resolution': 0.1,
    'days': 365,
    'cone_range': 0.01,
    'cone_angle': 60,
    'season': 90,
}

# CONSTANTS FOR MOVEMENT MODELS
MOVEMENT_PARAMETERS = {
    'hr_days': 365,
    'steps_per_day': 4,
    'velocity': {
        'alpha': 0.0,
        'beta': 1.0},
    'home_range': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 0.0,
        'hr_exp': 1.2,
        'density_exp': 1.75,
        'niche_size_exp': 1.75},
    'movement': {}
}
