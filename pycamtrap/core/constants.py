"""Constants for simulator"""
# INITIAL CONDITION CONSTANTS
RANGE = 20
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 10
MAX_ITERS = 10
RESOLUTION = 0.1

# GLOBAL HOME RANGE CONSTANTS
HR_DAYS = 365

# MOVEMENT DATA DEFAULTS
DAYS = 365

# GLOBAL MOVEMENT CONSTANTS
STEPS_PER_DAY = 4

# GLOBAL CAMERA CONSTANTS
CONE_RANGE = 0.01
CONE_ANGLE = 60

# GLOBAL OCCUPANCY CONSTANTS
SEASON = 90

GLOBAL_CONSTANTS = {
    'range': RANGE,
    'min_clusters': MIN_CLUSTERS,
    'max_clusters': MAX_CLUSTERS,
    'min_neighbors': MIN_NEIGHBORS,
    'max_neighbors': MAX_NEIGHBORS,
    'resolution': RESOLUTION,
    'max_iters': MAX_ITERS,
    'hr_days': HR_DAYS,
    'days': DAYS,
    'cone_range': CONE_RANGE,
    'cone_angle': CONE_ANGLE,
    'season': SEASON,
    'steps_per_day': STEPS_PER_DAY,
}

# CONSTANTS FOR MOVEMENT MODELS
MOVEMENT_PARAMETERS = {
    'velocity': {
        'alpha': 0.0,
        'beta': 1.0},
    'home_range': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 22.0,
        'hr_exp': 1.2,
        'occ_exp': 1.75},
    'movement': {}
}


def handle_global_constants(params):
    if params is None:
        params = {}
    copy = GLOBAL_CONSTANTS.copy()
    copy.update(params)
    return copy
