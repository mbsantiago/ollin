"""Constants for simulator"""
# UNITS OF MEASUREMENT
DT = 365
DX = 0.2

# INITIAL CONDITIONS
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 10
MAX_ITERS = 10

# MOVEMENT CONSTANTS
RANGE = 22
MAX_INDIVIDUALS = 10000
STEPS_PER_DAY = 4
DAYS = 365
BETA = 35
POWER = 0.54
MIN_VELOCITY = 0.01
MAX_POINTS = 10000

# -- Constant Levy
ALPHA = 1.8

# -- Variable Levy
MIN_ALPHA = 1.1
MAX_ALPHA = 1.9

# -- Variable Brownian
DEV = 0.5

# DENSITY CONSTANTS
OMEGA = 1.2
GAMMA = 22.0
TAU = 1.75
SEASON = 90

# CAMERA CONSTANTS
CONE_RANGE = 0.005
CONE_ANGLE = 60


def handle_parameters(params):
    if params is None:
        params = {}
    all_params = {
        'DT': params.get('DT', DT),
        'DX': params.get('DX', DX),
        'MIN_CLUSTERS': params.get('MIN_CLUSTERS', MIN_CLUSTERS),
        'MAX_CLUSTERS': params.get('MAX_CLUSTERS', MAX_CLUSTERS),
        'MIN_NEIGHBORS': params.get('MIN_NEIGHBORS', MIN_NEIGHBORS),
        'MAX_NEIGHBORS': params.get('MAX_NEIGHBORS', MAX_NEIGHBORS),
        'MAX_ITERS': params.get('MAX_ITERS', MAX_ITERS),
        'RANGE': params.get('RANGE', RANGE),
        'MAX_INDIVIDUALS': params.get('MAX_INDIVIDUALS', MAX_INDIVIDUALS),
        'STEPS_PER_DAY': params.get('STEPS_PER_DAY', STEPS_PER_DAY),
        'DAYS': params.get('DAYS', DAYS),
        'BETA': params.get('BETA', BETA),
        'POWER': params.get('POWER', POWER),
        'MIN_VELOCITY': params.get('MIN_VELOCITY', MIN_VELOCITY),
        'MAX_POINTS': params.get('MAX_POINTS', MAX_POINTS),
        'OMEGA': params.get('OMEGA', OMEGA),
        'GAMMA': params.get('GAMMA', GAMMA),
        'TAU': params.get('TAU', TAU),
        'SEASON': params.get('SEASON', SEASON),
        'CONE_RANGE': params.get('CONE_RANGE', CONE_RANGE),
        'CONE_ANGLE': params.get('CONE_ANGLE', CONE_ANGLE),
        'MIN_ALPHA': params.get('MIN_ALPHA', MIN_ALPHA),
        'MAX_ALPHA': params.get('MAX_ALPHA', MAX_ALPHA),
        'ALPHA': params.get('ALPHA', ALPHA),
        'DEV': params.get('DEV', DEV),
    }
    params.update(all_params)
    return params
