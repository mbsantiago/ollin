"""Constants for simulator"""
# INITIAL CONDITION CONSTANTS
RANGE = 20
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 10
MAX_ITERS = 10

# GLOBAL HOME RANGE CONSTANTS
HR_DAYS = 365

# MOVEMENT DATA DEFAULTS
DAYS = 365

# GLOBAL VELOCITY CONSTANTS
HR2VEL_EXPONENT = 0.54

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
    'max_iters': MAX_ITERS,
    'hr_days': HR_DAYS,
    'days': DAYS,
    'steps_per_day': STEPS_PER_DAY,
    'cone_range': CONE_RANGE,
    'cone_angle': CONE_ANGLE,
    'season': SEASON,
    'hr2vel_exponent': HR2VEL_EXPONENT,
}

# CONSTANTS FOR MOVEMENT MODELS
# --- CONSTANT BROWNIAN MODEL ---
CBM_PARAMETERS = {
    'velocity': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 22.0,
        'hr_exp': 1.2,
        'occ_exp': 1.75},
    'movement': {},
}

# --- VARIABLE BROWNIAN MODEL ---
VBM_PARAMETERS = {
    'velocity': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 22.0,
        'hr_exp': 1.2,
        'occ_exp': 1.75},
    'movement': {
        'niche_weight': 0.2},
}

# --- GRADIENT BROWNIAN MODEL ---
GBM_PARAMETERS = {
    'velocity': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 22.0,
        'hr_exp': 1.2,
        'occ_exp': 1.75},
    'movement': {
        'grad_weight': 0.5},
}

# --- CONSTANT LEVY MODEL ---
CLM_PARAMETERS = {
    'velocity': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 22.0,
        'hr_exp': 1.2,
        'occ_exp': 1.75},
    'movement': {
        'pareto': 1.8},
}


# --- VARIABLE LEVY MODEL ---
VLM_PARAMETERS = {
    'velocity': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 22.0,
        'hr_exp': 1.2,
        'occ_exp': 1.75},
    'movement': {
        'min_pareto': 1.1,
        'max_pareto': 1.9},
}

# --- GRADIENT LEVY MODEL ---
GLM_PARAMETERS = {
    'velocity': {
        'alpha': 35.0,
        'exponent': 0.54},
    'density': {
        'alpha': 22.0,
        'hr_exp': 1.2,
        'occ_exp': 1.75},
    'movement': {
        'min_pareto': 1.1,
        'max_pareto': 1.9,
        'grad_weight': 2.0},
}


def handle_global_constants(params):
    copy = GLOBAL_CONSTANTS.copy()
    copy.update(params)
    return copy


def add_missing_entries(dict1, dict2):
    for key in dict1:
        if key in dict2:
            if isinstance(dict1[key], dict):
                copy = dict1[key].copy()
                copy.update(dict2[key])
                dict2[key] = copy
        else:
            dict2[key] = dict1[key]
    if 'global' in dict2:
        dict2['global'] = handle_global_constants(dict2['global'])
    else:
        dict2['global'] = GLOBAL_CONSTANTS
    return dict2


def handle_movement_parameters(params, model):
    if params is None:
        params = {}
    if model == 'Constant Brownian Model':
        params = add_missing_entries(CBM_PARAMETERS, params)
    elif model == 'Variable Brownian Model':
        params = add_missing_entries(VBM_PARAMETERS, params)
    elif model == 'Gradient Brownian Model':
        params = add_missing_entries(GBM_PARAMETERS, params)
    elif model == 'Constant Levy Model':
        params = add_missing_entries(CLM_PARAMETERS, params)
    elif model == 'Variable Levy Model':
        params = add_missing_entries(VLM_PARAMETERS, params)
    elif model == 'Gradient Levy Model':
        params = add_missing_entries(GLM_PARAMETERS, params)
    return params
