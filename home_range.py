from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import movement
from constants import (ALPHA, BETA, DT, RANGE)

PARAMETERS = {
    'alpha': ALPHA,
    'beta': BETA,
    'dt': DT,
    'range': RANGE
}


@contextmanager
def set_constants(alpha=ALPHA, range=RANGE, dt=DT, beta=BETA):
    global PARAMETERS
    tmp_constants = PARAMETERS
    PARAMETERS['alpha'] = alpha
    PARAMETERS['beta'] = beta
    PARAMETERS['dt'] = dt
    PARAMETERS['range'] = range
    yield
    PARAMETERS = tmp_constants


def make_grid(home_range, constants=PARAMETERS):

    dt = PARAMETERS['dt']
    alpha = PARAMETERS['alpha']
    range = PARAMETERS['range']
    beta = PARAMETERS['beta']

    velocity = movement.home_range_to_velocity(home_range, beta=beta, dt=dt)
    dx = max(velocity, 0.01)
    mov_data = movement.make_data(
        velocity, num=1, steps=dt, alpha=alpha, range=range)['data']
    mov = mov_data['data'][0]

    shape = int(np.ceil(range/float(dx)))
    indices = np.floor_divide(mov, dx).astype(np.int)

    trace = np.zeros([shape, shape])
    for index in indices:
        trace[index[0], index[1]] = 1
    return trace


def calculate(mobility, constants=PARAMETERS):
    dx = PARAMETERS['dx']
    grid = make_grid(mobility, constants)
    area = np.sum(grid) * (dx ** 2)
    return area


def estimate(mobility, N, alpha=ALPHA, range=RANGE, dt=DT, beta=BETA):
    with set_constants(alpha=alpha, range=range, dt=dt, beta=beta):
        p = Pool()
        results = p.map(calculate, [mobility for _ in xrange(N)])
        p.close()
        p.join()
    return results


def MSE(mobility, N, alpha=ALPHA, range=RANGE, dt=DT, beta=BETA):
    estimation = estimate(
        mobility, N, alpha=alpha, range=range, dt=dt, beta=beta)
    mse = np.mean((np.array(estimation) - mobility)**2)
    return mse
