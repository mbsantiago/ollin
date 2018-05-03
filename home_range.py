from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import movement
from constants import *

CONSTANTS = {
    'alpha': ALPHA,
    'beta': BETA,
    'dx': DX,
    'dt': DT,
    'range': RANGE
}

@contextmanager
def set_constants(alpha=ALPHA, range=RANGE, dx=DX, dt=DT, beta=BETA):
    global CONSTANTS 
    tmp_constants = CONSTANTS
    CONSTANTS['alpha'] = alpha
    CONSTANTS['beta'] = beta 
    CONSTANTS['dx'] = dx 
    CONSTANTS['dt'] = dt 
    CONSTANTS['range'] = range 
    yield
    CONSTANTS = tmp_constants


def make_grid(mobility, constants=CONSTANTS):
    dt = CONSTANTS['dt']
    dx = CONSTANTS['dx']
    alpha = CONSTANTS['alpha']
    range = CONSTANTS['range']
    beta = CONSTANTS['beta']

    mov_data = movement.make_data(mobility, num=1, steps=dt, alpha=alpha, range=range, beta=beta)
    mov = mov_data['data'][0]

    shape = int(np.ceil(range/float(dx)))
    indices = np.floor_divide(mov, dx).astype(np.int)

    trace = np.zeros([shape, shape])
    for index in indices:
        trace[index[0], index[1]] = 1
    return trace
   

def calculate(mobility, constants=CONSTANTS):
    dx = CONSTANTS['dx']
    grid = make_grid(mobility, constants) 
    area = np.sum(grid) * (dx **2)
    return area


def estimate(mobility, N, alpha=ALPHA, range=RANGE, dx=DX, dt=DT, beta=BETA):
    with set_constants(alpha=alpha, range=range, dx=dx, dt=dt, beta=beta):
        p = Pool()
        results = p.map(calculate, [mobility for _ in xrange(N)])
        p.close()
        p.join()
    return results

def MSE(mobility, N, alpha=ALPHA, range=RANGE, dx=DX, dt=DT, beta=BETA):
    estimation = estimate(mobility, N, alpha=alpha, range=range, dx=dx, dt=dt, beta=beta)
    mse = np.mean((np.array(estimation) - mobility)**2)
    return mse
