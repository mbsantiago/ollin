from __future__ import print_function
from multiprocessing import Pool

import sys
import numpy as np
import pycamtrap as pc

TRIALS_PER_WORLD = 1000
NUM_WORLDS = 10


HOME_RANGES = [0.5, 0.8, 1.2, 1.9, 2.5, 3.5, 4.8, 8.1, 12.0, 19.0]
OCCUPANCIES = np.linspace(0.2, 0.9, 5)

PARAMETERS = None


def get_single_home_range_info(
        movement_model,
        home_range,
        occupancy):
    init = pc.InitialCondition(
        occupancy,
        home_range)
    mov = pc.MovementData.simulate(
        init,
        num=TRIALS_PER_WORLD,
        days=365,
        parameters=PARAMETERS)
    hr = pc.HomeRange(mov)
    return hr.home_ranges


def _aux(argument):
    mov_model, hr, occ, nw = argument
    home_range = HOME_RANGES[hr]
    occupancy = OCCUPANCIES[occ]
    return get_single_home_range_info(mov_model, home_range, occupancy)


def get_all_home_range_info(movement_model, parameters=None):
    global PARAMETERS
    PARAMETERS = parameters

    n_hr = len(HOME_RANGES)
    n_oc = len(OCCUPANCIES)
    all_info = np.zeros([n_hr, n_oc, NUM_WORLDS, TRIALS_PER_WORLD])
    arguments = [
        (movement_model, i, j, k)
        for i in xrange(n_hr)
        for j in xrange(n_oc)
        for k in xrange(NUM_WORLDS)
    ]

    msg = 'Starting parallel processing. N_processes={}'.format(len(arguments))
    print(msg)
    pool = Pool()
    try:
        results = pool.map(_aux, arguments)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        sys.exit()
        quit()
    print('done')

    for arg, res in zip(arguments, results):
        _, i, j, k = arg
        all_info[i, j, k, :] = res

    return all_info
