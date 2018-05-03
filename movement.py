import argparse
import os
import time

import numpy as np

from constants import *

def make_plot(name, data, num=10, show=False, time=365):
    import matplotlib.pyplot as plt
    
    plotdata = data['data'][:num,:time]

    range = data['range']
    ambito = data['mobility']
    alpha = data['alpha']
    rango = data['range'] 
    beta = data['beta']

    fig, ax = plt.subplots()
    for traj in plotdata:
        X, Y = zip(*traj)
        ax.plot(X,Y)
    rect = plt.Rectangle([0, 0], range, range, fill=False)
    ax.add_patch(rect)

    title = 'Especie: {}\nAmbito: {} km2;  Rango: {} km;  Alpha: {};  Beta: {};'.format(
        name,
        ambito,
        rango,
        alpha,
        beta)
    plt.title(title)

    extra = 0.05 * range
    plt.xlim(-extra, range + extra)
    plt.ylim(-extra, range + extra)

    plt.axis('off')
    if show:
        plt.show()
    else:
        return fig


def make_data(mobility, num=MAX_INDIVIDUALS, steps=STEPS, alpha=ALPHA, range=RANGE, beta=BETA):
    random_positions = np.random.uniform(0, range, [num, 2])

    mean = beta * (mobility * (alpha - 1)/alpha) / 365.0
    stack = [random_positions]
    for _ in xrange(steps):
        random_angles = np.random.uniform(0, 2 * np.pi, [num])
        random_directions = np.stack([np.cos(random_angles), np.sin(random_angles)], axis=-1)
        random_magnitudes = mean * np.random.pareto(alpha, [num])
        random_directions *= random_magnitudes[:, None]

        tmp1 = random_positions + random_directions
        tmp2 = np.mod(tmp1, 2 * range)
        reflections = np.greater(tmp2, range)
        tmp3 = (1 - reflections) * np.mod(tmp2, range) + reflections * np.mod(-tmp2, range)
        random_positions = tmp3

        stack.append(random_positions)
    data = {
        'data': np.stack(stack,1),
        'alpha': alpha,
        'range': range,
        'mobility': mobility,
        'beta': beta
    }
    return data


def save_data(name, data):
    path = os.path.join('species_movement', 'rango_' + str(data['range']), 'alpha_' + str(data['alph']), name + '.npy')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.save(path, data['data'])


def make_and_save(info):
    key, value = info 
    print('Making movement data for species {}'.format(key))
    print('Species mobility: {}'.format(value['ambito']))
    data = make_data()
    save_data(name, data)
    print('Species {} movement data saved'.format(name))


def parse_arguments():
    p = argparse.ArgumentParser('Make many species movement data')
    p.add_argument('--plot', action='store_true', help='display plot of trajectories.')
    p.add_argument('--all', action='store_true', help='process all species.')
    p.add_argument('species', nargs='?', type=int, help='Species index.')
    return p.parse_args()


def main():
    from species_data import SPECIES

    flags = parse_arguments()

    if flags.species is None and not flags.all:
        raise ValueError('No species provided')
    else:
        if not flags.all:
            name = SPECIES.keys()[flags.species]
            mobility = SPECIES[name]['ambito']
            print('Making movement data for species {}'.format(name))
            print('Species mobility: {}'.format(mobility))

            data = make_data(mobility)
            if flags.plot:
                make_plot(name, data)
            else:
                save_data(name, data)
                print('Species {} movement data saved'.format(name))
        else:
    	    from multiprocessing import Pool
    	    p = Pool()
            p.imap(make_and_save, SPECIES.iteritems())
	    p.close()
	    p.join()


if __name__ == '__main__':
    main()
