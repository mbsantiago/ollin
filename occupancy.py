import os
import numpy as np

from constants import *


def make_grid(data, dx=DX):
    range = data['range']
    bins = int(np.ceil(range / dx))

    mov_data = data['data']
    length = mov_data.shape[1]
    
    stack = []
    for t in xrange(length):
        array = np.zeros([bins, bins])
        indices = np.floor_divide(mov_data[:, t, :], dx).reshape([-1,2]).astype(np.int)
        X = indices[:,0]
        Y = indices[:,1]
        array[X, Y] = 1
        stack.append(array)
    grid = np.stack(stack)
    return grid


def plot_grid(grid, t=0):
    fig, ax = plt.subplots()
    ax.pcolormesh(grid[:,:,0])
    return fig


def calculate(data, dx=DX, season=SEASON):
    grid = make_grid(data, dx=dx)
    nsteps = max(len(grid) - season, 1)
    values = np.array([np.mean(np.amax(grid[i: i + season], axis=0)) for i in xrange(nsteps)])
    return values


def estimate(detection_data):
    pass
