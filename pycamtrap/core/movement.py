from __future__ import division
from __future__ import print_function
from importlib import import_module
import os

import numpy as np
from cycler import cycler
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

from .utils import (occupancy_to_density,
                    home_range_to_velocity,
                    velocity_modification)
from ..movement_models.basemodel import MovementModel


@lru_cache()
def load_movement_model(model):
    model_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(
        model_path, 'movement_models', '{}.py'.format(model))

    if os.path.exists(model_path):
        try:
            cls = import_module(
                'pycamtrap.movement_models.{}'.format(model)).Model
            return cls
        except Exception as e:
            print('Unexpected exception occured while loading model file')
            raise e


def get_movement_model(model, parameters=None):
    cls = load_movement_model(model)
    return cls(parameters)


class MovementData(object):
    def __init__(
            self,
            initial_conditions,
            movement_data,
            movement_model,
            velocity,
            num,
            home_range=None,
            occupancy=None):

        self.initial_conditions = initial_conditions
        self.movement_model = movement_model
        self.data = movement_data

        self.num_experiments, self.num, self.steps, _ = movement_data.shape
        steps_per_day = movement_model.parameters['steps_per_day']
        self.days = self.steps / steps_per_day

        self.num = num
        self.velocity = velocity
        self.home_range = home_range
        self.occupancy = occupancy

    @classmethod
    def simulate(
            cls,
            initial_conditions,
            days=None,
            num=None,
            occupancy=None,
            home_range=None,
            velocity=None,
            num_experiments=1,
            parameters=None,
            movement_model='variable_levy'):

        if not isinstance(movement_model, MovementModel):
            movement_model = get_movement_model(
                movement_model,
                parameters=parameters)
        parameters = movement_model.parameters

        if velocity is None:
            if home_range is None:
                msg = 'Arguments velocity or home_range must be provided'
                raise ValueError(msg)
            velocity = home_range_to_velocity(
                home_range,
                parameters=parameters['home_range'])

        if num is None:
            if occupancy is None:
                msg = 'Arguments num or occupancy must be provided'
                raise ValueError(msg)
            rangex, rangey = initial_conditions.range
            if home_range is None:
                msg = 'If num is not specified home range AND occupancy'
                msg += ' must be provided'
                raise ValueError(msg)
            dens = occupancy_to_density(
                occupancy,
                home_range,
                initial_conditions.niche_size,
                initial_conditions.range,
                parameters=parameters['density'])
            num = int(rangex * rangey * dens)

        if days is None:
            days = parameters['days']

        velocity_mod = velocity_modification(
                initial_conditions.niche_size, parameters)
        steps_per_day = parameters['steps_per_day']
        num_ = int(num * num_experiments)
        steps = int(days * steps_per_day)
        initial_positions = initial_conditions.sample(num_)
        movement_data = movement_model.generate_movement(
            initial_positions,
            initial_conditions,
            days,
            velocity * velocity_mod).reshape(
                [num_experiments, num, steps, 2])

        return cls(
            initial_conditions,
            movement_data,
            movement_model,
            velocity,
            num,
            home_range=home_range,
            occupancy=occupancy)

    def num_slice(self, key):
        if not isinstance(key, (int, slice)):
            if isinstance(key, (list, tuple)):
                key = slice(*key)
            else:
                msg = 'Num slice only accepts (int/list/tuple/slice) as'
                msg += ' arguments. {} given.'.format(type(key))
                raise ValueError(msg)
        data = self.data[:, key, :, :]
        num = data.shape[1]

        mov = MovementData(
            self.initial_conditions,
            data,
            self.movement_model,
            self.velocity,
            num,
            home_range=self.home_range,
            occupancy=self.occupancy)
        return mov

    def sample(self, num, replace=False):
        selection = np.random.choice(
            np.arange(self.num),
            size=num,
            replace=replace)
        data = self.data[:, selection, :, :]

        mov = MovementData(
            self.initial_conditions,
            data,
            self.movement_model,
            self.velocity,
            num,
            home_range=self.home_range,
            occupancy=self.occupancy)
        return mov

    def select(self, selection):
        if isinstance(selection, (tuple, list)):
            selection = np.array(selection)
        num = selection.size
        data = self.data[:, selection, :, :]

        mov = MovementData(
            self.initial_conditions,
            data,
            self.movement_model,
            self.velocity,
            num,
            home_range=self.home_range,
            occupancy=self.occupancy)
        return mov

    def time_slice(self, key):
        if not isinstance(key, (int, slice)):
            if isinstance(key, (list, tuple)):
                key = slice(*key)
            else:
                msg = 'Time slice only accepts (int/list/tuple/slice) as'
                msg += ' arguments. {} given.'.format(type(key))
                raise ValueError(msg)

        steps_per_day = self.movement_model.parameters['steps_per_day']
        if isinstance(key, int):
            key = steps_per_day * key
        else:
            start = None if key.start is None else key.start * steps_per_day
            end = None if key.end is None else key.end * steps_per_day
            step = None if key.step is None else key.step * steps_per_day
            key = slice(start, end, step)

        data = self.data[:, :, key, :]

        mov = MovementData(
                self.initial_conditions,
                data,
                self.movement_model,
                self.velocity,
                self.num,
                home_range=self.home_range,
                occupancy=self.occupancy)
        return mov

    def plot(
            self,
            include=None,
            num=10,
            days=365,
            ax=None,
            mov_cmap='Greens',
            simplify=None,
            experiment_number=0,
            figsize=(10, 10),
            **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        if include is None:
            include = [
                'heatmap',
                'niche',
                'occupation_zone',
                'rectangle',
                'trajectories']

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        self.initial_conditions.plot(
            include=include, ax=ax, **kwargs)

        if 'trajectories' in include:

            cmap = plt.get_cmap(mov_cmap)
            colors = [cmap(i) for i in np.linspace(0.05, .8, 10)]
            ax.set_prop_cycle(cycler('color', colors))

            steps_per_day = self.movement_model.parameters['steps_per_day']
            num = min(self.num, num)
            steps = min(self.steps, days * steps_per_day)

            if simplify is None:
                stride = 1
            else:
                stride = max(int(steps / simplify), 1)
            experiment_number = max(
                min(experiment_number, self.num_experiments - 1), 0)
            trajectories = self.data[experiment_number, :num, :steps:stride, :]

            for trajectory in trajectories:
                xcoord, ycoord = zip(*trajectory)
                ax.plot(xcoord, ycoord)

        range_ = self.initial_conditions.range
        xticks = np.linspace(0, range_[0], 2)
        yticks = np.linspace(0, range_[1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        return ax


class MovementAnalyzer(object):
    def __init__(self, mov):
        self.movement_data = mov
        self.velocities, self.bearings, self.turn_angles = self.analyze()

    @property
    def mean_velocity(self):
        return self.velocities.mean(axis=1).mean()

    def analyze(self):
        mov = self.movement_data.movement_model
        steps_per_day = mov.parameters['steps_per_day']
        num_experiments, num, steps, _ = self.movement_data.data.shape
        data = self.movement_data.data.reshape([-1, steps, 2])

        directions = (data[:, 1:, :] - data[:, :-1, :])
        complex_directions = directions[:, :, 0] + 1j * directions[:, :, 1]
        velocities = steps_per_day * np.abs(complex_directions)
        bearings = np.angle(complex_directions)
        turn_angles = np.angle(complex_directions[:, 1:] /
                               complex_directions[:, :-1])
        return velocities, bearings, turn_angles

    def plot_velocities(self, num_individual=0, ax=None, bins=20):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        if num_individual == 'all':
            velocities = self.velocities.ravel()
        else:
            velocities = self.velocities[num_individual]

        plt.hist(velocities, bins=bins)
