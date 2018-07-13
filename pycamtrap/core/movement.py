from __future__ import division
from __future__ import print_function
from importlib import import_module
import os

import numpy as np
from cycler import cycler

from utils import density, home_range_to_velocity
from ..movement_models.basemodel import MovementModel


def load_movement_model(model, parameters=None, path=None):
    if path is None:
        model_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(
            model_path, 'movement_models', '{}.py'.format(model))
    else:
        model_path = os.path.join(path, '{}.py'.format(model))

    if os.path.exists(model_path):
        try:
            if path is None:
                cls = import_module(
                    'pycamtrap.movement_models.{}'.format(model)).Model
            else:
                # TODO
                msg = 'Custom loads are not implemented yet'
                raise NotImplementedError(msg)
            model = cls(parameters)
            return model
        except Exception as e:
            print('Unexpected exception occured while loading model file')
            raise e
    else:
        msg = 'Model file ({}) not found at ({})'
        msg = msg.format(model + '.py', path)
        raise IOError(msg)


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
            movement_model = load_movement_model(
                    movement_model,
                    parameters=parameters)
        parameters = movement_model.parameters

        if velocity is None:
            if home_range is None:
                msg = 'Arguments velocity or home_range must be provided'
                raise ValueError(msg)
            velocity = home_range_to_velocity(
                home_range,
                parameters=parameters)

        if num is None:
            if occupancy is None:
                msg = 'Arguments num or occupancy must be provided'
                raise ValueError(msg)
            rangex, rangey = initial_conditions.range
            if home_range is None:
                msg = 'If num is not specified home range AND occupancy'
                msg += ' must be provided'
                raise ValueError(msg)
            dens = density(
                    occupancy, home_range, parameters=parameters['density'])
            num = int(rangex * rangey * dens)

        if days is None:
            days = parameters['days']

        num_ = num * num_experiments
        steps_per_day = parameters['steps_per_day']
        steps = days * steps_per_day
        initial_positions = initial_conditions.sample(num_)
        movement_data = movement_model.generate_movement(
            initial_positions,
            initial_conditions,
            days,
            velocity=velocity).reshape(
                [num_experiments, num, steps, 2])

        return cls(
            initial_conditions,
            movement_data,
            movement_model,
            velocity,
            num,
            home_range=home_range,
            occupancy=occupancy)

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

        range = self.initial_conditions.range
        xticks = np.linspace(0, range[0], 2)
        yticks = np.linspace(0, range[1], 2)
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
