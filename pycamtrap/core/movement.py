"""Module defining Movement Data class and Movement Analyzer

Either simulated data or data incoming for real telemetry data can be stored in
a :py:class:`MovementData` object. The main information stored in such an
object is the whole history of individual positions, arranged in an array of
shape [num_individuals, time_steps, 2]. This information can then be
plotted for trajectory visualization, or used in further processing.

A specialized type of Movement data is data produced by some movement model
simulation. Information of movement data and movement model information can
be packed into a :py:class:`Movement` object.

Movement analysis, such as distribution of velocities, heading angles and
turning angles can be extracted and stored in an :py:class:`MovementAnalyzer`
object.
"""
from __future__ import division
from __future__ import print_function
import copy

import numpy as np
from cycler import cycler

from ..movement_models.basemodel import MovementModel
from .constants import GLOBAL_CONSTANTS
from ..movement_models import get_movement_model
from .utils import (occupancy_to_density,
                    home_range_to_velocity,
                    velocity_modification)


class MovementData(object):
    """Container for Movement data.

    All animal movement data can be stored in an array of shape of shape
    [num_individuals, time_steps, 2] which represents the positions of every
    individual along some time interval. If::

        x = array[i, j, 0]
        y = array[i, j, 1]

    then the i-th individual was at the place with (x, y)-coordinates at the
    j-th time step.

    Apart from spatial information, times at which the timesteps where taken
    are store in another array of shape [time_steps].

    Attributes
    ----------
    site : :py:obj:`pycamtrap.Site`
        Information of Site at which movement took place.
    movement_data : array
        Array of shape [num_individuals, time_steps, 2] holding coordinate
        information of individual location through movement.
    times : array
        Array of shape [time_steps] with time at which the time steps took
        place. Units are in days.
    home_range : float or None
        Home range value of species. Only necessary for occupancy calculation.
        See :py:class:`pycamtrap.Occupancy`.

    """
    def __init__(self, site, movement_data, times, home_range=None):
        """Construct Movement Data object.

        Arguments
        ---------
        site : :py:obj:`pycamtrap.Site`
            Information of Site at which movement took place.
        movement_data : array
            Array of shape [num_individuals, time_steps, 2] holding coordinate
            information of individual location through movement.
        times : array
            Array of shape [time_steps] with time at which the time steps took
            place. Units are in days.
        home_range : float, optional
            Home range value of species. Only necessary for occupancy
            calculation. See :py:class:`pycamtrap.Occupancy`.

        """
        self.site = site
        self.data = movement_data
        self.times = times
        self.home_range = home_range
        self.num, self.steps, _ = movement_data.shape

    def num_slice(self, key):
        """Extract motion from slice of individuals.

        Select a subset of individuals from motion data using a
        slice.

        Arguments
        ---------
            key : int or list or tuple or slice
                If key is an integer the result will be a
                :py:obj:`MovementData` object holding only motion data for the
                corresponding individual. If key is a list or tuple, its
                contents will be passed to the :py:func:`slice` function, and
                said slice will be extracted from data array in the first axis,
                and returned in an :py:obj:`MovementData` object.

        Returns
        -------
            newcopy : :py:obj:`MovementData`
                New :py:obj:`MovementData` object sharing site and times
                attributes but with movement data corresponding to individuals
                slice.

        Example
        -------
        To extract the movement of the first ten individuals::

            first_ten = movement.num_slice((None, 10, None))

        To extract the movement of the last 20 individuals::

            last_20 = movement.num_slice((-20, None, None))

        To extract all even individuals::

            even = movement.num_slice((None, None, 2))

        """
        if not isinstance(key, (int, slice)):
            if isinstance(key, (list, tuple)):
                key = slice(*key)
            else:
                msg = 'Num slice only accepts (int/list/tuple/slice) as'
                msg += ' arguments. {} given.'.format(type(key))
                raise ValueError(msg)
        data = self.data[key, :, :]

        newcopy = copy.copy(self)
        newcopy.data = data
        newcopy.num, newcopy.steps, _ = data.shape
        return newcopy

    def sample(self, num):
        """Extract a sample of individual movement.

        Select a random sample of individuals of a given size to form a new
        :py:obj:`MovementData` object.

        Arguments
        ---------
        num : int
            Size of sample

        Returns
        -------
        newcopy : :py:obj:`MovementData`
            Movement data corresponding to sample.

        """
        selection = np.random.choice(
                np.arange(self.num),
                size=num)
        data = self.data[selection, :, :]
        newcopy = copy.copy(self)
        newcopy.data = data
        newcopy.num, newcopy.steps, _ = data.shape
        return newcopy

    def select(self, selection):
        """Select a subset of individual movement.

        Use an array of indices to select a subset of individuals and return
        movement data of the corresponding individuals.

        Arguments
        ---------
        selection : array or tuple or list
            List of indices of selected individuals

        Returns
        -------
        newcopy : :py:obj:`MovementData`
            Movement data of selected individuals.

        """
        if isinstance(selection, (tuple, list)):
            selection = np.array(selection)
        data = self.data[selection, :, :]
        newcopy = copy.copy(self)
        newcopy.data = data
        newcopy.num, newcopy.steps, _ = data.shape
        return newcopy

    def time_slice(self, key):
        """Select a slice of timesteps from movement.

        Arguments
        ---------
        key : int or list or tuple or slice
            If key is integer the resulting :py:obj:`MovementData` object will
            only hold the individuals position at the corresponding timestep.
            If key is list or tuple, its contents will be passed to the
            :py:func:slice function and the slice will be used to extract some
            times steps from the movement data.

        Returns
        -------
        newcopy : :py:obj:`MovementData`
            Movement data with selected time steps.

        Example
        -------
        To select the first 10 days of movement::

            first_10_days = movement_data.time_slice((None, 10, None))

        To select the last 20 days of movement::

            last_20_days = movement_data.time_slice((-20, None, None))

        To select every other step::

            every_other = movement_data.time_slice((None, None, 2))

        """
        if not isinstance(key, (int, slice)):
            if isinstance(key, (list, tuple)):
                key = slice(*key)
            else:
                msg = 'Time slice only accepts (int/list/tuple/slice) as'
                msg += ' arguments. {} given.'.format(type(key))
                raise ValueError(msg)

        data = self.data[:, key, :]
        newcopy = copy.copy(self)
        newcopy.data = data
        newcopy.num, newcopy.steps, _ = data.shape
        return newcopy

    def plot(
            self,
            ax=None,
            figsize=(10, 10),
            include=None,
            num=10,
            steps=1000,
            mov_cmap='Greens',
            simplify=None,
            **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        if include is None:
            include = [
                    'niche',
                    'niche_boundary',
                    'rectangle',
                    'trajectories']

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

        self.site.plot(
                include=include, ax=ax, **kwargs)

        if 'trajectories' in include:
            cmap = plt.get_cmap(mov_cmap)
            colors = [cmap(i) for i in np.linspace(0.05, .8, 10)]
            ax.set_prop_cycle(cycler('color', colors))

            steps = min(self.steps, steps)

            if simplify is None:
                stride = 1
            else:
                stride = max(int(steps / simplify), 1)
            trajectories = self.data[:num, :steps:stride, :]

            for trajectory in trajectories:
                xcoord, ycoord = zip(*trajectory)
                ax.plot(xcoord, ycoord)

        return ax


class Movement(MovementData):
    def __init__(
            self,
            site,
            movement_data,
            movement_model,
            velocity,
            home_range=None):

        self.movement_model = movement_model
        self.velocity = velocity

        num, steps, _ = movement_data.shape
        steps_per_day = movement_model.parameters['steps_per_day']
        days = steps / steps_per_day
        times = np.linspace(0, days, steps)

        super(Movement, self).__init__(
            site, movement_data, times, home_range=home_range)

    @classmethod
    def simulate(
            cls,
            site,
            days=None,
            num=None,
            occupancy=None,
            home_range=None,
            velocity=None,
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
            rangex, rangey = site.range
            if home_range is None:
                msg = 'If num is not specified home range AND occupancy'
                msg += ' must be provided'
                raise ValueError(msg)
            dens = occupancy_to_density(
                occupancy,
                home_range,
                site.niche_size,
                site.range,
                parameters=parameters['density'])
            num = int(rangex * rangey * dens)

        if days is None:
            days = GLOBAL_CONSTANTS['days']

        velocity_mod = velocity_modification(
            site.niche_size, parameters)
        steps_per_day = parameters['steps_per_day']
        sim_velocity = velocity * velocity_mod / steps_per_day

        steps = int(days * steps_per_day)

        initial_positions = site.sample(num)
        movement_data = movement_model.generate_movement(
            initial_positions,
            site,
            steps,
            sim_velocity)

        return cls(
            site,
            movement_data,
            movement_model,
            velocity,
            home_range=home_range)

    def extend(self, days, inplace=True):
        parameters = self.movement_model.parameters
        steps_per_day = parameters['steps_per_day']
        steps = int(steps_per_day * days)

        velocity_mod = velocity_modification(
            self.site.niche_size, parameters)
        velocity = self.velocity * velocity_mod / steps_per_day

        initial_positions = self.data[:, -1, :]

        new_data = self.movement_model.generate_movement(
            initial_positions,
            self.site,
            steps + 1,
            velocity)
        data = np.append(
                self.data, new_data[:, 1:, :], 1)

        old_steps = self.data.shape[1]
        total_days = (old_steps + steps) / steps_per_day
        times = np.linspace(0, total_days, old_steps + steps)

        if inplace:
            extension = self
        else:
            extension = copy.copy(self)

        extension.data = data
        extension.times = times
        return extension


class MovementAnalyzer(object):
    def __init__(self, mov):
        self.movement = mov
        self.velocities, self.bearings, self.turn_angles = self.analyze()

    @property
    def mean_velocity(self):
        return self.velocities.mean(axis=1).mean()

    def analyze(self):
        num, steps, _ = self.movement.data.shape
        data = self.movement.data.reshape([-1, steps, 2])
        times = self.movement.times

        dtimes = (times[1:] - times[:-1])[None, :, None]
        directions = (data[:, 1:, :] - data[:, :-1, :]) / dtimes
        complex_directions = directions[:, :, 0] + 1j * directions[:, :, 1]
        velocities = np.abs(complex_directions)
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
        return ax
