"""Module defining Movement Data class and Movement Analyzer

Either simulated data or data incoming from real telemetry data can be stored
in a :py:class:`MovementData` object. The main information held in such an
object is the full history of individual positions arranged in an array of
shape [num_individuals, time_steps, 2]. This information can then be
plotted for trajectory visualization, or used in further processing.

Data produced by simulation can be stored in a specialized type of MovementData
which also holds movement model information. See :py:class:`Movement`.

Movement analysis, such as distribution of velocities, heading angles and
turning angles can be extracted and stored in an :py:class:`MovementAnalysis`
object.
"""
from __future__ import division
from __future__ import print_function
import copy

import numpy as np

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
    site : :py:obj:`ollin.Site`
        Information of Site at which movement took place.
    movement_data : array
        Array of shape [num_individuals, time_steps, 2] holding coordinate
        information of individual location through movement.
    times : array
        Array of shape [time_steps] with time at which the time steps took
        place. Units are in days.
    home_range : float or None
        Home range value of species. Only necessary for occupancy calculation.
        See :py:class:`ollin.Occupancy`.

    """
    def __init__(self, site, movement_data, times, home_range=None):
        """Construct Movement Data object.

        Arguments
        ---------
        site : :py:obj:`ollin.Site`
            Information of Site at which movement took place.
        movement_data : array
            Array of shape [num_individuals, time_steps, 2] holding coordinate
            information of individual location through movement.
        times : array
            Array of shape [time_steps] with time at which the time steps took
            place. Units are in days.
        home_range : float, optional
            Home range value of species. Only necessary for occupancy
            calculation. See :py:class:`ollin.Occupancy`.

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
        """Plot trajectories from Movement data.

        Movement Data plotting adds the following optional components to the
        plot:

        1. "trajectories":
           If present in include list, some trajectories will be plotted as a
           broken line. Trajectory simplification is possible through the
           simplify keyworded argument. Several trajectories will be plotted.
           Color of line will be chosen at random from some colormap.

        All other components in the include list will be passed down to the
        Site plotting method. See :py:meth:`ollin.Site.plot` for all plot
        components defined at that level.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot movement trajectories.
        figsize : list or tuple, optional
            Size of figure to create if no axes object was given. Defaults to
            (10, 10).
        include : list or tuple, optional
            List of components to add to the plot. Components list will be
            passed to the Site object to add the corresponding components.
        num : int, optional
            Number of trajectories to plot. Defaults to 10.
        steps : int, optional
            Number of time steps to plot in trajectories. Defaults to all.
        mov_cmap : str, optional
            Name of colormap to choose trajectories colors from. See
            :py:mod:`matplotlib.cm` to see all options. Defaults to 'Greens'.
        simplify : int, optional
            Trajectories will be forced to consist of this number of points, so
            if given, some time steps might be skipped.
        kwargs : dict, optional
            All other keyworded arguments will be passed to the Site's plotting
            method.

        Returns
        -------
        ax : :py:obj:`matplotlib.axes.Axes`
            Return axes for further plotting.

        """
        import matplotlib.pyplot as plt  # pylint: disable=import-error
        from cycler import cycler

        if include is None:
            include = [
                'niche',
                'niche_boundary',
                'rectangle',
                'trajectories']

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        self.site.plot(
            include=include, ax=ax, **kwargs)

        if 'trajectories' in include:
            cmap = plt.get_cmap(mov_cmap)
            colors = [cmap(i) for i in np.linspace(.8, .1, 10)]
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

    def analyze(self):
        """Analyze movement and return analysis."""
        return MovementAnalysis(self)


class Movement(MovementData):
    """Class for simulated movement data.

    Extension of :py:class:`MovementData` class. When movement data arises from
    simulation, the applied movement model is also stored within the object.

    Attributes
    ----------
    site : :py:obj:`ollin.Site`
        Information of Site at which movement took place.
    movement_data : array
        Array of shape [num_individuals, time_steps, 2] holding coordinate
        information of individual location through movement.
    times : array
        Array of shape [time_steps] with time at which the time steps took
        place. Units are in days.
    home_range : float or None
        Home range value of species. Only necessary for occupancy calculation.
        See :py:class:`ollin.Occupancy`.
    movement_model : :py:obj:`ollin.MovementModel`
        Movement model used to generate movement.
    velocity : float
        Mean velocity (in Km/Day) used to movement simulation.

    """

    def __init__(
            self,
            site,
            movement_data,
            movement_model,
            velocity,
            home_range=None):
        """Create Movement object for simulated movement.

        Arguments
        ---------
        site : :py:obj:`ollin.Site`
            Site in which movement took place.
        movement_data : array
            Array of shape [num_individuals, time_steps, 2] holding coordinate
            information of all individuals along all simulated time steps.
        movement_model : :py:obj:`ollin.MovementModel`
            Movement model used to generate movement_data.
        velocity : float
            Mean velocity (in Km/Day) used in the simulation.
        home_range : float, optional
            Home range of simulated species. Used mainly for occupancy
            calculation, or home range calibration. See
            :py:class:`ollin.Occupancy`.

        """
        self.movement_model = movement_model
        self.velocity = velocity

        steps = movement_data.shape[1]
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
        """Make simulated movement data.

        Use some movement model from the model library to generate simulated
        movement data for some virtual species with a fixed velocity (in
        Km/Day).

        Number of individuals and mean velocity must be specified, but it is
        also possible to use home range and/or occupancy as proxies for density
        and mean velocity, respectively. The faithfullness of these proxies
        depend on the correct calibration of the parameters asociated to the
        movement model.

        If using a movement model from the library, these should be
        pre-calibrated, and hence home range (or occupancy) can be used to
        estimate mean velocity (or density) with some degree of accuracy.

        Otherwise the user must first be sure that the model is calibrated.
        See :py:mod:`ollin.calibration`.

        Arguments
        ---------
        site : :py:obj:`ollin.Site`
            Site in which simulate movement.
        days : int, optional
            Number of simulation days. Movement models include a steps_per_day
            parameter, so number of simulated time steps is days *
            steps_per_day. Defaults to 365.
        num : int, optional
            Number of individuals to include in simulation. If not given,
            occupancy argument must be provided.
        occupancy : float, optional
            If provided the relationship occupancy <-> density will be used to
            estimate the number of individuals to include in simulation. See
            :py:func:`ollin.core.utils.occupancy_to_density`.
        velocity : float, optional
            Mean velocity in Km/Day to use in movement model. If not given,
            home range argument must be provided.
        home_range : float, optional
            Home range of simulated species. If provided the relationship
            home_range <-> mean velocity will be used to estimate the mean
            velocity of species. See
            :py:func:`ollin.core.utils.home_range_to_velocity`.
        movement_model : str or :py:obj:`ollin.movement_models.MovementModel`
            Name of movement model in library o MovementModel instance to use
            to generate simulated movement.

        Returns
        -------
        mov : :py:obj:`Movement`
            Movement instance with simulated movement data.

        Raises
        ------
        ValueError
            If both num and occupancy, or velocity and home_range, are given
            simultaneously.

        """
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
            area = site.range[0] * site.range[1]
            home_range_proportion = home_range / area
            dens = occupancy_to_density(
                occupancy,
                home_range_proportion,
                site.niche_size,
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
        """Extend movement data with new simulated movement.

        Use last position as starting point to generate new simulated
        movement and append to existing. This method will use the same mean
        velicity and movement model to generate new movement.

        Arguments
        ---------
        days : int
            Number of days of new simulated movement.
        inplace : bool, optional
            If true, only Movement object attributes will be changed, otherwise
            a copy of the object will be made with the new movement data.

        Returns
        -------
        extension : :py:obj:`Movement`
            Movement object with extended movement data.

        """

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


class MovementAnalysis(object):
    """Class for movement analysis.

    Movement analysis refers to the calculation of the following information:

    1. Velocities:
        Information of time at which time steps where taken with distance
        travelled can be used to calculate velocity of every individual at
        each timestep.

    2. Bearings:
        Bearing refers to the angle of direction at some time step. The angle
        is taken in reference to the x-axis. The units are radians and the
        angle value ranges from [-pi, pi]. Bearing values can be calculated for
        every individual at each timestep.

    3. Turn Angles:
        Turn angle refers to angle formed between two adjacent bearing
        directions in the same trajectory. The turn angle is in radians and its
        values ranges from [-pi, pi]. Turn angle values can be calculated for
        every individual at each timestep.

    MovementAnalysis objects hold this information.

    Attributes
    ----------
    movement : :py:obj:`ollin.MovementData`
        Movement data analyzed.
    velocities : array
        Array of shape [num_individuals, time_steps - 1] holding all velocity
        information.::

            velocities[i, j] = v

        Means that the i-th individual had velocity v at the j-th time step.
    bearings : array
        Array of shape [num_individuals, time_steps - 1] holding all bearing
        information.
    turn_angles : array
        Array of shape [num_individuals, time_steps - 2] holding all turn angle
        information.

    """

    def __init__(self, mov):
        """Analyze movement object.

        Arguments
        ---------
        mov : :py:obj:`MovementData`
            Movement data to analyze.
        """

        self.movement = mov
        self.velocities, self.bearings, self.turn_angles = self.analyze(mov)

    def get_mean_velocity(self):
        """Calculate and return mean velocity"""
        return self.velocities.mean()

    @staticmethod
    def analyze(movement):
        """Calculate and return velocities, bearings and turn angles"""
        data = movement.data
        times = movement.times

        dtimes = (times[1:] - times[:-1])[None, :, None]
        directions = (data[:, 1:, :] - data[:, :-1, :]) / dtimes
        complex_directions = directions[:, :, 0] + 1j * directions[:, :, 1]
        velocities = np.abs(complex_directions)
        bearings = np.angle(complex_directions)
        turn_angles = np.angle(complex_directions[:, 1:] /
                               complex_directions[:, :-1])
        return velocities, bearings, turn_angles

    def plot_velocity_distribution(
            self,
            ax=None,
            figsize=(10, 10),
            num_individual=0,
            bins=20,
            width=None,
            cmap='Reds',
            alpha=0.8,
            log=True):
        """Plot distribution of velocities.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot.
        figsize : tuple or list, optional
            Size of figure to create if no axes are provided.
        num_individual : int or tuple or list or array, optional
            Selection of individuals from which to draw velocity
            information. If num_individual='all', all information will
            be plotted.
        bins : int, optional
            Number of bins to use in histogram of velocity distribution.
            Defaults to 20.
        width : float, optional
            Width of bars in histogram. If none is given, width will be
            maximum possible before overlap.
        cmap : str, optional
            Colormap to use to assign colors to histogram bars. Defaults
            to 'Reds'.
        alpha : float, optional
            Alpha value of plot. Defaults to 0.8.
        log : bool, optional
            If true, yaxis in histogram will have logarithmic scale.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if num_individual == 'all':
            vdata = self.velocities.ravel()
        elif isinstance(num_individual, int):
            vdata = self.velocities[num_individual, :]
        else:
            vdata = self.velocities[num_individual, :].ravel()

        range_ = (0, vdata.max())
        histogram, _ = np.histogram(
            vdata, bins=bins, range=range_, normed=not log)
        if log:
            histogram = np.log(histogram + 1)

        if width is None:
            width = (range_[1] - range_[0]) / bins
        theta = np.linspace(range_[0], range_[1], bins, endpoint=False)

        bars = ax.bar(theta, histogram, width=width, bottom=0)
        mins, maxs = histogram.min(), histogram.max()

        # Use custom colors and opacity
        cmap = get_cmap(cmap)
        for value, pbar in zip(histogram, bars):
            pbar.set_facecolor(
                cmap((value - mins + 0.1) / (0.1 + maxs - mins)))
            pbar.set_alpha(alpha)
        ticks = np.linspace(0, histogram.max(), 10)
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.round(ticks, 2))

        ax.set_title('Velocity distribution')
        ax.set_xlabel('Velocity (Km/Day)')

        if log:
            ax.set_ylabel('Log count')
        else:
            ax.set_ylabel('Proportion')

        return ax

    def plot_bearing_distribution(
            self,
            ax=None,
            figsize=(10, 10),
            num_individual=0,
            bins=20,
            width=None,
            cmap='Reds',
            alpha=0.8):
        """Plot distribution of bearing angles.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot.
        figsize : tuple or list, optional
            Size of figure to create if no axes are provided.
        num_individual : int or tuple or list or array, optional
            Selection of individuals from which to draw bearing angle
            information. If num_individual='all', all information will
            be plotted.
        bins : int, optional
            Number of bins to use in histogram of bearing distribution.
            Defaults to 20.
        width : float, optional
            Width of bars in histogram. If none is given, width will be
            maximum possible before overlap.
        cmap : str, optional
            Colormap to use to assign colors to histogram bars. Defaults
            to 'Reds'.
        alpha : float, optional
            Alpha value of plot. Defaults to 0.8.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        if ax is None:
            _, ax = plt.subplots(
                figsize=figsize, subplot_kw={"polar": True})

        if num_individual == 'all':
            bdata = self.bearings.ravel()
        elif isinstance(num_individual, int):
            bdata = self.bearings[num_individual, :]
        else:
            bdata = self.bearings[num_individual, :].ravel()

        range_ = (-np.pi, np.pi)
        histogram, _ = np.histogram(
            bdata, bins=bins, range=range_, normed=True)

        if width is None:
            width = (range_[1] - range_[0]) / bins
        theta = np.linspace(range_[0], range_[1], bins, endpoint=False)

        bars = ax.bar(theta, histogram, width=width, bottom=0.05)
        mins, maxs = histogram.min(), histogram.max()

        # Use custom colors and opacity
        cmap = get_cmap(cmap)
        for value, pbar in zip(histogram, bars):
            pbar.set_facecolor(
                cmap((value - mins + 0.1) / (0.1 + maxs - mins)))
            pbar.set_alpha(alpha)
        ticks = np.linspace(0.05, 0.05 + histogram.max(), 4)
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.round(ticks - 0.05, 2))

        ax.set_title('Bearing distribution')
        return ax

    def plot_turn_angle_distribution(
            self,
            ax=None,
            figsize=(10, 10),
            num_individual=0,
            bins=20,
            width=None,
            cmap='Reds',
            alpha=0.8):
        """Plot distribution of turning angles.

        Arguments
        ---------
        ax : :py:obj:`matplotlib.axes.Axes`, optional
            Axes object in which to plot.
        figsize : tuple or list, optional
            Size of figure to create if no axes are provided.
        num_individual : int or tuple or list or array, optional
            Selection of individuals from which to draw turning angle
            information. If num_individual='all', all information will
            be plotted.
        bins : int, optional
            Number of bins to use in histogram of turn angle distribution.
            Defaults to 20.
        width : float, optional
            Width of bars in histogram. If none is given, width will be
            maximum possible before overlap.
        cmap : str, optional
            Colormap to use to assign colors to histogram bars. Defaults
            to 'Reds'.
        alpha : float, optional
            Alpha value of plot. Defaults to 0.8.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        if ax is None:
            _, ax = plt.subplots(
                figsize=figsize, subplot_kw={"polar": True})

        if num_individual == 'all':
            tdata = self.turn_angles.ravel()
        elif isinstance(num_individual, int):
            tdata = self.turn_angles[num_individual, :]
        else:
            tdata = self.turn_angles[num_individual, :].ravel()

        range_ = (-np.pi, np.pi)
        histogram, _ = np.histogram(
            tdata, bins=bins, range=range_, normed=True)

        if width is None:
            width = (range_[1] - range_[0]) / bins
        theta = np.linspace(range_[0], range_[1], bins, endpoint=False)

        bars = ax.bar(theta, histogram, width=width, bottom=0.05)
        mins, maxs = histogram.min(), histogram.max()

        # Use custom colors and opacity
        cmap = get_cmap(cmap)
        for value, pbar in zip(histogram, bars):
            pbar.set_facecolor(
                cmap((value - mins + 0.1) / (0.1 + maxs - mins)))
            pbar.set_alpha(alpha)
        ticks = np.linspace(0.05, 0.05 + histogram.max(), 4)
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.round(ticks - 0.05, 2))

        ax.set_title('Turn Angle distribution')
        return ax
