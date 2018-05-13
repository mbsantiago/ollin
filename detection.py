import os
import numpy as np
import time

from constants import *


def make_random_camera_positions(num):
    random_points = np.random.uniform(RANGE/10.0, size=[10, 10, 10, 2])
    shift = np.linspace(0, RANGE, 10, endpoint=False)
    shifts = np.stack(np.meshgrid(shift, shift), -1)
    points = random_points + shifts[:, :, None, :]

    points = points.reshape([-1, 2])
    np.random.shuffle(points)

    selection = [points[0]]
    for i in xrange(num - 1):
        selected = False
        for point in points[1:]:
            is_far = True
            for other_point in selection:
                distance = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                if distance <= 1:
                    is_far = False
                    break
            if is_far:
                selected = True
                selection.append(point)
                break
        if not selected:
            raise Exception('No funciono')

    return np.array(selection)


def make_random_directions(num):
    angles = np.random.uniform(0, 2*np.pi, size=[num])
    directions = np.stack([np.cos(angles), np.sin(angles)], -1)
    return directions


def make_random_camera_conf(num):
    pos = make_random_camera_positions(num)
    dirs = make_random_directions(num)
    configuration = {
        'positions': pos,
        'directions': dirs
    }
    return configuration
   

def make_detection_data(movement_data, camera_config):
    camera_position = camera_config['positions']
    camera_direction = camera_config['directions']
    species_movement = movement_data['data']

    relative_pos = species_movement[:, :, None, :] - camera_position[None, None, :, :]
    norm = np.sqrt(np.sum(relative_pos**2, -1))
    closeness = np.less(norm, CONE_RANGE)

    direction = np.divide(relative_pos, norm[..., None], where=closeness[..., None])
    angles = np.arccos(np.sum(camera_direction * direction, -1) , where=closeness)

    angle = np.pi * (CONE_ANGLE / 360.0)
    is_in_angle = np.less(angles, angle, where=closeness)

    detected = closeness * is_in_angle
    return detected


