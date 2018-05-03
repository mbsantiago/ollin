import os
import numpy as np
import time
from contextlib import contextmanager
from multiprocessing import Pool

from simulation import load_species_data
from make_species_movement_data import RANGE


CONE_RANGE = .05
CONE_ANGLE = 60
CURRENT_MOVEMENT_DATA = None

@contextmanager
def use_movement_data(mov):
    global CURRENT_MOVEMENT_DATA
    CURRENT_MOVEMENT_DATA = mov
    yield
    CURRENT_MOVEMENT_DATA = None


def make_random_cameras(num):
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

    return selection
    

def load_species_movement_data(species):
    path = os.path.join('species_movement', species + '.npy')
    mv_data = np.load(path)
    return mv_data


def make_detection_data(species_movement, camera_position, camera_direction):

    
    relative_pos = species_movement[:, :, :] - camera_position[None, None, :]
    norm = np.sqrt(np.sum(relative_pos**2, -1))
    closeness = np.less(norm, CONE_RANGE)

    direction = np.divide(relative_pos, norm[...,None], where=closeness[...,None])
    angles = np.arccos(np.sum(camera_direction * direction, -1) , where=closeness)

    angle = np.pi * (CONE_ANGLE / 360.0)
    is_in_angle = np.less(angles, angle, where=closeness)

    detected = closeness * is_in_angle
    return detected


def make_random_directions(num):
    angles = np.random.uniform(0, 2*np.pi, size=[num])
    directions = np.stack([np.cos(angles), np.sin(angles)], -1)
    return directions


def aux(args):
    cam_pos, cam_dir = args
    return make_detection_data(CURRENT_MOVEMENT_DATA, cam_pos, cam_dir)


def test_camera_arangement(mov_data, cam_pos, num_animals):
    num_cams = len(cam_pos)
    cam_direction = make_random_directions(num_cams)

    max_animals = len(mov_data)
    random_selection = np.random.choice(range(max_animals), size=num_animals, replace=False)
    mov_data = mov_data[random_selection]
    
    with use_movement_data(mov_data):
        p = Pool()
        args = zip(cam_pos, cam_direction)
        detection = p.imap(aux, args)
        p.close()
        p.join()

    detection = np.amax(np.stack(detection), 0)
    return detection


def calculate_ocupation_grid(mov_data, grid_num):
    size = RANGE / grid_num
    data = np.floor_divide(mov_data, size).astype(np.int).reshape([-1, 2])
    grid = np.zeros([grid_num, grid_num])
    for d in data:
        grid[d[0], d[1]] = 1
    return grid, data
   

def aggregate_data(data):
    first_detection = np.argmax(data, 1)
    agg = np.zeros(data.shape[1])
    for index in first_detection:
        if index != 0:
            agg[index:] += 1

    return agg

def main():
    species_data = load_species_data()
    mov_data = load_species_movement_data(species_data[0][0])

    timer = time.time()
    cameras = make_random_cameras(100)
    det = test_camera_arangement(mov_data, cameras, 1000)
    agg = aggregate_data(det)
    timer = time.time() - timer
    print(timer)
    print(agg)

if __name__ == '__main__':
    main()
