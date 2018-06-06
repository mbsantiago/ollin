from __future__ import print_function

import pandas as pd  # pylint: disable=import-error
import numpy as np


PROPERTIES = [
    'Species',
    'Common Name',
    'Nombre Comun',
    'Home Range',
    'Occupancy',
    'Detection Probability',
    'Season',
    'Num Cameras',
    'Area',
    'References']


class EmpiricalData(object):
    def __init__(self, data):
        for key, value in data.iteritems():
            setattr(self, key, value)


def load_species_data():
    df = pd.read_csv('all_species_data.csv')

    all_data = []
    for index, row in df.iterrows():
        try:
            data = parse_data(row)
            obj = EmpiricalData(data)
            all_data.append(obj)
        except ValueError:
            pass

    return all_data


def parse_data(row):
    if np.isnan(row['Home Range']) or np.isnan(row['Area']):
        raise ValueError

    data = {
        prop.replace(' ', '_').lower(): row[prop]
        for prop in PROPERTIES
    }
    return data


DATA = load_species_data()
