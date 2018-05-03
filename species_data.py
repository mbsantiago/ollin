from __future__ import print_function

import csv
import argparse
from collections import OrderedDict


def list_all_species():
    species = SPECIES.keys()
    for n, (s1, s2, s3) in enumerate(zip(species[::3], species[1::3], species[2::3])):
        print('{:>2}) {:<30}{:>2}) {:<30}{:>2}) {:<}'.format(3*n+1, s1, 3*n+2, s2, 3*n+3, s3))


def list_species_data():
    header = u'\t|{:^4}||{:^30}|{:^30}|{:^30}|'
    sep = '\t' + '-'*100
    print(sep)
    print(header.format('N', 'Especie', u'Ambito Hogareno (Km2)', 'Densidad Poblacional (/Km2)'))
    print(sep)
    for num, (specie, info) in enumerate(SPECIES.iteritems()):
        print(header.format(num+1, specie, info['ambito'], info['densidad']))
    print(sep)


def load_species_data():
    data = []
    with open('species_data.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data


def parse_data(data):
    parsed = OrderedDict()
    for row in data:
        parsed[row[0]] = {
            'ambito': float(row[1]),
            'densidad': float(row[2])
        }
    return parsed


def parse_arguments():
    p = argparse.ArgumentParser('Obtener la informacion de las especies.')
    p.add_argument('--data', action='store_true', help='Enlistar toda la informacion de las especies')
    return p.parse_args()

SPECIES = parse_data(load_species_data())

def main():
    flags = parse_arguments()
    if flags.data:
        list_species_data()
    else:
        list_all_species()


if __name__ == '__main__':
    main()
