from __future__ import print_function

import csv
from collections import OrderedDict


def list_all_species():
    species = SPECIES.keys()
    for n, (s1, s2, s3) in enumerate(
            zip(species[::3], species[1::3], species[2::3])):
        print('{:>2}) {:<30}{:>2}) {:<30}{:>2}) {:<}'.format(
            3*n+1, s1, 3*n+2, s2, 3*n+3, s3))


def list_species_data():
    header = u'\t|{:^4}||{:^30}|{:^30}|{:^30}|'
    sep = '\t' + '-'*100
    print(sep)
    print(header.format(
        'N', 'Species', u'Home Range(Km2)', 'Population Density(1/Km2)'))
    print(sep)
    for num, (specie, info) in enumerate(SPECIES.iteritems()):
        print(header.format(
            num+1, specie, info['home_range'], info['density']))
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
            'home_range': float(row[1]),
            'density': float(row[2])
        }
    return parsed


SPECIES = parse_data(load_species_data())

ARTICLE_SPECIES = OrderedDict([
    ("Spilogale putorius", {'occupancy': 0.245, 'detectability': 0.023}),
    ("Cervus canadensis", {'occupancy': 0.585, 'detectability': 0.024}),
    ("Puma concolor", {'occupancy': 0.600, 'detectability': 0.030}),
    ("Canis latrans", {'occupancy': 0.861, 'det}ctability': 0.031}),
    ("Lynx rufus", {'occupancy': 0.970, 'detectability': 0.040}),
    ("Urocyon cinereoargenteus", {'occupancy': 0.400, 'detectability': 0.063}),
    ("Ursus americanus", {'occupancy': 0.504, 'detectability': 0.072}),
    ("Odocoileus hemionus", {'occupancy': 0.925, 'detectability': 0.141}),
    ("Sylvilagus nuttallii", {'occupancy': 0.925, 'detectability': 0.190})
    ])


COMMON_SPECIES = OrderedDict([
    (species, SPECIES[species].copy())
    for species in ARTICLE_SPECIES.keys() if species in SPECIES.keys()
 ])

for species, value in COMMON_SPECIES.iteritems():
    value.update(ARTICLE_SPECIES[species])
