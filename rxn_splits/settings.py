
import configparser
from os import path as osp

dataset_paths = {}

UNKNOWN_STR = 'unk'


def populate_dataset_paths():
    global dataset_paths

    config = configparser.ConfigParser()
    config.read(osp.join(osp.dirname(__file__), '../config.ini'))

    dataset_paths['pistachio'] = config['Paths']['pistachio']


populate_dataset_paths()  # <-- call the above method on module load to populate the datasets path variable.
