"""
Code for iterating over the Pistachio dataset and parsing individual entries.

Some assumptions:
- all Pistachio reaction files are jsonlines files (even if they end with just the ".json" extension).
- all Pistachio reaction files have "reactions_wbib" in the name.
- the reaction jsons have consistent formats (see `get_pistachio_data` method below).
"""

import json
import os
import typing
from os import path as osp

from . import settings
from . import utils

REACTION_SUBSTRING = 'reactions_wbib'


def get_pistachio_reader(subfolders: typing.Union[str, typing.Iterable[str]] = 'applications'):
    """
    Creates generator of Pistachio json files.
    :param subfolders: subfolder or list of subfolders to walk through.
    """
    if isinstance(subfolders, str): subfolders= [subfolders]
    for subfolder in subfolders:
        root_folder = osp.join(settings.dataset_paths['pistachio'], 'data/extract', subfolder)
        yield from walk_get_all_jsons(root_folder)


def walk_get_all_jsons(top, verbose=True):
    """
    Walk through all the files/folders below `top` and provide a generator that provides each reaction one by one. Note
    that this function identifies files that contain reactions by checking whether `REACTION_SUBSTRING` is in the filename.
    Otherwise it will skip them.

    if `verbose` is `True` then print out `top` and all non-hidden files that are skipped.
    """
    if verbose: print(f"starting walk in {top}")
    for lvl in os.walk(top):
        (dirpath, dirnames, filenames) = lvl

        # Go through and get reactions from filenames
        for fn in filenames:
            if REACTION_SUBSTRING in fn:
                with open(osp.join(dirpath, fn), 'r') as fo:
                    for line in fo.readlines():
                        reaction = json.loads(line)
                        yield reaction
            elif fn[0] == '.':
                continue  # we'll ignore hidden files.
            else:
                if verbose: print(f"Skipping {fn}.")  # while we'll skip other files too, we will make a note of doing so


def get_core_patent_title(str_in):
    """
    Takes the main part of the patent name.
    """
    return str_in.split('_')[0]


def get_pistachio_data(json_in, extras=False):
    """
    Parse a Pistachio reaction json to extract the title, namerxn, year, and reaction SMILES.

    While not necessarily perfect at extracting all information, have checked against ground truth on a few examples
    to check it works well enough.
    """

    out_dict = {}

    # Title
    try:
        title = json_in['title']
    except KeyError:
        try:
            title = json_in["data"]['documentId']
        except:
            title = settings.UNKNOWN_STR
    out_dict['title'] = title

    # Name rxn
    try:
        name_rxn = json_in['data']['namerxn']
    except KeyError:
        name_rxn = settings.UNKNOWN_STR
    out_dict['name_rxn'] = name_rxn

    # Name rxn def
    try:
        name_rxn_def = json_in['data']['namerxndef']
    except KeyError:
        name_rxn_def = settings.UNKNOWN_STR
    out_dict['name_rxn_def'] = name_rxn_def

    # Year
    try:
        ymd = json_in['data']['date']
    except KeyError:
        ymd = ''
    if len(ymd) >= 4:
        y = ymd[:4]
    else:
        y = settings.UNKNOWN_STR
    if len(ymd) >= 6:
        m = ymd[4:6]
    else:
        m = settings.UNKNOWN_STR
    if len(ymd) > 6:
        d = ymd[6:]
    else:
        d = settings.UNKNOWN_STR
    out_dict['year'] = utils.cast(y, int)
    out_dict['month'] = utils.cast(m, int)
    out_dict['day'] = utils.cast(d, int)

    # Reaction SMILES
    try:
        smiles = json_in['data']['smiles']
    except KeyError:
        smiles = settings.UNKNOWN_STR
        # todo: could maybe try creating it ourselves from the components section of the json instead -- although do
        # not know whether it would ever be there and not present in this location.
    out_dict['smiles'] = smiles

    if extras:
        # assignees
        try:
            org = json_in['data']['assignees']
        except KeyError:
            org = settings.UNKNOWN_STR
        out_dict['assignees'] = org

        # authors
        try:
            auths = json_in['data']['authors']
        except KeyError:
            auths = settings.UNKNOWN_STR
        out_dict['authors'] = auths

    return out_dict
