import collections
import datetime
import hashlib
import itertools
import json
import logging
import socket
import subprocess
import sys

import numpy as np

from . import chem_utils
from . import settings


class NumpyFloatValuesEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle numpy float32 values.
    https://stackoverflow.com/questions/64154850/convert-dictionary-to-a-json-in-python
    """
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_logger(*, internal_log_name=None, log_file_name="run_log.log", capture_warnings=True):

    internal_log_name = internal_log_name or __name__
    logger = logging.getLogger(internal_log_name)

    # std out handler
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(f"{internal_log_name} - %(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File Handler
    fh = logging.FileHandler(log_file_name)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.setLevel(logging.DEBUG)
    logger.debug(f"Internal log name: {internal_log_name}")
    logger.debug(f"Log file name: {log_file_name}")

    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        se = logging.StreamHandler(sys.stderr)
        warnings_logger.addHandler(se)
        warnings_logger.addHandler(fh)
        logger.debug(f"Warnings logger also directed to file handler.")

    return logger


def cast(input, cast_func, pass_through_unknowns=True):
    """
    Tries to cast `input` according to `cast_func` but allows the returning of `settings.UNKNOWN_STR` if `cast_func`
    throws an exception and `pass_through_unknowns` is True.
    """
    try:
        out = cast_func(input)
    except Exception as ex:
        if pass_through_unknowns and input == settings.UNKNOWN_STR:
            out = settings.UNKNOWN_STR
        else:
            raise ex
    return out


def hash_file(file_path):
    """
    Returns the sha256 hash of a file at `file_path`.
    """
    BUFFER_SIZE = int(64*1024)

    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUFFER_SIZE)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def convert_canon_reactions_to_smiles(reactions_as_frozensets, rng: np.random.RandomState):
    """
    Converts a reaction defined as a tuple of frozenset of reactant-product molecule counts and converts it to a reaction
    SMILES string.

    e.g., would take in as input something that looks like:
        (frozenset({('C1COCCO1', 1),
                  ('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1),
                  ('Cl', 2)}),
         frozenset({('C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O', 1), ('Cl', 2)}))
    and return something that looks like:
        C1COCCO1.C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl.Cl.Cl>>C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O.Cl.Cl
        (or permutation thereof)
    """
    reactants = list(itertools.chain(*[[el] * i for el, i in reactions_as_frozensets[0]]))
    products = list(itertools.chain(*[[el] * i for el, i in reactions_as_frozensets[1]]))

    rng.shuffle(reactants)
    rng.shuffle(products)

    out = '.'.join(reactants) + '>>' + '.'.join(products)
    return out


def convert_canon_reactions_to_jsonl(reactions_as_frozensets, rng: np.random.RandomState):
    """
    Converts a reaction defined as a tuple of frozenset of reactant-product molecule counts and converts it into a jsonl
    SMILES representation suitable for Hugging Face.

    e.g., would take in as input something that looks like:
        (frozenset({('C1COCCO1', 1),
                  ('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1),
                  ('Cl', 2)}),
         frozenset({('C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O', 1), ('Cl', 2)}))
    and return something that looks like:
    {"translation": {"reactants": "C1COCCO1.C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl.Cl.Cl",
     "products": "C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O.Cl.Cl"}}

    """
    #todo: merge with above func
    reactants = list(itertools.chain(*[[el] * i for el, i in reactions_as_frozensets[0]]))
    products = list(itertools.chain(*[[el] * i for el, i in reactions_as_frozensets[1]]))

    rng.shuffle(reactants)
    rng.shuffle(products)

    out = {
        "translation":
            {
                "reactants": '.'.join(reactants),
                "products": '.'.join(products)
            }
    }
    return out


def convert_jsonl_to_canon_reactions(jsonl, canonicalize_flag=False):
    """
    Converts a jsonl SMILES representation suitable for Hugging Face into a reaction defined as a tuple of frozenset of
    reactant-product molecule counts.

    e.g., would take in as input something that looks like:
    {"translation": {"reactants": "C1COCCO1.C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl.Cl.Cl",
     "products": "C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O.Cl.Cl"}}
    and return something that looks like:
        (frozenset({('C1COCCO1', 1),
                  ('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1),
                  ('Cl', 2)}),
         frozenset({('C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O', 1), ('Cl', 2)}))
    """
    if canonicalize_flag:
        map_op = lambda x: map(chem_utils.canonicalize, x)
    else:
        map_op = lambda x: x

    reactants = map_op(jsonl['translation']['reactants'].split('.'))
    products = map_op(jsonl['translation']['products'].split('.'))

    reactants = frozenset(collections.Counter(reactants).items())
    products = frozenset(collections.Counter(products).items())

    return reactants, products


def permute_list_using_rng(input_list, rng: np.random.RandomState):
    """
    Permutes a list using a random number generator. Returns a copy rather than shuffling in place.
    """
    assert isinstance(input_list, list), f"Expected a list but got {type(input_list)}."
    perm = [int(i) for i in rng.permutation(len(input_list))]
    out = [input_list[i] for i in perm]
    return out, perm


def get_watermark():
    """
    Provides a dictionary with details about machine, git status etc (useful for reproducibility).
    """
    try:
        gitlabel = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
    except subprocess.CalledProcessError:
        gitlabel = "git_version_unknown"

    return dict(
        hostname=socket.gethostname(),
        datetime=datetime.datetime.now().date().strftime('%d-%b-%Y_%H-%M-%S'),
        git=gitlabel
    )

