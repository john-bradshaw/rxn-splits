
# Turn off RDKit logging to ignore the canonicalization errors (https://github.com/rdkit/rdkit/issues/2683)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse
import collections
import pickle
from dataclasses import (
    dataclass,
    field)
import time
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm

from rxn_splits import pistachio
from rxn_splits import chem_utils
from rxn_splits.settings import UNKNOWN_STR, dataset_paths
from rxn_splits import utils

MAX_TOKENS = 800


@dataclass
class Params:
    op_file_stem: str  # name for op file (date will get added)
    failure_op_file_stem:  Optional[str] = None
    folders: List[str] = field(default_factory=lambda: ["grants"])
    # ^ which subfolders to explore, : ['grants', 'ep', 'epo', 'applications', 'wo']
    store_extra_reaction_info: bool = True

    def __post_init__(self):
        # Create the file names from the stems
        self.date_ = datetime.now().strftime("%Y%m%d_%H%M_")
        op_file_ = self.date_ + self.op_file_stem
        error_op_file_ = self.date_ + self.failure_op_file_stem if self.failure_op_file_stem else op_file_
        output_folder = Path("outputs")
        self.failure_op_file = (output_folder / error_op_file_).with_suffix(".errors.pick")
        self.op_file = (output_folder / op_file_).with_suffix(".pick")
        self.log_name = (output_folder / op_file_).with_suffix(".log")

        # Create a logger
        self.logger = utils.get_logger(internal_log_name="clean_data", log_file_name=self.log_name)

        # Write out to logger!
        self.logger.info(f"OP file: {self.op_file}, Errors file: {self.failure_op_file}, Folders: {self.folders}")


def _convert_default_dict_counters_to_ordinary_dicts_of_frozen_sets(dd_counter):
    dict_of_counter = {k: frozenset(v.items()) for k, v in dd_counter.items()}
    return dict_of_counter


class FailedCheck(RuntimeError):
    pass


def _num_tokens(smiles_iter):
    str_ = '.'.join(list(smiles_iter))
    return len(chem_utils.smi_tokenizer(str_).split())


def _frozen_set_to_lst(frozen_set_in):
    out = []
    for smi, num in frozen_set_in:
        out.extend([smi]*num)
    return out


def main(params: Params):
    """
    Reads in the Pistachio data and cleans up the reactions and deduplicates, putting the cleaned reactions into a
    pickle.
    """
    stime = time.time()
    print(f"Looking in folders: {params.folders}")
    json_files_iterator = pistachio.get_pistachio_reader(params.folders)

    out_reactions = {}  # ← map from cleaned reaction to extra details

    molecule_checks = collections.OrderedDict([
        ("at_least_5_heavy_atoms_in_reactants", lambda reactant_smi, prod_smi: chem_utils.num_heavy_atoms(_frozen_set_to_lst(reactant_smi)) >= 5),
        ("at_least_one_carbon_in_reactants", lambda reactant_smi, prod_smi: chem_utils.at_least_one_carbon(_frozen_set_to_lst(reactant_smi))),
        (f"under_{MAX_TOKENS}_tokens", lambda reactant_smi, prod_smi: all([el < MAX_TOKENS for
                                                             el in (_num_tokens(_frozen_set_to_lst(reactant_smi)),
                                                                    _num_tokens(_frozen_set_to_lst(prod_smi)))])),
        ("at_least_two_bonds_in_one_molecule", lambda reactant_smi, prod_smi: chem_utils.max_number_of_bonds_per_mol(_frozen_set_to_lst(reactant_smi)) >= 2),
        ("at_least_one_larger_product_diff", lambda reaction_fs, product_fs: \
            chem_utils.at_least_one_larger_product_different(reaction_fs, product_fs, canonicalize=False, num_heavy_atom_required=2)),
            # ^ should already be in canonical form so unncessary to canonicalize again.
        ("not_deprotonation", lambda reaction_fs, product_fs: chem_utils.not_deprotonation(reaction_fs, product_fs))
    ])
    # ^ put cheaper checks first as will go through in order and stop early.
    check_list = '\n'.join(molecule_checks.keys())
    params.logger.info(f"Checks are:\n{check_list}")

    # ↓ will also use these supp. variables to store extra data about the cleaning procedure
    n_canon_fails = 0
    nignores_due_to_duplication = 0
    out_reactions_to_namerxn = collections.defaultdict(collections.Counter)
    out_reactions_to_year = collections.defaultdict(collections.Counter)
    out_reactions_to_uncleaned = collections.defaultdict(collections.Counter)
    nfails_due_to_molecule_checks = {k: 0 for k in molecule_checks}
    failures_canon = set()
    failures_checks = collections.defaultdict(collections.Counter)
    # ^ each failed reaction SMILES to a counter indicating the "first" reason it failed each time.

    # Now loop through each json.
    for i, json_in in enumerate(tqdm(json_files_iterator)):
        # 1. Get data!
        data_dict = pistachio.get_pistachio_data(json_in, params.store_extra_reaction_info)

        # 2. Try and clean it up (canonicalize and put in set)
        try:
            cleaned_reaction = chem_utils.clean_smiles_reaction(data_dict['smiles'])
        except Exception:
            n_canon_fails += 1
            failures_canon.add(data_dict['smiles'])
            continue   # ← skip rest...

        # 3. Check the other conditions for skipping
        try:
            for skip_name, condition in molecule_checks.items():
                if not condition(*cleaned_reaction):
                    # did not meet condition so record failure and skip
                    nfails_due_to_molecule_checks[skip_name] += 1
                    failures_checks[data_dict['smiles']].update([skip_name])
                    raise FailedCheck
        except FailedCheck:
            continue

        # 4. Record details into the supplementary variables
        # if it has SMILES add the namerxn and year to the output dictionary
        out_reactions_to_year[cleaned_reaction].update([data_dict['year']])
        out_reactions_to_namerxn[cleaned_reaction].update([data_dict['name_rxn']])
        out_reactions_to_uncleaned[cleaned_reaction].update([data_dict['smiles']])

        # 5. Work out if it should displace an existing reaction in our dict (if such a one exists)
        displace_flag = False  # <- by default we will keep the old reaction

        # We then consider if this reaction is already in the dataset
        if cleaned_reaction in out_reactions:
            # we record the number of reactions skipped due to duplication.
            nignores_due_to_duplication += 1

            # we then check if namerxn was unknown before and now is
            # (in which case we displace the previous reaction)
            namerxn = data_dict['name_rxn']
            if namerxn != UNKNOWN_STR and out_reactions[cleaned_reaction]['name_rxn'] == UNKNOWN_STR:
                displace_flag = True
            else:
                # if name rxn was known before but date is earlier then displace
                year = data_dict['year']
                previous_record_year = out_reactions[cleaned_reaction]['year']
                if previous_record_year == UNKNOWN_STR or \
                        (isinstance(year, int) and year < out_reactions[cleaned_reaction]['year']):
                    displace_flag = True

        else:
            # if not in the dataset then we will automatically "displace"
            displace_flag = True

        # 6. Assuming it should displace a reaction add it to our dict.
        if displace_flag:
            # add it straight away
            out_reactions[cleaned_reaction] = data_dict

    # put in a different data format, more applicable to pickling
    out_reactions_to_namerxn = _convert_default_dict_counters_to_ordinary_dicts_of_frozen_sets(out_reactions_to_namerxn)
    out_reactions_to_year = _convert_default_dict_counters_to_ordinary_dicts_of_frozen_sets(out_reactions_to_year)
    out_reactions_to_uncleaned = _convert_default_dict_counters_to_ordinary_dicts_of_frozen_sets(out_reactions_to_uncleaned)

    with open(params.op_file, 'wb') as fo:
        pickle.dump(dict(
            reactions=out_reactions,
            meta=dict(
                out_reactions_to_namerxn=out_reactions_to_namerxn,
                out_reactions_to_year=out_reactions_to_year,
                out_reactions_to_uncleaned=out_reactions_to_uncleaned,
                n_canon_fails=n_canon_fails,
                nignores_due_to_duplication=nignores_due_to_duplication,
                failures=failures_canon,
                nfailures_for_checks=nfails_due_to_molecule_checks,
                folders=params.folders,
                datestr=params.date_,
                data_root=dataset_paths['pistachio']
            )
        ), fo)
    with open(params.failure_op_file, "wb") as fo:
        pickle.dump(
            failures_checks, fo
        )
    params.logger.info(f"Number of canonicalization failures: {n_canon_fails}")
    params.logger.info(f"Number of ignores due to duplication: {nignores_due_to_duplication}")
    for name, num_fails in nfails_due_to_molecule_checks.items():
        params.logger.info(f"Number of ignores due to {name} check: {num_fails}")

    etime = time.time()
    params.logger.info(f"{len(out_reactions)} reactions dumped out to {params.op_file}")
    params.logger.info(f"Time taken was {etime - stime:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("op_file_stem")
    args = parser.parse_args()
    main(Params(args.op_file_stem))
    print('Done!')
