"""
Collects all the reactions corresponding to a given NameRxn code and puts them in a test set.

This is separate to the NameRxn splits, and may be useful for instance in the time splits where you may want to test
an older set of years on a reaction class that is yet to be discovered.

Note this code does not canonicalize on the individual SMILES molecule level.
"""

import argparse
import json
import os
import pathlib
import pickle

import numpy as np
import tabulate

from rxn_splits import utils
from rxn_splits import split_utils

OP_FOLDER = pathlib.Path(__file__).parent / pathlib.Path("outputs")


def main():
    argparser = argparse.ArgumentParser(description="Obtain the reactions corresponding to NameRxn codes.")
    argparser.add_argument('--split_def_path', type=str, required=True)
    args = argparser.parse_args()

    # Load in split def
    with open(args.split_def_path, 'r') as fo:
        split_args = json.load(fo)
    rng = np.random.RandomState(split_args["random_seed"])

    # Set up log
    output_folder_ = OP_FOLDER / split_args["name"]
    os.makedirs(output_folder_, exist_ok=False)
    log_name = (output_folder_ / split_args["name"]).with_suffix(".log")
    log = utils.get_logger(log_file_name=log_name)
    log.info(f"Arguments: {split_args}")

    # Load in all the clean reactions
    clean_data_location = split_args["clean_data_location"]
    with open(clean_data_location, 'rb') as fo:
        clean_data = pickle.load(fo)

    # Get the reactions corresponding to the NameRXN code
    in_set_func = split_utils.create_is_of_set_function(split_args["namerxn_codes"])
    namerxn_def_checker = split_utils.create_name_rxn_def_check(split_args["terms_to_check"])
    collected_reactions = {}
    for k, react_dict in clean_data["reactions"].items():
        if in_set_func(react_dict["name_rxn"]):
            collected_reactions[k] = react_dict
        else:
            namerxn_def_checker(react_dict["name_rxn_def"], react_dict["name_rxn"])

    # Write out the non-filtered set if applicable
    log.info(f"Collected {len(collected_reactions)} reactions")
    if split_args.get("create_exclusion_ignored_set", True):
        op_path = output_folder_ / (split_args["name"] + "-all.jsonl")
        split_utils.write_jsonl_reactions(op_path, collected_reactions.keys(), f"{split_args['name']} all", rng)
        log.info(f"Written out all reactions to {op_path}")

    # Now go through and created the filtered set. This first means reading in all of the listed excluded datasets
    out_details = []  # <- will keep details of reaction overlap for printing at end
    excluded_reactions = set()
    for excluded_filename in split_args.get("reactions_jsonl_datasets_to_exclude", []):
        log.info(f"considering reactions in {excluded_filename} for exclusion")
        with open(excluded_filename, 'r') as fo:
            reaction_data = [json.loads(line) for line in fo.readlines()]
        reaction_data = set(utils.convert_jsonl_to_canon_reactions(el) for el in reaction_data)

        # will check the reaction against all clean data to make sure was at least in original set.
        for el in reaction_data:
            if el not in clean_data["reactions"]:
                raise RuntimeError("Looks like SMILES augmentation was being used, which this script cannot currently handle.")

        excluded_reactions.update(reaction_data)

        out_details.append([excluded_filename, len(reaction_data & set(collected_reactions.keys()))])

    # Before we do the exclusion we will write out some details of how many reactions overlap.
    log.info(tabulate.tabulate(out_details, headers=["Excluded dataset", "Number of overlapping reactions"]))
    left_reactions = set(collected_reactions.keys()) - excluded_reactions
    log.info(f"In total we have excluded {len(collected_reactions) - len(left_reactions)} reactions")
    log.info(f"There are{ len(left_reactions)} left after filtering.")

    # Write out the filtered set
    op_path = output_folder_ / (split_args["name"] + "-filtered.jsonl")
    split_utils.write_jsonl_reactions(op_path, left_reactions, f"{split_args['name']} filtered", rng)
    log.info(f"Written out filtered reactions to {op_path}")

    # Write out all the collected reactions (in case we want their details later
    op_path = output_folder_ / (split_args["name"] + "-collected_reactions.pick")
    with open(op_path, 'wb') as fo:
        pickle.dump(collected_reactions, fo)
    log.info(f"Written out collected reactions to {op_path}")

    log.info("Done!")


if __name__ == "__main__":
    main()
