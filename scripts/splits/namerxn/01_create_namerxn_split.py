"""
Splits based on NameRxns.

Note:
    * all reactions with namerxn undefined or 0 are discarded.
    * set an upper bound for OOD test set, otherwise use all we can.
    * train, valid, id_test, finetune, od_test are all now document splitted too.
"""
import argparse
import collections
import itertools
import json
import logging
import pathlib
import warnings

import numpy as np
from tqdm import tqdm

from rxn_splits import pistachio
from rxn_splits import settings
from rxn_splits import split_utils
from rxn_splits import utils

OP_FOLDER = pathlib.Path(__file__).parent / pathlib.Path("outputs")
LEEWAY = 200
# ^ set leeway as will be hard to get exact split sizes (particularly OD test where we want as many as possible).


def _is_same_document_as_existing(documents_used_before: set, reactions: dict, react_repr):
    """
    Returns whether the reaction is from a document that we have already used in a previous split.
    """
    patent_title = reactions[react_repr]["title"]
    if patent_title == "" or patent_title == settings.UNKNOWN_STR:
        return True  # we don't know what document it is from, so we will defensively suggest it may match existing docs
    core_patent_title = pistachio.get_core_patent_title(patent_title)
    return core_patent_title in documents_used_before


def create_split_func(namerxns_to_exclude, name_rxn_defs_to_check, exclude_all_relevant_docs=True):
    """
    :param exclude_all_relevant_docs: whether (when making the ID sets) to exclude all docs that had OOD reactions in
    them even if they were ultimately not used.
    """
    exclusion_func = split_utils.create_is_of_set_function(namerxns_to_exclude)
    namerxn_def_warning_func = split_utils.create_name_rxn_def_check(name_rxn_defs_to_check)

    def split_on_name_rxns(shuffled_reactions, reactions, params: split_utils.SplitArgs, log: logging.Logger):
        rng = np.random.RandomState(params.random_seed)

        # 1. == Remove unknown reaction
        shuffled_all_namerxns = [el for el in shuffled_reactions if (reactions[el]["name_rxn"] != settings.UNKNOWN_STR
                                                                 and reactions[el]["name_rxn"] != '0.0')]
        num_removed_due_to_namerxns = len(shuffled_reactions) - len(shuffled_all_namerxns)
        log.info(f"{num_removed_due_to_namerxns} removed for missing or 0.0 namerxn")

        # 2. == Out of distribution
        # Get the reactions of interest:
        od_test_reactions = [el for el in shuffled_all_namerxns if exclusion_func(reactions[el]["name_rxn"])]
        od_test_name_rxns_ = dict(collections.Counter([reactions[el]["name_rxn"] for el in od_test_reactions]).items())
        log.info(f"Test set set size is {len(od_test_reactions)}")
        log.info(od_test_name_rxns_)

        if len(od_test_reactions) == 0:
            log.info("No OD reactions so skipping creation of sets")
            created_datasets_od = {"ft_reactions": [], "od_test_reactions": []}
            split_meta_od_sets = {"info": "OOD set was zero so not created"}
            documents_used_in_od_set = []
        else:
            # We will work out how many reactions to have in the OD test set. We have a target number, but given that
            # we are doing a very specific split, we may have many less than this, we will not be too concerned
            # about being under -- but will make a note in the log.
            od_test_available_reactions = len(od_test_reactions) - params.finetune_amount
            od_test_amount = min(params.od_test_amount, od_test_available_reactions)
            diff_between_od_test_and_available = od_test_available_reactions - od_test_amount
            if diff_between_od_test_and_available > 0:
                log.info(f"{diff_between_od_test_and_available} reactions not used in OOD test due to set sizes")
            else:
                log.info(f"OOD set will be at least <{-diff_between_od_test_and_available} less than requested due to "
                         f"amount available")

            od_dataset_sizes = [
                [("ft_reactions", params.finetune_amount)],
                [("od_test_reactions", od_test_amount)]
            ]
            (created_datasets_od,
                all_od_docs, _, split_meta_od_sets) = split_utils.document_splitter_helper(od_test_reactions,
                                                                        od_dataset_sizes, rng, reactions, log, leeway=LEEWAY)
            if exclude_all_relevant_docs:
                documents_used_in_od_set = all_od_docs
            else:
                documents_used_in_od_set = set(pistachio.get_core_patent_title(reactions[el]["title"]) for el in
                                            itertools.chain(*created_datasets_od.values()))
                log.info(f"Exclude all relevant docs: {exclude_all_relevant_docs}, "
                         f"({len(documents_used_in_od_set)} docs excluded).")


        # 3. == Create the in distribution splits
        # 3a. get the ID reactions, two parts to this:
        # i. filter out the namerxns that should be excluded.
        id_reactions = [el for el in tqdm(shuffled_all_namerxns, desc="filtering out namerxns to exclude (creating ID sets)")
                     if not exclusion_func(reactions[el]["name_rxn"])]
        # at this point we defensively check the namerxn defs to ensure that they do not include any terms which
        # indicate reaction types that should be excluded.
        for react_repr in tqdm(id_reactions, desc="checking namerxn defs"):
            namerxn_def_warning_func(reactions[react_repr]["name_rxn_def"], reactions[react_repr]["name_rxn"])

        ## ii. filter out any reactions from documents that we used to create the ft and od test sets.
        documents_used_before = set(documents_used_in_od_set)
        num_id_reactions = len(id_reactions)
        id_reactions = [el for el in tqdm(id_reactions, desc="filtering out documents used in OOD sets (creating ID sets)")
                            if not _is_same_document_as_existing(documents_used_before, reactions, el)]
        num_removed_from_id_set_due_to_same_doc = num_id_reactions - len(id_reactions)
        log.info(f"{num_removed_from_id_set_due_to_same_doc} excluded from ID sets due to overlapping documents with OD.")

        # 3b. we can now create the id splits
        num_train_reactions_pre_limit = len(id_reactions) - (params.valid_amount + params.id_test_amount)
        train_amount = min(params.hard_limit_train, num_train_reactions_pre_limit)

        num_excluded_due_to_hardlimit = num_train_reactions_pre_limit - train_amount
        log.info(f"{num_excluded_due_to_hardlimit} excluded from train due to hard limit.")
        if num_excluded_due_to_hardlimit < 0: warnings.warn("Train not up to hard limit (before document splitting).")

        id_dataset_sizes = [
            [("valid_reactions", params.valid_amount)],
            [("id_test_reactions", params.id_test_amount)],
            [("train_reactions", train_amount)]
        ]
        (created_datasets_id, _,
            _, split_meta_id_sets) = split_utils.document_splitter_helper(id_reactions,
                                                                    id_dataset_sizes, rng, reactions, log, leeway=LEEWAY)
        assert len(created_datasets_id.keys() & created_datasets_od.keys()) == 0, "OD and ID sets should not overlap in names."

        # 4. == counters of classes
        name_rxn_counts = {}
        for dataset_name, reaction_dataset in {**created_datasets_id, **created_datasets_od}.items():
            name_rxn_counts[dataset_name] = dict(collections.Counter([reactions[el]["name_rxn"] for el in reaction_dataset]).items())

        # 5. == Create the output
        meta_dict = dict(
            namerxns_to_exclude=list(namerxns_to_exclude),
            num_removed_due_to_namerxns=num_removed_due_to_namerxns,
            num_excluded_from_train_due_to_hardlimit=num_excluded_due_to_hardlimit,
            name_rxn_counts=name_rxn_counts,
            id_meta=split_meta_id_sets,
            od_meta=split_meta_od_sets
        )
        return (created_datasets_id["train_reactions"], created_datasets_id["valid_reactions"],
                created_datasets_od["od_test_reactions"], created_datasets_id["id_test_reactions"], created_datasets_od["ft_reactions"], meta_dict)
    return split_on_name_rxns


def main():
    # Get args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--params_path", nargs='+', default=["split_def.json"])
    # ^ nb this argument accepts a list in case you want to consider different kinds of NameRxn splits.
    # Although we settled on one in the final paper, we did initially consider different dataset sizes.
    argparser.add_argument("--splits_folder", type=str, default="specific_split_defs")
    args = argparser.parse_args()

    for split_arg_filename in args.params_path:
        split_args = split_utils.SplitArgs.from_json(split_arg_filename)

        # Set up log
        log_name = (OP_FOLDER / split_args.name).with_suffix(".log")
        log = utils.get_logger(log_file_name=log_name)
        log.info(f"Arguments: {split_args.dict_items}")

        # Create split func for each of the splits
        splits = list(pathlib.Path(args.splits_folder).glob("*.json"))
        log.info(f"All splits: {splits}")
        for split_ in splits:
            with open(split_, "r") as fo:
                split_params = json.load(fo)
            log.info(f"split: {split_}")
            log.info(f"params: {split_params}")

            op_folder = OP_FOLDER / f"{split_args.name}_{split_params['name']}"
            log.info(f"Output folder is {op_folder}")

            split_func = create_split_func(split_params["excluded_name_rxns"], split_params["terms_to_check"], split_args.other_args["exclude_all_relevant_docs"])
            split_utils.main(op_folder, split_func, split_args, log)

            log.info("done split!\n")


if __name__ == '__main__':
    main()
    print("done!")
