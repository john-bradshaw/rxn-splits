"""
Create a time-based split.

Logic:
    1. Organize reactions by year.
    2. For each year, create a train, val, and test set. If document based split then pick train, valid,
        and test from different documents, by going through each document in a random order and adding all of its reactions
        to one of the different train/val/test sets until the target sizes are reached
        If not document based (this path has not been tested recently),
         then just pick a random subset of the reactions from that year.
        Note:
            - test set is a fixed number of reactions and only split off if the year is above a certain size.
            - val set is (roughly) a fixed proportion of the remaining reactions (rest go to train).
    3. Create the different time splits by taking all the train and val reactions up to a given year and subsampling if
        necessary.
"""

import argparse
import collections
import datetime
import itertools
import json
import os
import pathlib
import pickle
import warnings
from dataclasses import dataclass

import numpy as np

from rxn_splits import split_utils
from rxn_splits import utils

OP_FOLDER = pathlib.Path(__file__).parent / pathlib.Path("outputs")


def _shuffle_and_select_subset(reactions, rng, max_num_to_select):
    """
    :param reactions: iterable of reactions
    :param max_num_to_select: Upperbound on the number of reactions to return. If np.inf will select the whole set.
    """
    # todo: duplicated code with utils for first part.
    num_reactions = len(reactions)

    perm = [int(i) for i in rng.permutation(num_reactions)]
    reactions = [reactions[i] for i in perm]

    if max_num_to_select == np.inf:
        max_num_to_select = len(reactions)

    out = reactions[:max_num_to_select]

    return out


def _get_yr_reactions(yr, reactions_canon_sets_only, all_reactions):
    """
    get reactions belonging to a given year.
    """
    return [el for el in reactions_canon_sets_only if (isinstance(all_reactions[el]["year"], int)
                                         and all_reactions[el]["year"] == yr)]


@dataclass
class TimeSplitArgs:
    """
    Parameters for creating a series of time-based splits.
    """
    years_to_break: list   # list of years to create the time splits on.
    clean_data_location: str  # path (relative) to the run directory to load to create the splits.

    document_split: bool = True  # whether to ensure train, val, and test for each year come from separate documents
    control_for_training_size: bool = False  # whether to ensure that the training set size is fixed for each split.
    valid_proportion: int = 0.05  # how much of each train/validation amount for each year gets put into a validation set.
    test_amount: int = 2000  # test set size for each year. If we don't have at least double this for a year then we
                             #  will not create a test set for that year.

    max_training_size: int = 1000000000  # upper bound on the training set size for each split.
    max_val_size: int = 1000000000  # upper bound on the validation set size for each split.
    random_seed: int = 3284
    name: str = "time_split"  # name of the split -- note will get edited with date and clean data location in post_init.

    def __post_init__(self):
        self.date = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
        clean_name = pathlib.Path(self.clean_data_location).stem
        self.name = f"{self.date}-{clean_name}-{self.name}"

        assert 0. <= self.valid_proportion <= 1., "valid_proportion must be between 0 and 1."

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as fo:
            params = json.load(fo)
        return cls(**params)

    @property
    def dict_items(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (float, int, str, list))}


def main():
    # == Get args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--params_path", default="time_split_def.json")
    args = argparser.parse_args()

    # == Load in the split definition from json
    params = TimeSplitArgs.from_json(args.params_path)

    # == Create overall folder
    splits_op_folder = OP_FOLDER / params.name
    try:
        os.makedirs(splits_op_folder, exist_ok=False)
    except OSError:
        raise OSError(f"Quitting early as data folder ({splits_op_folder}) already exists, "
                      "manually delete and rerun if sure you wish to proceed!")

    # == Set up log
    log_name = (splits_op_folder / params.name).with_suffix(".log")
    log = utils.get_logger(log_file_name=log_name)
    log.info(f"Arguments: {params.dict_items}")

    # == Setup (seeds, meta dict)
    rng = np.random.RandomState(params.random_seed)
    meta_dict = {"run-params": params.dict_items}  # <-- store the parameters used

    # == Read in the cleaned reactions and shuffle
    file_hash_clean_data = utils.hash_file(params.clean_data_location)
    log.info(f"Reading in {params.clean_data_location} (hash: {file_hash_clean_data}).")
    with open(params.clean_data_location, 'rb') as fo:
        data = pickle.load(fo)

    reactions = data["reactions"]
    reactions_canon_sets_only = sorted(list(reactions.keys()))

    num_reactions = len(reactions_canon_sets_only)
    perm = [int(i) for i in rng.permutation(num_reactions)]
    shuffled_reactions = [reactions_canon_sets_only[i] for i in perm]

    # == Calculate the number of reactions per year.
    # will write out if we are losing any non-integer years (e.g., if they have been encoded incorrectly etc).
    year_counts_all = collections.Counter([el["year"] for el in reactions.values()])
    year_counts = {k: v for k, v in year_counts_all.items() if isinstance(k, int)}
    assert len(year_counts)
    log.debug(f"Removing years {set(year_counts_all.keys()) - set(year_counts.keys())} from set as not integer years.")
    total_reactions = sum(year_counts.values())
    log.info(f"Total reactions is {total_reactions}")
    meta_dict["total_reactions"] = total_reactions
    meta_dict["year_counts_all"] = year_counts_all
    meta_dict["year_counts"] = year_counts

    # == Split each year into train/valid/test
    yr_train = {}
    yr_val = {}
    yr_test = {}
    meta_dict["ds-sizes"] = {}
    meta_dict["doc-details"] = {}
    for yr in sorted(list(year_counts.keys())):
        log.debug(f"doing year {yr}")

        # Some datasets had weird years pre 1900s, so if this is the case then just skip these!
        if yr < 1900:
            warnings.warn(f"Skipping year {yr}! (as pre-1900!)")
            year_counts.pop(yr)   # <-- also removing it from counts so do not erroneously try to use these reactions later.
            continue

        reactions_for_yr = _get_yr_reactions(yr, shuffled_reactions, reactions)
        num_reactions_per_yr = len(reactions_for_yr)

        # work out the split sizes -- note if the full set is not at least two times the test set size then we will
        # not create a test set for this year.
        if num_reactions_per_yr > 2 * params.test_amount:
            test_amount = params.test_amount
        else:
            log.warning(f"Not enough reactions for year {yr} to create a test set (only {num_reactions_per_yr})!")
            test_amount = 0
        train_val_amount = num_reactions_per_yr - test_amount
        train_amount = int(train_val_amount * (1 - params.valid_proportion))
        valid_amount = train_val_amount - train_amount

        if params.document_split:
            # ^ note we can do this here as do not have documents across multiple years.

            # fill up the test and valid sets first as these are smaller. (the distribution of document sizes might
            # be such that we cannot get the exact number of reactions for each set we want -- see `leeway` below).
            if test_amount > 0:
                 dataset_sizes = [[("test", test_amount)]]
            else:
                dataset_sizes = []
            dataset_sizes.extend([
                [("valid", valid_amount)],
                [("train", train_amount)],
            ])

            created_data, *_, split_meta = split_utils.document_splitter_helper(
                reactions_for_yr, dataset_sizes, rng, reactions, log, leeway=750
            )

            meta_dict["doc-details"][yr] = split_meta
            yr_train[yr] = created_data["train"]
            yr_val[yr] = created_data["valid"]
            if test_amount > 0:
                yr_test[yr] = created_data["test"]
        else:
            yr_train[yr] = reactions_for_yr[:train_amount]
            yr_val[yr] = reactions_for_yr[train_amount:train_amount+valid_amount]
            if test_amount > 0:
                yr_test[yr] = reactions_for_yr[-test_amount:]

        meta_dict["ds-sizes"][yr] = dict(all=num_reactions_per_yr,
                                         train=len(yr_train[yr]),
                                         val=len(yr_val[yr]),
                                         test=len(yr_test.get(yr, [])))

    split_utils.check_no_overlap(*yr_train.values(), *yr_val.values(), *yr_test.values())
    # ^ reactions should have been deduplicated at this stage, but just checking.

    # == We then work out the training/validation set sizes -- we maybe want to control for this when considering
    # different years so pick the largest we can.
    min_split_year = min(params.years_to_break)
    if params.control_for_training_size:
        log.debug("Controlling for training set size (working out largest possible size by examining earliest split).")
        number_in_smallest_set_train = sum([len(yr_train[yr]) for yr in year_counts if yr <= min_split_year])
        log.info(f"Number in the smallest set: {number_in_smallest_set_train} (train).")
        meta_dict["min-train-size"] = number_in_smallest_set_train
    else:
        log.debug("Not controlling for training set size")
        number_in_smallest_set_train = np.inf
    max_train_size = min(number_in_smallest_set_train, params.max_training_size)
    max_val_size = params.max_val_size

    # == Write out the year tests
    test_yr_folder = splits_op_folder / "test_years"
    os.makedirs(test_yr_folder, exist_ok=False)
    meta_dict['hashes-test'] = {}
    for yr, test_reactions in yr_test.items():
        test_file_name = test_yr_folder / f"test-{yr}.jsonl"
        split_utils.write_jsonl_reactions(test_file_name, test_reactions, f"test-{yr}", rng)
        meta_dict['hashes-test'][yr] = utils.hash_file(test_file_name)
    log.info("Written out the different year test files.")

    # == Split out different training/validation sets. Note that we will mix up the years before sampling so that we
    # are not using the same number of reactions per year.
    # note that we are not keeping subsampling wrt year, and so will likely end up with more recent reactions.
    train_val_splits_folder = splits_op_folder / "train_val_splits"
    os.makedirs(train_val_splits_folder, exist_ok=False)
    meta_dict['amounts-excluded'] = {}
    meta_dict['yr-set-sizes'] = {}
    meta_dict['hashes'] = {}
    for yr in params.years_to_break:
        train_val_yr_folder = train_val_splits_folder / f"y{yr}"
        os.makedirs(train_val_yr_folder, exist_ok=False)

        # === Train
        possible_train_reactions = list(itertools.chain(*[r for y, r in yr_train.items() if y <= yr]))
        selected_train_reactions = _shuffle_and_select_subset(possible_train_reactions, rng, max_train_size)
        # ^ note that the shuffle is necessary as we want to mix the years up
        train_file_name = train_val_yr_folder / "train.jsonl"

        split_utils.write_jsonl_reactions(train_file_name, selected_train_reactions, f"train-{yr}", rng)

        # === Valid
        possible_val_reactions = list(itertools.chain(*[r for y, r in yr_val.items() if y <= yr]))
        selected_val_reactions = _shuffle_and_select_subset(possible_val_reactions, rng, max_val_size)
        val_file_name = train_val_yr_folder / "valid.jsonl"
        split_utils.write_jsonl_reactions(val_file_name, selected_val_reactions, f"valid-{yr}", rng)


        # === Meta details
        meta_dict['amounts-excluded'][yr] = {
            'train': len(possible_train_reactions) - len(selected_train_reactions),
            'val': len(possible_val_reactions) - len(selected_val_reactions)
        }
        meta_dict['yr-set-sizes'][yr] = {
            'train': len(selected_train_reactions),
            'val': len(selected_val_reactions)
        }
        log.info(f"{yr}: dataset sizes, {meta_dict['yr-set-sizes'][yr]}")

        meta_dict['hashes'][yr] = dict(train=utils.hash_file(train_file_name), valid=utils.hash_file(val_file_name))
    log.debug("Written out the train/val files.")

    # == Sort out and print the meta data file.
    meta_dict['file_hash_clean_data'] = file_hash_clean_data
    meta_dict['num_reactions'] = num_reactions

    with open(splits_op_folder / "meta.json", 'w') as fo:
        json.dump(meta_dict, fo)
    log.debug("written out split meta data!")


if __name__ == '__main__':
    main()
    print("Done!")

