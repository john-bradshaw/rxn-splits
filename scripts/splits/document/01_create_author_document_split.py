"""
Creates an author- and document-based split, i.e. on both different authors _and_ different patents.

Note:
    * make simplifying assumption author names are correct and unique (i.e., do minimal pre-processing on names)
    * we assume dealing with US patents only and that the patent number is in the format USXXXXXXX.
    * the patent name we will use will only be the first part of the title, ignoring anything after an underscore.
        e.g., USXXXXXXX_A1
    * we operate on the cleaned reaction database (which has been deduplicated) and so do not take into account the fact
     that the reaction may also be in more than one patent.
"""
import argparse
from dataclasses import dataclass
import json
import logging
import pathlib

import numpy as np

from rxn_splits import utils
from rxn_splits import split_utils

OP_FOLDER = pathlib.Path(__file__).parent / pathlib.Path("outputs")


@dataclass
class AuthSplitArgs(split_utils.SplitArgs):
    """
    Extend SplitArgs to add an additional author OOD test amount.
    """
    author_od_test_amount: int = 0


def author_document_split_creator(shuffled_reactions, reactions, params: AuthSplitArgs, log: logging.Logger):
    """
    Splits the reactions based on the author and document (i.e., patents).

    """
    # Create a queue of the datasets we need to fill up
    # if we need to fill up multiple datasets simultaneously, we will store in a tuple at the same level in the queue.
    dataset_sizes = [
        # Author Level
        [
            # Document Level
            [
                ("author_od_test", params.author_od_test_amount),
            ]
        ],
        [
            [
                ("doc_od_test", params.od_test_amount),
                ("ft", params.finetune_amount)
            ],
            [
                ("train", params.hard_limit_train),
                ("id_test", params.id_test_amount),
                ("valid", params.valid_amount)
            ]
        ]
    ]
    rng = np.random.RandomState(params.random_seed)

    # Create the splits.
    created_datasets, shuffled_authors, authors_to_docs, author_split_details = \
        split_utils.author_document_based_splitter(
            shuffled_reactions, dataset_sizes, rng, reactions, log, author_overbuffer=200
        )

    # Create the meta
    amounts = {f"{k}_amount": len(v) for k, v in created_datasets.items()}
    meta_dict = dict(
        **amounts,
        random_seed_used_to_shuffle_patents=params.random_seed,
        document_split_details=author_split_details
    )

    return created_datasets, meta_dict


def main():
    # Get args
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--split_def_path', type=str)
    args = argparser.parse_args()

    # Load in the split definition from json
    with open(args.split_def_path, 'r') as fo:
        split_args = json.load(fo)

    # Set up log
    log_name = (OP_FOLDER / split_args["name"]).with_suffix(".log")
    log = utils.get_logger(log_file_name=log_name)
    log.info(f"Arguments: {split_args}")

    # Set up output folder
    op_folder = OP_FOLDER / split_args["name"]
    log.info(f"Output folder is {op_folder}")

    # Create split args
    split_args = AuthSplitArgs(
        valid_amount=split_args["valid_amount"],
        id_test_amount=split_args["id_test_amount"],
        hard_limit_train=split_args["train_amount"],

        od_test_amount=split_args["od_test_amount_docs"],
        author_od_test_amount=split_args["od_test_amount_authors"],
        finetune_amount=split_args.get("finetune_amount", 0),

        clean_data_location=split_args["clean_data_location"],
        random_seed=split_args["random_seed"],
        name=split_args["name"]
    )

    # do the split
    split_utils.main(op_folder, author_document_split_creator, split_args, log)
    log.info("done split!")


if __name__ == '__main__':
    main()
    print("Done!")
