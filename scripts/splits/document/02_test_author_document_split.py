"""
Script to check train, OOD doc test, OOD authors come from different authors and documents.
"""
import argparse
import itertools
import json
import pickle

from rxn_splits import utils


def check_no_overlap(*ds, reaction_data, get_field, with_unpacking=False, set_tool=None):
    if set_tool is None:
        set_tool = lambda x, y: x.intersection(y)
    if with_unpacking:
        all_sets = [set(itertools.chain(*(get_field(reaction_data[el]) for el in d_))) for d_ in ds]
    else:
        all_sets = [set(get_field(reaction_data[el]) for el in d_) for d_ in ds]
    for combo in itertools.combinations(all_sets, 2):
        if len(set_tool(combo[0], combo[1])): return True
    return False


def potential_author_overlap(train_ds, ood_ds, reaction_data, get_field):
    train_authors = set(itertools.chain(*(get_field(reaction_data[el]) for el in train_ds)))
    ood_authors = [set(get_field(reaction_data[el])) for el in ood_ds]
    for author_set in ood_authors:
        if not len(author_set - train_authors):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Check the author and document splits.")
    parser.add_argument("clean_data_location", type=str, help="Location of the cleaned data.")
    parser.add_argument("train_path", type=str, help="Location of the train data.")
    parser.add_argument("id_test_path", type=str, help="Location of the ID test data data.")
    parser.add_argument("ood_doc_path", type=str, help="Location of the OOD doc test data.")
    parser.add_argument("ood_auth_path", type=str, help="Location of the OOD author test data.")
    parser.add_argument("--canonicalize", action="store_true", help="Whether to canonicalize the jsonl data (for instance if it has been noncanonicalized).")
    args = parser.parse_args()

    # # Load in the clean data
    with open(args.clean_data_location, 'rb') as fo:
        data = pickle.load(fo)
    reaction_info = data["reactions"]
    print("loaded cleaned data")

    # # Load in the datasets
    loaded_data = {}
    for nm, pth in [("train", args.train_path), ("id_test", args.id_test_path), ("ood_doc", args.ood_doc_path), ("ood_auth", args.ood_auth_path)]:
        with open(pth, 'r') as fo:
            d = [json.loads(el.strip()) for el in fo.readlines()]
            d = [utils.convert_jsonl_to_canon_reactions(el, args.canonicalize) for el in d]
        loaded_data[nm] = d
    print("loaded datasets")

    # # Check the document overlap
    overlaps = check_no_overlap(loaded_data["train"], loaded_data["ood_doc"], loaded_data["ood_auth"], reaction_data=reaction_info,
                     get_field=lambda x: x["title"].split("_")[0])
    if overlaps:
        raise ValueError("There is an overlap between the train, OOD doc test, and OOD author test data with documents.")

    # # Check the authors overlap
    overlaps = potential_author_overlap(loaded_data["train"], loaded_data["ood_auth"], reaction_data=reaction_info,
                     get_field=lambda x: x["authors"])
    if overlaps:
        raise ValueError("There is an overlap between the train and OOD author test data with authors.")

    # # Check the ID test does overlap with the train
    overlaps = check_no_overlap(loaded_data["train"], loaded_data["id_test"], reaction_data=reaction_info,
                     get_field=lambda x: x["title"].split("_")[0])
    if not overlaps:
        raise ValueError("There is no overlap between the train and ID test data with documents. (expected some!)")
    # note document overlap implies author overlap so do not need to test this.

    print("Looks good!")


if __name__ == "__main__":
    main()
