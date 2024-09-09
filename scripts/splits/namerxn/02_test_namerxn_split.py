
import argparse
import itertools
import json
import pickle

from rxn_splits import utils


def check_no_overlap(*ds, reaction_data, get_field, set_tool=None):
    if set_tool is None:
        set_tool = lambda x, y: x.intersection(y)

    all_sets = [set(get_field(reaction_data[el]) for el in d_) for d_ in ds]
    for combo in itertools.combinations(all_sets, 2):
        if len(set_tool(combo[0], combo[1])): return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Check the namerxns splits come from different namerxns!")
    parser.add_argument("clean_data_location", type=str, help="Location of the cleaned data.")
    parser.add_argument("--train_paths", nargs='+', type=str, help="Location of the train data.")
    parser.add_argument("--ood_test_paths", type=str, nargs='+', help="Location of the OOD test data.")
    parser.add_argument("--canonicalize", action="store_true", help="Whether to canonicalize the jsonl data (for instance if it has been noncanonicalized).")
    args = parser.parse_args()


    # # Load in the clean data
    with open(args.clean_data_location, 'rb') as fo:
        data = pickle.load(fo)
    reaction_info = data["reactions"]
    print("loaded cleaned data")


    # # Load in the datasets
    loaded_train_data = []
    for pth in args.train_paths:
        with open(pth, 'r') as fo:
            d = [json.loads(el.strip()) for el in fo.readlines()]
            d = [utils.convert_jsonl_to_canon_reactions(el, args.canonicalize) for el in d]
        loaded_train_data.append(d)

    loaded_ood_data = []
    for pth in args.ood_test_paths:
        with open(pth, 'r') as fo:
            d = [json.loads(el.strip()) for el in fo.readlines()]
            d = [utils.convert_jsonl_to_canon_reactions(el, args.canonicalize) for el in d]
        loaded_ood_data.append(d)
    print("loaded datasets")


    # # Check the namerxn overlap
    for trn, tst in zip(loaded_train_data, loaded_ood_data):
        overlaps = check_no_overlap(trn, tst, reaction_data=reaction_info,
                         get_field=lambda x: x["name_rxn"])
        if overlaps:
            raise ValueError("Overlap!")

    print("looks good!")


if __name__ == '__main__':
    main()
