
import collections
import datetime
import functools
import itertools
import json
import logging
import operator
import os
import pathlib
import pickle
import re
import warnings
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from . import pistachio
from . import settings
from . import utils

EPS = 1e-6


def check_no_overlap(*args):
    """
    e.g. pass in train, val, test sets to check for no overlap.
    """
    no_overlap_check = lambda x, y: len(set(x) & set(y)) == 0
    for combo in itertools.combinations(args, 2):
        assert no_overlap_check(*combo), "sets overlap"


def write_jsonl_reactions(fname, reactions_, name, rng):
    jsonl_reactions = [utils.convert_canon_reactions_to_jsonl(react, rng) for react in
                       tqdm(reactions_, desc=f"converting reactions in {name}")]
    with open(fname, 'w') as fo:
        fo.writelines("\n".join(map(json.dumps, jsonl_reactions)))


def create_is_of_set_function(namerxns_to_include):
    """
    Create a function which returns whether the passed in name_rxn is in the `namerxns_to_exclude` set.
    Note that the `namerxns_to_exclude` set allows the definition of higher level classes. e.g., "3.2" would mean
    excluding all the 3.2 subclasses, e.g. 3.2.1, 3.2.2, etc.
    """
    levels_to_mapping = sorted([(el.count('.'), el) for el in namerxns_to_include], key=operator.itemgetter(0))
    # ^ sorted necessary as about to use groupby on first element and so want them to come in order.
    levels_to_mapping = {k: set(map(operator.itemgetter(1), v)) for k,v in itertools.groupby(levels_to_mapping, operator.itemgetter(0))}

    @functools.cache
    def is_in_set(name_rxn):
        for level, set_to_check in levels_to_mapping.items():
            name_rxn_at_level = '.'.join(name_rxn.split('.')[:level+1])
            if name_rxn_at_level in set_to_check:
                return True
        return False
    return is_in_set


def create_name_rxn_def_check(groups_to_check):
    """
    Creates a function that checks whether the namerxn def contains terms that are close to those which we wished to
     exclude, possibly suggesting we should widen the number of namerxn groups considered.

    :param groups_to_check: a list of groups to check where we will thow a warning if all regrexes within a group match.
    """
    regexs = [[re.compile(str_, flags=re.IGNORECASE) for str_ in el] for el in groups_to_check]

    @functools.cache
    def function_checker(namerxn_def, name_rxn):
        for i, regex_grp in enumerate(regexs):
            issue_warning = all(el.search(namerxn_def) for el in regex_grp)
            if issue_warning:
                warnings.warn(f"{namerxn_def} (namerxn: {name_rxn}),"
                              f" seems to have terms close to those excluded: {groups_to_check[i]}.")
    return function_checker


@dataclass
class SplitArgs:
    """
    Args for creating static splits.
    """
    valid_amount: int
    id_test_amount: int
    hard_limit_train: int
    clean_data_location: str

    random_seed: int
    name: str  # init with base name -- date and clean data location will be added automatically!

    finetune_amount: int = 0
    od_test_amount: int = 0
    date = None
    other_args: dict = None

    def __post_init__(self):
        self.date = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')

        clean_name = pathlib.Path(self.clean_data_location).stem
        self.name = f"{self.date}-{clean_name}-{self.name}"

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, "r") as fo:
            data = json.load(fo)
            out = cls(**data)
        return out

    @property
    def dict_items(self):
        return self._return_jsonable_repr(self.__dict__)

    @classmethod
    def _return_jsonable_repr(cls, dict_):
        """
        return parts that can be represented by json
        """
        out = {}
        for k, v in dict_.items():
            if isinstance(v, (float, int, str, list)):
                out[k] = v
            elif isinstance(v, dict):
                out[k] = cls._return_jsonable_repr(v)
        return out


def main(data_folder_path: pathlib.Path, split_func, params: SplitArgs, log):
    """
    :param split_func: does the actual splitting.
     Function should either return:
            train set, valid set, od_test set, id_test set, ft set, meta dictionary.
        or
            dict of reaction sets, meta dictionary.
    If using a dictionary this function should always return a "train" entry and
    any finetuning entries should begin with "ft_".
    """
    log.debug(f"data folder name: {data_folder_path}")
    try:
        os.makedirs(data_folder_path, exist_ok=False)
    except OSError:
        raise OSError("Quitting early as data folder already exists, "
                      "manually delete and rerun if sure you wish to proceed.")

    rng = np.random.RandomState(params.random_seed)

    file_hash_clean_data = utils.hash_file(params.clean_data_location)
    log.info(f"Reading in {params.clean_data_location} (hash: {file_hash_clean_data}).")
    with open(params.clean_data_location, 'rb') as fo:
        data = pickle.load(fo)

    reactions = data["reactions"]
    reactions_canon_sets_only = sorted(list(reactions.keys()))

    num_reactions = len(reactions_canon_sets_only)
    shuffled_reactions, perm = utils.permute_list_using_rng(reactions_canon_sets_only, rng)

    *different_reaction_sets, split_meta = \
        split_func(shuffled_reactions, reactions, params, log)

    # by default we will have train, valid, od_test, id_test, ft sets. (if you want to override this just return a dict)
    #todo: change so just always return a dict -- less ambiguous (this current pattern developed "organically")
    if len(different_reaction_sets) == 5:
        different_reaction_sets = dict(zip(
            ["train", "valid", "od_test", "id_test", "ft"],
            different_reaction_sets
        ))
    elif len(different_reaction_sets) == 1:
        different_reaction_sets = different_reaction_sets[0]
        assert isinstance(different_reaction_sets, dict), ("should return a dictionary of reaction sets if not doing "
                                                           "default train, val, od test, id_test, ft.")
    else:
        raise RuntimeError("unrecognized number of arguments from split function.")

    check_no_overlap(*different_reaction_sets.values())
    # should be checked by above, but check key cases again using below (as have not yet written a simple test).
    # (pretty much all splits have train, valid and id so just checking test leakage here.
    assert len(set(different_reaction_sets["train"]) & set(different_reaction_sets["id_test"])) == 0
    assert len(set(different_reaction_sets["valid"]) & set(different_reaction_sets["id_test"])) == 0

    hashes = {}
    reaction_sizes = {}
    for name, reactions_ in different_reaction_sets.items():
        fname = data_folder_path / f"{name}.jsonl"
        write_jsonl_reactions(fname, reactions_, name, rng)
        hashes[name] = utils.hash_file(fname)
        reaction_sizes[name] = len(reactions_)
        log.info(f"{name} is {len(reactions_)} in size.")
    log.info("Written out the dataset files.")

    # Write out a combined train & ft if have finetuning reactions.
    for name, reactions_ in different_reaction_sets.items():
        if name.startswith("ft_") and len(reactions_) > 0:
                name = f"train_and_{name}"
                fname = data_folder_path / f"{name}.jsonl"
                train_and_ft_reactions = different_reaction_sets["train"] + reactions_
                rng.shuffle(train_and_ft_reactions)
                write_jsonl_reactions(fname, train_and_ft_reactions, name, rng)
                hashes[name] = utils.hash_file(fname)
                reaction_sizes[name] = len(train_and_ft_reactions)
                log.info(f"{name} is {len(train_and_ft_reactions)} in size.")
                log.info(f"Written out {name} file as finetune set was not empty.")

    meta_dict = dict(
        file_hash_clean_data=file_hash_clean_data,
        out_hashes=hashes,
        param_dict=params.dict_items,
        num_reactions=num_reactions,
        split_meta=split_meta,
        reaction_sizes=reaction_sizes,
        watermark=utils.get_watermark(),
        perm=perm
    )
    with open(data_folder_path / "meta.json", 'w') as fo:
        json.dump(meta_dict, fo)
    log.info(f"written out split meta data to {data_folder_path}!")


def document_splitter_helper(shuffled_reactions: list, dataset_sizes: list, rng: np.random.RandomState,
                             reactions: dict, log: logging.Logger, doc_first_two: str="US", leeway: int = 0):
    """
    Creates a document-based split of the different reactions.

    Note this split will be "document dense".

    :param shuffled_reactions: a list of reactions (shuffled) in canonical (i.e., frozenset) form.
    :param dataset_sizes: a list defining the size of the splits to make. each item of the list is a list of tuples of
           the form (split_name, split_size). These are the splits to create from the same documents and their respective
            sizes. As an example, say dataset_sizes = `[[("train", 100), ("valid", 20)], [("test", 20)]]`, then this
            function will create three splits. The first two splits will be created from the same documents, but the
            first split will be 100 reactions and the second split will be 20 reactions. The third split will be created
            from a different set of documents and will be 20 reactions.
    :param rng: random number generator.
    :param reactions: a dictionary mapping from the canonical reaction to the reaction dictionary (contains the title).
    :param log: log to write debug and info messages to.
    :param leeway: whether to allow the fact that the final created dataset sizes might not exactly match the dataset sizes
        fed in. This can happen if you are trying to use all the reactions in the dataset, but the dataset/document sizes
        are such that you need part of a document to complete a split and so have to throw them away from the other splits.
        Leeway is the number of reactions to allow the created dataset sizes (at a given level) to be different from the
        requested dataset size. If the difference is greater than the leeway, an error is thrown.
        Note:
         1. that the earlier datasets in dataset_sizes are created first, so the latter items in this list are likely to
        be the datasets affected.

    :return:
        * created_datasets: a dictionary mapping from the dataset names to the list of reactions in that split.
        * shuffled_documents: a list of the shuffled documents used to create the splits.
        * doc_title_to_reactions: a dictionary mapping from document titles to the reactions in that document.
        * split_meta: a dictionary containing split information, e.g., the number of documents used etc.
    """
    # Create a mapping from document titles to reactions:
    doc_title_to_reactions = collections.defaultdict(list)
    num_unknown_document_titles = 0
    for reaction_ in shuffled_reactions:
        patent_title = reactions[reaction_]["title"]
        if patent_title == "" or patent_title == settings.UNKNOWN_STR:
            num_unknown_document_titles += 1
            continue  # skip unknown patent titles.
        core_patent_title = pistachio.get_core_patent_title(patent_title)
        doc_title_to_reactions[core_patent_title].append(reaction_)
    if num_unknown_document_titles > 0:
        log.warning(f"Found {num_unknown_document_titles} reactions with unknown document titles. Skipping these.")
    log.info(f"There are {len(doc_title_to_reactions)} documents in the dataset.")
    log.info(f"made up of {sum(map(len, doc_title_to_reactions.values()))} reactions.")

    # Get a list of all documents shuffled.
    shuffled_documents = list(doc_title_to_reactions.keys())
    rng.shuffle(shuffled_documents)

    # Go through and fill up the different split levels with reactions from successive patents until we have got enough to
    # form the splits. (we will split these among the splits within a level later).
    num_documents_used_per_split = []
    split_sizes = collections.deque([sum(inner_el[1] for inner_el in el) for el in dataset_sizes])
    split_up_reactions = []  # <- will hold the reactions for each level (i.e., list of lists).
    current_split = []  # <- will hold the reactions for the current level being assembled.
    for i, patent_title in enumerate(shuffled_documents):
        reactions_want_from_split = split_sizes[0] - len(current_split)  # target size - current size

        current_split.extend(doc_title_to_reactions[patent_title][:reactions_want_from_split])

        new_reactions_want_from_split = split_sizes[0] - len(current_split)  # target size -  _new_ current size
        if new_reactions_want_from_split <= 0:
            log.info(
                f"Filled up all the datasets at level {len(split_up_reactions)}."
                f"Moving to next level. Have used {i + 1} patents so far.")
            split_up_reactions.append(current_split)
            current_split = []
            split_sizes.popleft()
            num_documents_used_per_split.append(i + 1)

        if len(split_sizes) == 0:
            break
    else:
        # if we get here, then we have not filled up all the datasets with the documents we have access to. if we have
        # broken the leeway constraint then throw an error.

        # first finish up last split
        split_up_reactions.append(current_split)
        num_documents_used_per_split.append(i + 1)

        # then check whether it's ok with leeway
        target_split_sizes = np.array([sum(inner_el[1] for inner_el in el) for el in dataset_sizes])
        actual_split_sizes = np.array([len(el) for el in split_up_reactions])
        max_difference = np.max(target_split_sizes - actual_split_sizes)
        if max_difference > leeway:
            raise RuntimeError(f"Could not fill up all the datasets with the given sizes. Max difference "
                               f"was {max_difference}. Leeway was {leeway}.")
        else:
            log.warning(f"Could not fill up all the datasets with the given sizes. Using {actual_split_sizes} instead."
                        f"The max difference from proposed was {max_difference}."
                        f" Leeway was {leeway}.")
    num_docs_used = i + 1
    log.info(f"Used {num_docs_used} patent documents (out of {len(shuffled_documents)} total) to create splits.")

    # Now go through the split levels again and split the number of reactions in each level to the different datasets.
    created_datasets = {}
    actual_dataset_sizes = []
    for i, (shuffled_reactions_at_level, dataset_details_at_level) in enumerate(zip(split_up_reactions, dataset_sizes)):

        rng.shuffle(shuffled_reactions_at_level)  # as consecutive reactions in the document will be together.
        actual_dataset_sizes_at_level = []

        start = 0
        for (split_name, split_size) in dataset_details_at_level:
            assert split_name not in created_datasets, "should not have the same dataset name multiple times."
            created_datasets[split_name] = shuffled_reactions_at_level[start:start + split_size]
            start += (num_reactions_actually_used := len(created_datasets[split_name]))
            actual_dataset_sizes_at_level.append((split_name, num_reactions_actually_used))
            if num_reactions_actually_used != split_size:
                log.warning(f"Could not create dataset {split_name} with size {split_size}."
                            f" Using {len(created_datasets[split_name])} instead.")
        actual_dataset_sizes.append(actual_dataset_sizes_at_level)

    # Check that we are using only one kind of document (otherwise might get more or less the same document in different
    # patent jurisdictions).
    first_two_chars = set([el[:2] for el in shuffled_documents])
    assert len(first_two_chars) == 1, f"Expected all patents to start with the same two characters, but got {first_two_chars}"
    assert first_two_chars.pop() == doc_first_two, f"Expected all patents to start with {doc_first_two}, but got {first_two_chars}"

    # Create split meta:
    split_meta = {
        "num_unknown_patent_titles": num_unknown_document_titles,
        "num_total_documents": len(shuffled_documents),
        "num_documents_used": num_docs_used,
        "num_documents_used_per_level": num_documents_used_per_split,
        "dataset_levels": dataset_sizes,
        "actual_dataset_levels": actual_dataset_sizes,
        "leeway": leeway,
    }

    return created_datasets, shuffled_documents, doc_title_to_reactions, split_meta


def author_document_based_splitter(shuffled_reactions: list, dataset_sizes: list, rng: np.random.RandomState,
                             reactions: dict, log: logging.Logger, doc_first_two: str="US", leeway: int = 0,
                                   author_overbuffer: int = 0):
    """
    Two level split on authors and then documents. Makes use of `document_splitter_helper` for second level.
    Authors are lower cased and stripped of whitespace before being used.

    Notes:
        * will not use any reactions that do not have any authors or document titles.
        * this split will be both "author dense" and "document dense".

    :param shuffled_reactions: a list of reactions (shuffled) in canonical (i.e., frozenset) form.
    :param dataset_sizes: a list defining the size of the splits to make. Each item of the list is a list of the
            document based splits to make (i.e., what would be fed into something like document_splitter_helper).
            Specifically each of these inner lists is of tuples of the form (split_name, split_size).
            These are the splits to create from the same documents and their respective sizes.

            As an example, say
            ```dataset_sizes = [
                [
                    [("author_ood", 100)]
                ],
                [
                    [("train", 100), ("valid", 20)],
                    [("test", 20)]
                ]
            ]```
            Then this function will create four splits.
            The first will be created from different authors (and documents) to the next three and will be 100 reactions
             in size. The next splits (splits 2, 3, and 4) will not necessarily be created from different authors.
            Splits 2 and 3 will be created from the same documents, with split 2 containing 100 reactions
            and split 3 20. The final split (split 4 or "test") will be created from a different set of documents to
             split 2 and 3, but possibly the same authors. It will be 20 reactions in size.
    :param rng: random number generator.
    :param reactions: a dictionary mapping from the canonical reaction to the reaction dictionary (contains the title).
    :param log: log to write debug and info messages to.
    :param leeway: This integer controls whether to allow the fact that the final created dataset sizes might not
        exactly match the dataset sizes fed in (and how much this discrepancy can be). This can happen if you are trying
         to use all the reactions in the dataset, but the dataset/document sizes are such that you need part of a
         document to complete a split and so have to throw them away from the other splits. Leeway is the number of
         reactions to allow the created dataset sizes (at a given level) to be different from the requested dataset size.
          (it is only used at the document splitting level). If the difference is greater than the leeway, an error is
          thrown.
        Note:
         1. that the earlier datasets in dataset_sizes are created first, so the latter items in this list are likely to
        be the datasets affected.
    :param author_overbuffer: how far to go over the number of reactions needed in the highest level split before calling
        the document splitter. Can consider increasing if getting leeway errors, but note that it will affect the
        "denseness" of the author split.

    :return:
        * created_datasets: a dictionary mapping from the dataset names to the list of reactions in that split.
        * shuffled_authors: a list of the shuffled authors used to create the splits.
        * doc_title_to_reactions: a dictionary mapping from document titles to the reactions in that document.
        * split_meta: a dictionary containing split information, e.g., the number of documents used etc.
    """

    # Create a mapping of authors to documents, and documents to reactions:
    author_to_documents = collections.defaultdict(list)
    doc_title_to_reactions = collections.defaultdict(list)
    num_unknown_document_titles = 0
    num_unknown_authors = 0  # defined as empty author list or unknown str.
    for reaction_ in shuffled_reactions:
        # document title
        patent_title = reactions[reaction_]["title"]
        if patent_title == "" or patent_title == settings.UNKNOWN_STR:
            num_unknown_document_titles += 1
            continue  # skip unknown patent titles.
        core_patent_title = pistachio.get_core_patent_title(patent_title)
        # ^ get patent title but hold off adding this info to the doc_title_to_reactions until we know we have authors.

        # author
        authors = reactions[reaction_]["authors"]
        if authors == settings.UNKNOWN_STR or len(authors) == 0:
            num_unknown_authors += 1
            continue
        else:
            for auth in reactions[reaction_]["authors"]:
                canonical_auth = auth.lower().strip()  # <- could be more fancy than that, but then I guess might
                # accidentally deduplicate authors that should not be deduplicated.
                author_to_documents[canonical_auth].append(core_patent_title)

        # If we've reached here we had some authors, so we can add the reaction to the document list too.
        doc_title_to_reactions[core_patent_title].append(reaction_)

    if num_unknown_document_titles > 0:
        log.warning(f"Found {num_unknown_document_titles} reactions with unknown document titles. Skipping these.")
    if num_unknown_authors > 0:
        log.warning(f"Found {num_unknown_authors} reactions with unknown document authors. Skipping these.")
    log.info(f"There are {len(doc_title_to_reactions)} documents in the dataset.")
    log.info(f"There are {len(author_to_documents)} authors in the dataset.")
    log.info(f"There are  {sum(map(len, doc_title_to_reactions.values()))} reactions in the dataset.")


    # Get a list of all authors shuffled.
    shuffled_authors = list(author_to_documents.keys())
    rng.shuffle(shuffled_authors)


    # Go through and fill up the different split levels with reactions from successive authors until enough.
    # ## We will first set up some containers for data we'll need in the loop
    # ### Info needed after.
    num_authors_used_per_split = []
    documents_used = set()  # <- store the documents used. note the first author we find will "claim" the document
    # and so it cannot be used by listed authors after.
    level_metas = []
    created_datasets = {}
    level_working_on = 0

    # ### Queue to manage the split sizes at each highest level.
    def get_split_size(split_level):
        # each split level is a list of list of tuples, ie what would be given to document_splitter_helper
        inner_sums = sum([sum(inner_el[1] for inner_el in el) for el in split_level])
        return inner_sums
    split_sizes = collections.deque([get_split_size(el) for el in dataset_sizes])

    # ### Info for the current level
    class CurrentLevel:
        def __init__(self):
            self.current_reactions = None  # <- will hold the reactions for the current level being assembled.
            self.current_docs = None  # <- will hold the docs for the current level being assembled.
            self.reset()

        def reset(self):
            self.current_docs = []
            self.current_reactions = []

        @property
        def num_reactions_for_current_level(self):
            return len(self.current_reactions)

    current_lvl_info = CurrentLevel()

    # ## Now we can start the loop over authors
    for i, author in enumerate(shuffled_authors):

        # ### add this author's reactions (from docs they are associated with and we have not already used) to the
        # current level's reactions.
        documents = utils.permute_list_using_rng(author_to_documents[author], rng)[0]
        for doc in documents:
            if doc not in documents_used:  # could have been used for another author.
                current_lvl_info.current_docs.extend(doc)
                current_lvl_info.current_reactions.extend(doc_title_to_reactions[doc])
                documents_used.add(doc)

        # ### check if we have filled up the current level and if so run the document splitter
        if current_lvl_info.num_reactions_for_current_level >= (split_sizes[0] + author_overbuffer):
            log.info(
                f"Author Splits: filled up all the datasets at level {level_working_on}."
                f"Have used {i + 1} authors so far. Performing doc based split")

            # we can now send it down to the document splitter helper for final splitting
            created_datasets_at_lvl, _, _, split_meta_at_lvl = document_splitter_helper(
                                                utils.permute_list_using_rng(current_lvl_info.current_reactions, rng)[0],
                                            dataset_sizes[level_working_on], rng, reactions, log, doc_first_two, leeway)
            assert len(created_datasets_at_lvl.keys() & created_datasets.keys()) == 0, "should not have the same dataset name multiple times."
            created_datasets.update(created_datasets_at_lvl)
            level_metas.append(split_meta_at_lvl)

            # check that the reactions used in the datasets are only from the docs used at this level.
            # (this check should not be necessary, but added it in case to ensure code is working as expected).
            for rxn_in_ds in created_datasets_at_lvl.values():
                docs_used = set(pistachio.get_core_patent_title(reactions[el]["title"]) for el in rxn_in_ds)
                assert len(set(current_lvl_info.current_docs) - docs_used) >= 0, "should not have used other docs."

            # record stats
            num_authors_used_per_split.append(i + 1)

            # reset counters
            current_lvl_info.reset()

            # pop off this level and record onto next
            level_working_on += 1
            split_sizes.popleft()

            log.info(
                f"Finished doc split at author level. Moving to next level.\n\n")

        # ### If we have filled up all the levels then we can break out of the loop.
        if len(split_sizes) == 0:
            break


    # # Check that the created datasets do not have any overlap
    # (this check should not be necessary, but added it in case to ensure code is working as expected).
    check_no_overlap(*created_datasets.values())
    for ds in created_datasets.values():
        assert len(ds) == len(set(ds)), "should not have duplicates within a dataset."


    # # Create split meta
    split_meta = {
        "num_unknown_patent_titles": num_unknown_document_titles,
        "num_unknown_authors": num_unknown_authors,
        "num_total_authors": len(author_to_documents),
        "upper_bound_on_num_authors_used_per_level": num_authors_used_per_split,
        # ^ upper bound as may not actually use the reactions discovered by this author. (e.g., they may only belong
        # to the same document as another author already used).
        "dataset_levels": dataset_sizes,
        "actual_dataset_sizes": {k: len(v) for k, v in created_datasets.items()},
        "leeway": leeway,
        "author_overbuffer": author_overbuffer,
        "individual_level_metas": level_metas
    }

    return created_datasets, shuffled_authors, author_to_documents, split_meta
