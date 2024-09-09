
from rxn_splits import utils


def test_permute_list_using_rng():
    input_list = [0, -1, 3, 5, 7]
    class MockedRNG:
        def permutation(self, n):
            return [2, 3, 4, 1, 0]

    out, _ = utils.permute_list_using_rng(input_list, MockedRNG())
    expected = [3, 5, 7, -1, 0]
    assert out == expected


def test_convert_jsonl_to_canon_reactions():
    jsonl = {"translation": {"reactants": "C1COCCO1.C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl.Cl.Cl",
                             "products": "C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O.Cl.Cl"}}
    out = utils.convert_jsonl_to_canon_reactions(jsonl)
    expected = (frozenset({('C1COCCO1', 1),
                          ('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1),
                          ('Cl', 2)}),
              frozenset({('C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O', 1), ('Cl', 2)}))
    assert out == expected


def test_convert_canon_reactions_to_jsonl():
    import numpy as np
    reactions_as_frozensets = (frozenset({('C1COCCO1', 1),
                          ('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1),
                          ('Cl', 2)}),
              frozenset({('C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O', 1), ('Cl', 2)}))
    out = utils.convert_canon_reactions_to_jsonl(reactions_as_frozensets, np.random.RandomState(43))
    assert set(out["translation"]["reactants"].split(".")) == {"C1COCCO1", "C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl", "Cl", "Cl"}
    assert set(out["translation"]["products"].split(".")) == {"C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O", "Cl", "Cl"}


def test_convert_canon_reactions_to_smiles():
    import numpy as np
    reactions_as_frozensets = (frozenset({('C1COCCO1', 1),
                          ('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1),
                          ('Cl', 2)}),
              frozenset({('C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O', 1), ('Cl', 2)}))
    out = utils.convert_canon_reactions_to_smiles(reactions_as_frozensets, np.random.RandomState(43))
    reactants, products = out.split(">>")
    assert set(reactants.split(".")) == {"C1COCCO1", "C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl", "Cl", "Cl"}
    assert set(products.split(".")) == {"C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O", "Cl", "Cl"}


def test_cast():
    input = "3"
    out = utils.cast(input, int)
    assert out == 3

    input = "3.7"
    import pytest
    with pytest.raises(Exception):
        utils.cast(input, int)

    input = "unk"
    out = utils.cast(input, int)
    assert out == "unk"

