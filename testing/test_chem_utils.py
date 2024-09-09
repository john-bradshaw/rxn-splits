"""
Tests the common chemical utils.
this uses precomputed canonicalizations from RDKit
rdkit                     2021.03.5        py39h88273a1_0    conda-forge
on MacOS
"""
import collections

import numpy as np

from rxn_splits import chem_utils


def test_clean_smiles_reaction():
    reaction_smi = ("[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1"
                    "[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1"
                    "[NH:15][CH3:14]")
    cleaned_reaction = chem_utils.clean_smiles_reaction(reaction_smi)

    expected_reactants = collections.Counter(["O=C(O)c1ccc(Cl)c([N+](=O)[O-])c1", "O", "CN"])
    expected_products = collections.Counter(["CNc1ccc(C(=O)O)cc1[N+](=O)[O-]"])
    assert cleaned_reaction == (frozenset(expected_reactants.items()), frozenset(expected_products.items()))


def test_canonicalize():
    smiles_in = "O[C:1](=[O:17])[c:3]1[cH:4][c:5]([N+:6](=[O:7])[O-:8])[c:9]([S:10][c:11]2[c:12]([Cl:13])[cH:14][n:15]" \
                   "[cH:16][c:2]2[Cl:18])[s:19]1"
    smiles_canon = chem_utils.canonicalize(smiles_in)
    expected_smi = "O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"
    assert smiles_canon == expected_smi

    smiles_canon2 = chem_utils.canonicalize(smiles_canon)
    assert smiles_canon2 == smiles_canon



def test_num_heavy_atoms():
    smi = ["O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1", 'CCC(=O)O']
    expected_num = 25
    calc_num = chem_utils.num_heavy_atoms(smi)
    assert calc_num == expected_num


def test_at_least_one_carbon():
    smi = ["O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1", 'CCC(=O)O']
    expected_answer = True
    calc_num = chem_utils.at_least_one_carbon(smi)
    assert calc_num == expected_answer


def test_at_least_one_carbon_false():
    smi = ["Cl", 'O=O']
    expected_answer = False
    calc_num = chem_utils.at_least_one_carbon(smi)
    assert calc_num == expected_answer


def test_max_number_of_bonds_per_molecule():
    smi = ["CC", "O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1", 'CCC(=O)O']
    expected_answer = 21
    calc_num = chem_utils.max_number_of_bonds_per_mol(smi)
    assert calc_num == expected_answer


def test_at_least_one_larger_product_different():
    reactant_fz = {('C1COCCO1', 1), ('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1), ('Cl', 2)}
    product_fz = {('C[C@H](N)c1cc2cc(Cl)ccc2[nH]c1=O', 1), ('Cl', 2)}
    expected_answer = True

    output = chem_utils.at_least_one_larger_product_different(reactant_fz, product_fz)
    assert output == expected_answer

    product_fz2 = {('C[C@H](N[S@](=O)C(C)(C)C)c1cc2cc(Cl)ccc2nc1Cl', 1)}
    expected_answer = False
    output = chem_utils.at_least_one_larger_product_different(reactant_fz, product_fz2)
    assert output == expected_answer

    product_fz3 = {('C', 2)}
    expected_answer = False
    output = chem_utils.at_least_one_larger_product_different(reactant_fz, product_fz3)
    assert output == expected_answer


def test_not_deprotonation():
    reactant_fz = {('CC(=O)O', 1), ('OCC', 1)}
    product_fz = {('CC(=O)OCC', 1)}
    expected_answer = True
    output = chem_utils.not_deprotonation(reactant_fz, product_fz)
    assert output == expected_answer

    product_fz2 = {('CC(=O)[O-]', 1)}
    expected_answer = False
    output = chem_utils.not_deprotonation(reactant_fz, product_fz2)
    assert output == expected_answer


def test_remove_isotope_info_from_mol_in_place():
    from rdkit import Chem
    smi = "[19F][13C@H]([16OH])[35Cl]"

    expected_mol = "F[C@H](O)Cl"
    mol = Chem.MolFromSmiles(smi)
    assert Chem.MolToSmiles(mol, canonical=False) == smi, "should not change before"

    chem_utils.remove_isotope_info_from_mol_in_place(mol)

    assert Chem.MolToSmiles(mol, canonical=False) == expected_mol, "should change after"


def test_try_to_canonicalize_fails_gracefully():
    incorrect_smi = "CC(C)(C)(C)(C)"
    expected_smi = incorrect_smi
    output = chem_utils.try_to_canonicalize(incorrect_smi)
    assert output == expected_smi


def test_smi_tokenizer():
    smi = "O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"
    tokens = chem_utils.smi_tokenizer(smi)

    assert ''.join(tokens.split(' ')) == smi  # test round trip
    assert tokens[:26] == "O = C ( O ) c 1 c c ( [N+]"  # test manually computed first 26 characters


def test_get_atom_map_nums():
    smi = "[C:14][C:12](C)([C:1])"
    atom_map = set(chem_utils.get_atom_map_nums(smi))
    assert atom_map == {1, 12, 14}


def test_get_fp_returns_array():
    import numpy as np
    smi = "O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"
    fp = chem_utils.get_fp(smi)
    assert isinstance(fp, np.ndarray)
    assert len(fp) == 2048
    assert fp.sum() > 0.5  # should have some non-zero values
    assert fp.sum() < len(fp)  # should have some zero values


def test_get_changed_bonds():
    rxn_smi = "[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]"
    changed_bonds = chem_utils.get_changed_bonds(rxn_smi)
    expected_bond_changes = set([(12, 13, 0.0), (12, 15, 1.0)])
    assert set(changed_bonds) == expected_bond_changes


def test_split_reagents_out_from_reactants_and_products():
    rxn_smi = "[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]"
    reactants, reagents, products = chem_utils.split_reagents_out_from_reactants_and_products(rxn_smi)

    reagents = chem_utils.canonicalize(reagents, remove_atm_mapping=True)
    assert reagents == "O"

    expected_products = chem_utils.canonicalize(products, remove_atm_mapping=True)
    products = chem_utils.canonicalize("[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]", remove_atm_mapping=True)
    assert expected_products == products

    reactants = chem_utils.canonicalize(reactants, remove_atm_mapping=True)
    react1 = chem_utils.canonicalize("[CH3:14][NH2:15]", remove_atm_mapping=True)
    react2 = chem_utils.canonicalize("[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13]", remove_atm_mapping=True)
    smi1 = f"{react1}.{react2}"
    smi2 = f"{react2}.{react1}"
    assert reactants == smi1 or reactants == smi2
