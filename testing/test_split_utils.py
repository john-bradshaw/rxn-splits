
from rxn_splits import split_utils


def test_create_is_of_set_function():
    namerxns = ["3.1", "3.1.2", "6.3.4.2", "2.1", "2.1.3"]
    is_of_set = split_utils.create_is_of_set_function(namerxns)

    assert is_of_set("3.1")
    assert is_of_set("3.1.2")
    assert is_of_set("6.3.4.2")
    assert is_of_set("2.1")
    assert is_of_set("2.1.3")

    assert is_of_set("3.1.4")  # not in set explicitly but covered by superclass "3.1
    assert is_of_set("2.1.127.21342")  # not in set explicitly but covered by superclass "3.1

    assert not is_of_set("3.2")
    assert not is_of_set("2.2")
    assert not is_of_set("6.3.4.1")
    assert not is_of_set("5")


def test_check_no_overlap():
    list1 = ["one", "two", "three"]
    list2 = ["four", "five", "six"]
    list3 = ["seven", "eight", "seven"] # overlap within set is fine.
    assert split_utils.check_no_overlap(list1, list2, list3) is None

    list3[-1] = "six"
    import pytest
    with pytest.raises(AssertionError):
        split_utils.check_no_overlap(list1, list2, list3)

