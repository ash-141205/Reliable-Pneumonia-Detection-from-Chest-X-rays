from pneumreliab.data import collect, patient_id, patient_split


def test_patient_id():
    assert patient_id("person12_bacteria_45.jpeg") == "person12"
    assert patient_id("IM-0115-0001.jpeg") == "IM-0115"
    assert patient_id("NORMAL2-IM-1427-0001.jpeg") == "IM-1427"
    assert patient_id("abc.png") == "abc"


def test_no_patient_leakage_and_full_coverage(synthetic_root):
    items = collect(synthetic_root)
    sp = patient_split(items, seed=0)
    pats = {k: {x[2] for x in v} for k, v in sp.items()}
    assert not (pats["train"] & pats["val"]) and not (pats["train"] & pats["test"]) \
        and not (pats["val"] & pats["test"])
    assert sum(len(v) for v in sp.values()) == len(items)
    assert all(len(v) > 0 for v in sp.values())


def test_split_is_reproducible(synthetic_root):
    items = collect(synthetic_root)
    assert patient_split(items, 3) == patient_split(items, 3)
