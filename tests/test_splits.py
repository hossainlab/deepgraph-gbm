"""Patient-level split disjointness and FFPE balance."""

import json
from pathlib import Path

import pytest

SPLITS = Path(__file__).resolve().parents[1] / "configs" / "splits.json"


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.json not generated yet")
def test_patient_disjointness():
    sp = json.loads(SPLITS.read_text())
    train, val, test = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.json not generated yet")
def test_all_patients_covered():
    sp = json.loads(SPLITS.read_text())
    all_p = set(sp["train"]) | set(sp["val"]) | set(sp["test"])
    assert len(all_p) == len(sp["all_patients"])
    assert set(sp["all_patients"]) == all_p
