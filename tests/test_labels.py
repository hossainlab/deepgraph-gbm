"""Label-rule correctness on synthetic spots."""

import pandas as pd
import pytest

from deepgraph_gbm.data.labels import NICHE_CLASSES, assign_niche_labels, encode_niche, mes_target


@pytest.fixture()
def rules():
    return {
        "niche_rules": {
            "necrotic": {"AF": ["PAN", "PNZ"]},
            "immune_hot": {"metaprogram": ["immune"], "not_AF": ["PAN", "PNZ"]},
            "normal": {"AF": ["LE_GM", "LE_WM"], "metaprogram": ["LE_GM", "LE_WM"]},
            "immune_cold": {"AF": ["CT", "IT", "IGN"], "metaprogram": ["CT", "IT", "hypoxic"]},
        },
        "priority": ["necrotic", "immune_hot", "normal", "immune_cold"],
        "mes_states": ["MES (malig.)", "MES.Hyp (malig.)", "MES.Ast (malig.)"],
    }


def _meta(rows):
    return pd.DataFrame(rows, columns=["AF", "metaprogram", "greenwald_metaprograms"])


def test_necrotic_priority_over_immune(rules):
    # PAN + immune metaprogram -> necrotic wins (priority order)
    m = _meta([["PAN", "immune", "MES.Hyp (malig.)"]])
    assert assign_niche_labels(m, rules).iloc[0] == "necrotic"


def test_immune_hot(rules):
    m = _meta([["IGN", "immune", "Mac"]])
    assert assign_niche_labels(m, rules).iloc[0] == "immune_hot"


def test_normal_requires_concordance(rules):
    concordant = _meta([["LE_WM", "LE_WM", "Oligo"]])
    discordant = _meta([["LE_WM", "CT", "AC (malig.)"]])
    assert assign_niche_labels(concordant, rules).iloc[0] == "normal"
    assert assign_niche_labels(discordant, rules).iloc[0] == "exclude"


def test_immune_cold(rules):
    m = _meta([["CT", "hypoxic", "MES (malig.)"]])
    assert assign_niche_labels(m, rules).iloc[0] == "immune_cold"


def test_vascular_excluded(rules):
    m = _meta([["MVP", "vascular", "Vasc"]])
    assert assign_niche_labels(m, rules).iloc[0] == "exclude"


def test_encode_niche(rules):
    m = _meta([
        ["PAN", "hypoxic", "MES.Hyp (malig.)"],
        ["IGN", "immune", "Mac"],
        ["MVP", "vascular", "Vasc"],
    ])
    enc = encode_niche(assign_niche_labels(m, rules))
    assert enc[0] == NICHE_CLASSES.index("necrotic")
    assert enc[1] == NICHE_CLASSES.index("immune_hot")
    assert enc[2] == -1


def test_mes_target(rules):
    m = _meta([
        ["CT", "CT", "MES (malig.)"],
        ["CT", "CT", "AC (malig.)"],
        ["CT", "CT", "nan"],
    ])
    t = mes_target(m, rules)
    assert t.iloc[0] == 1.0
    assert t.iloc[1] == 0.0
    assert pd.isna(t.iloc[2])
