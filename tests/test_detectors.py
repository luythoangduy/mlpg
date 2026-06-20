import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from mlpgad.detectors import build_bank


def _toy(n=60, f=12, seed=0):
    torch.manual_seed(seed)
    return Data(x=torch.randn(n, f), edge_index=erdos_renyi_graph(n, 0.2),
                y=torch.zeros(n, dtype=torch.long))


def test_strong_bank_keys_and_ranks():
    d = _toy()
    bank = build_bank(d, level="strong")
    assert {"feat_pca", "feat_lof", "struct_spec_nonsmooth",
            "struct_spec_outlier"} == set(bank)
    for r in bank.values():
        assert r.shape == (60,)
        assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0


def test_all_bank_is_union_of_trivial_strong_learned():
    d = _toy()
    triv = build_bank(d, level="trivial")
    strong = build_bank(d, level="strong")
    learned = build_bank(d, level="learned")  # name=None -> trains fresh, no cache
    allb = build_bank(d, level="all")
    assert set(allb) == set(triv) | set(strong) | set(learned)
    assert len(allb) == len(triv) + len(strong) + len(learned)


def test_learned_bank_keys_and_ranks():
    d = _toy()
    learned = build_bank(d, level="learned")
    assert {"learn_dom_attr", "learn_dom_struct"} == set(learned)
    for r in learned.values():
        assert r.shape == (60,)
        assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0
