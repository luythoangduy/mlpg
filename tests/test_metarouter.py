import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from mlpgad.metarouter import _family, meta_features


def _toy(n=50, f=8, seed=0):
    torch.manual_seed(seed)
    return Data(x=torch.randn(n, f), edge_index=erdos_renyi_graph(n, 0.15),
                y=torch.zeros(n, dtype=torch.long))


def test_meta_features_finite_and_keyed():
    mf = meta_features(_toy())
    assert len(mf) >= 12
    for k, v in mf.items():
        assert np.isfinite(v), k


def test_family_classification():
    assert _family("struct_negdeg") == "structure"
    assert _family("struct_spec_outlier") == "structure"
    assert _family("learn_dom_struct") == "structure"
    assert _family("feat_global") == "feature"
    assert _family("learn_dom_attr") == "feature"
