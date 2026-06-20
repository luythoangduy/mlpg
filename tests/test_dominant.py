import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from mlpgad.models.dominant import train_dominant


def _toy_with_anomalies(n=120, f=16, seed=0):
    """Planted feature anomalies: a few nodes get a large constant offset."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, f, generator=g)
    y = torch.zeros(n, dtype=torch.long)
    anom = torch.arange(0, 10)
    x[anom] += 8.0  # clearly out-of-distribution features
    y[anom] = 1
    ei = erdos_renyi_graph(n, 0.08)
    return Data(x=x, edge_index=ei, y=y)


def test_train_dominant_api_and_shapes():
    d = _toy_with_anomalies()
    raw = train_dominant(d, epochs=30, seed=0)
    assert set(raw) == {"learn_dom_attr", "learn_dom_struct"}
    for v in raw.values():
        assert v.shape == (d.x.shape[0],)
        assert np.isfinite(v).all()


def test_attr_recon_separates_planted_feature_anomalies():
    d = _toy_with_anomalies()
    raw = train_dominant(d, epochs=80, seed=0)
    attr = raw["learn_dom_attr"]
    anom = d.y.numpy() == 1
    # planted out-of-distribution nodes should reconstruct worse than normal ones
    assert attr[anom].mean() > attr[~anom].mean()


def test_neg_sampling_path_runs_for_large_n():
    d = _toy_with_anomalies(n=80)
    raw = train_dominant(d, epochs=10, neg_struct_threshold=10, seed=0)  # force neg path
    assert raw["learn_dom_struct"].shape == (80,)
    assert np.isfinite(raw["learn_dom_struct"]).all()
