import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from mlpgad.models.mlpgad import MLPGAD
from mlpgad.train import score_nodes, train_one_class


def _toy_data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(60, 10, generator=g)
    ei = erdos_renyi_graph(60, 0.15)
    y = torch.zeros(60, dtype=torch.long)
    return Data(x=x, edge_index=ei, y=y)


def test_training_reduces_mean_residual():
    data = _toy_data()
    model = MLPGAD(in_dim=10, hid_dim=16, target_type="mlp_frozen")
    masked = torch.arange(60)
    before = model(data.x, data.edge_index, masked).mean().item()
    train_one_class(model, data, epochs=30, lr=1e-2, seed=0)
    after = model(data.x, data.edge_index, masked).mean().item()
    assert after < before


def test_score_nodes_returns_rank_in_unit_interval():
    data = _toy_data()
    model = MLPGAD(in_dim=10, hid_dim=16, target_type="mlp_frozen")
    train_one_class(model, data, epochs=5, lr=1e-2, seed=0)
    s = score_nodes(model, data, rounds=4, seed=0)
    assert s.shape == (60,)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0
