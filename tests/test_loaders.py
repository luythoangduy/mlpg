import os

import torch
from torch_geometric.data import Data
from mlpgad.data.loaders import load_mat, load_inj_cora

UNPROMPT_DIR = os.environ.get(
    "UNPROMPT_DIR", r"D:\notes\graph_anomaly\UNPrompt\Datasets")
CORA_ROOT = os.environ.get("CORA_ROOT", r"D:\notes\graph_anomaly\data_cache\cora")


def test_load_mat_disney_shape():
    path = os.path.join(UNPROMPT_DIR, "Disney.mat")
    data = load_mat(path)
    assert isinstance(data, Data)
    assert data.x.dim() == 2 and data.x.shape[0] == data.y.shape[0]
    assert data.edge_index.shape[0] == 2
    assert set(data.y.unique().tolist()).issubset({0, 1})
    assert data.y.sum() > 0  # disney has anomalies


def test_inj_cora_has_injected_anomalies():
    data = load_inj_cora(CORA_ROOT, seed=0)
    assert isinstance(data, Data)
    assert data.y.sum() > 0
    assert data.x.shape[0] == data.y.shape[0]
    # injection target ~ 5% of Cora's 2708 nodes
    assert 100 <= int(data.y.sum()) <= 600
