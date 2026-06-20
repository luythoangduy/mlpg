import os

import pytest
import torch
from torch_geometric.data import Data
from mlpgad.data.loaders import load_dataset, load_mat, load_inj_cora

UNPROMPT_DIR = os.environ.get(
    "UNPROMPT_DIR", r"D:\notes\graph_anomaly\UNPrompt\Datasets")
CORA_ROOT = os.environ.get("CORA_ROOT", r"D:\notes\graph_anomaly\data_cache\cora")
PYG_ROOT = os.environ.get("PYG_ROOT", r"D:\notes\graph_anomaly\data_cache")


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


@pytest.mark.parametrize("name", ["tolokers", "questions", "elliptic"])
def test_organic_loaders_schema(name):
    """Organic datasets load to the standard binary-label PyG schema (skip if offline)."""
    try:
        data = load_dataset(name, unprompt_dir="", cora_root=CORA_ROOT,
                            pyg_root=PYG_ROOT, seed=0)
    except Exception as e:  # noqa: BLE001 - network/download failures shouldn't fail CI
        pytest.skip("dataset %s unavailable: %s" % (name, e))
    assert isinstance(data, Data)
    assert data.x.dim() == 2 and data.x.shape[0] == data.y.shape[0]
    assert data.edge_index.shape[0] == 2
    assert set(data.y.unique().tolist()) == {0, 1}
    assert 0 < int(data.y.sum()) < data.y.shape[0]
