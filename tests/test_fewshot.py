import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from mlpgad.fewshot_channel import build_channels, fewshot_select


def test_build_channels_returns_ranked_bank():
    torch.manual_seed(0)
    d = Data(x=torch.randn(40, 5), edge_index=erdos_renyi_graph(40, 0.2),
             y=torch.zeros(40, dtype=torch.long))
    ranks = build_channels(d)
    assert {"feat_nonsmooth", "feat_global", "struct_negdeg",
            "struct_posdeg", "struct_outlier"} <= set(ranks)
    for r in ranks.values():
        assert r.shape == (40,)
        assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0


def test_fewshot_selects_channel_where_shots_are_extreme():
    # channel "good" puts shots at the top; "noise" is uninformative for shots
    n = 100
    shots = np.array([0, 1, 2])
    good = np.zeros(n)
    good[shots] = 1.0  # shots have the highest raw values -> top rank
    noise = np.full(n, 0.5)
    ranks = {"good": np.argsort(np.argsort(good)) / (n - 1),
             "noise": np.argsort(np.argsort(noise)) / (n - 1)}
    name, sign = fewshot_select(ranks, shots)
    assert name == "good"
    assert sign == 1.0


def test_fewshot_flips_orientation_when_shots_rank_low():
    n = 100
    shots = np.array([0, 1, 2])
    inverted = np.zeros(n)
    inverted[shots] = 1.0
    # invert so shots sit at the BOTTOM of this channel's rank
    r = 1.0 - np.argsort(np.argsort(inverted)) / (n - 1)
    ranks = {"inv": r, "flat": np.full(n, 0.5)}
    name, sign = fewshot_select(ranks, shots)
    assert name == "inv"
    assert sign == -1.0  # selection detects inversion and flips
