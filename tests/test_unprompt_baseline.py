import torch
from torch_geometric.utils import erdos_renyi_graph
from mlpgad.models.unprompt_baseline import UNPromptBaseline


def test_forward_residual_shape_and_finite():
    torch.manual_seed(0)
    x = torch.randn(25, 6)
    ei = erdos_renyi_graph(25, 0.25)
    model = UNPromptBaseline(in_dim=6, hid_dim=16)
    masked = torch.tensor([1, 4, 9])
    r = model(x, ei, masked)
    assert r.shape == (3,)
    assert torch.isfinite(r).all()


def test_target_is_raw_attribute_not_learned_embedding():
    # predictor output dim must equal in_dim (predicting raw normalized attrs)
    model = UNPromptBaseline(in_dim=6, hid_dim=16)
    assert model.predictor[-1].out_features == 6
