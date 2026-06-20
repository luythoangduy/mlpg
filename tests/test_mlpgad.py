import torch
from torch_geometric.utils import erdos_renyi_graph
from mlpgad.models.mlpgad import MLPGAD


def _toy(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(30, 8)
    edge_index = erdos_renyi_graph(30, 0.2)
    return x, edge_index


def test_forward_returns_residual_per_masked_node():
    x, ei = _toy()
    model = MLPGAD(in_dim=8, hid_dim=16, target_type="mlp_frozen")
    masked = torch.tensor([0, 5, 10])
    r = model(x, ei, masked)
    assert r.shape == (3,)
    assert torch.isfinite(r).all()
    assert float(r.min()) >= 0.0  # 1 - cos in [0, 2]


def test_mlp_frozen_target_has_no_grad_and_is_fixed():
    x, ei = _toy()
    model = MLPGAD(in_dim=8, hid_dim=16, target_type="mlp_frozen")
    t1 = model._target(x, ei).detach().clone()
    # an optimizer step on the predictive path must not change the target
    masked = torch.arange(30)
    loss = model(x, ei, masked).mean()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.data -= 0.1 * p.grad
    t2 = model._target(x, ei).detach()
    assert torch.allclose(t1, t2)


def test_gnn_ema_update_moves_target_toward_online():
    x, ei = _toy()
    model = MLPGAD(in_dim=8, hid_dim=16, target_type="gnn_ema",
                   ema_momentum=0.0)  # momentum 0 => target copies online
    # perturb online encoder
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.randn_like(p))
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target()
    after = list(model.target_encoder.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed
