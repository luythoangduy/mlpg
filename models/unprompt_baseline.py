import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, GraphSAGE

from mlpgad.normalize import coordinate_normalize

_BACKBONES = {"sage": GraphSAGE, "gcn": GCN}


def _mlp(in_dim, hid_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, out_dim))


class UNPromptBaseline(nn.Module):
    """Neighborhood-predictability of the raw normalized attribute (no EMA target)."""

    def __init__(self, in_dim, hid_dim=64, num_layers=2, backbone="sage"):
        super().__init__()
        backbone_cls = _BACKBONES[backbone]
        self.encoder = backbone_cls(in_channels=in_dim, hidden_channels=hid_dim,
                                    num_layers=num_layers, out_channels=hid_dim)
        self.predictor = _mlp(hid_dim, hid_dim, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x, edge_index, masked_nodes):
        x_norm = coordinate_normalize(x)
        x_ctx = x_norm.clone()
        x_ctx[masked_nodes] = self.mask_token  # center always masked
        z = self.encoder(x_ctx, edge_index)
        pred = self.predictor(z[masked_nodes])
        target = x_norm[masked_nodes].detach()
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 1.0 - (pred * target).sum(dim=-1)

    @torch.no_grad()
    def update_target(self):  # parity with MLPGAD trainer interface
        return
