import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, GraphSAGE

from mlpgad.normalize import coordinate_normalize

_BACKBONES = {"sage": GraphSAGE, "gcn": GCN}


def _mlp(in_dim, hid_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, out_dim))


class MLPGAD(nn.Module):
    def __init__(self, in_dim, hid_dim=64, num_layers=2, backbone="sage",
                 target_type="mlp_frozen", mask=True, ema_momentum=0.99):
        super().__init__()
        if target_type not in ("mlp_frozen", "gnn_ema"):
            raise ValueError("target_type must be 'mlp_frozen' or 'gnn_ema'")
        self.target_type = target_type
        self.mask = mask
        self.ema_momentum = ema_momentum

        backbone_cls = _BACKBONES[backbone]
        self.encoder = backbone_cls(in_channels=in_dim, hidden_channels=hid_dim,
                                    num_layers=num_layers, out_channels=hid_dim)
        self.predictor = _mlp(hid_dim, hid_dim, hid_dim)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

        if target_type == "mlp_frozen":
            self.target_mlp = _mlp(in_dim, hid_dim, hid_dim)
            for p in self.target_mlp.parameters():
                p.requires_grad = False
        else:  # gnn_ema
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False

    def _target(self, x_norm, edge_index):
        if self.target_type == "mlp_frozen":
            return self.target_mlp(x_norm)
        return self.target_encoder(x_norm, edge_index)

    def forward(self, x, edge_index, masked_nodes):
        x_norm = coordinate_normalize(x)
        x_ctx = x_norm.clone()
        if self.mask:
            x_ctx[masked_nodes] = self.mask_token
        z = self.encoder(x_ctx, edge_index)
        h = self.predictor(z[masked_nodes])
        with torch.no_grad():
            t = self._target(x_norm, edge_index)[masked_nodes]
        h = F.normalize(h, dim=-1)
        t = F.normalize(t, dim=-1)
        return 1.0 - (h * t).sum(dim=-1)

    @torch.no_grad()
    def update_target(self):
        if self.target_type != "gnn_ema":
            return
        m = self.ema_momentum
        for o, t in zip(self.encoder.parameters(),
                        self.target_encoder.parameters()):
            t.data.mul_(m).add_(o.data, alpha=1.0 - m)
