"""DOMINANT-style reconstruction detector (pure torch_geometric, CPU-cheap).

A learned, training-based detector for the channel bank. A shared GCN encoder feeds two
decoders that reconstruct the two anomaly channels separately:

  attribute decoder  -> X_hat ;  attr recon error  ||Xn_v - X_hat_v||      (feature channel)
  structure decoder  -> A_hat = sigmoid(Z Z^T) ;  struct recon error       (structure channel)

This mirrors Ding et al. 2019 "DOMINANT". The two per-node error vectors plug into the
few-shot channel-selection bank as `learn_dom_attr` / `learn_dom_struct`, giving the bank a
genuine *learned* hypothesis instead of only hand-crafted statistics.

Scalability: the dense structure decoder is O(N^2). For N <= `neg_struct_threshold` we use
the canonical dense row-reconstruction loss/score; above it we switch to a negative-sampling
approximation (BCE over positive edges + sampled non-edges) so 48k/200k-node organic graphs
stay trainable on CPU. The negative-sampling score is a sampled estimate of the same per-node
adjacency-row reconstruction error.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN
from torch_geometric.utils import negative_sampling

from mlpgad.normalize import coordinate_normalize


class Dominant(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = GCN(in_channels=in_dim, hidden_channels=hid_dim,
                           num_layers=num_layers, out_channels=hid_dim, dropout=dropout)
        self.attr_decoder = GCN(in_channels=hid_dim, hidden_channels=hid_dim,
                                num_layers=num_layers, out_channels=in_dim, dropout=dropout)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode_attr(self, z, edge_index):
        return self.attr_decoder(z, edge_index)


def _edge_logits(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def train_dominant(data, hid_dim=64, num_layers=2, epochs=100, lr=5e-3, dropout=0.0,
                   weight_decay=5e-4, alpha=0.5, neg_struct_threshold=5000, device="cpu",
                   seed=0, verbose=False):
    """Train a DOMINANT autoencoder; return per-node {learn_dom_attr, learn_dom_struct}.

    `alpha` weights attribute vs structure reconstruction in the training loss. The returned
    arrays are raw (un-ranked) reconstruction errors, higher = more anomalous.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    x = coordinate_normalize(data.x.float()).to(dev)
    edge_index = data.edge_index.to(dev)
    n = x.shape[0]
    dense = n <= neg_struct_threshold

    model = Dominant(x.shape[1], hid_dim, num_layers, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if dense:
        A = torch.zeros(n, n, device=dev)
        A[edge_index[0], edge_index[1]] = 1.0

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        z = model.encode(x, edge_index)
        x_hat = model.decode_attr(z, edge_index)
        attr_loss = F.mse_loss(x_hat, x)
        if dense:
            a_hat = torch.sigmoid(z @ z.t())
            struct_loss = F.mse_loss(a_hat, A)
        else:
            neg = negative_sampling(edge_index, num_nodes=n,
                                    num_neg_samples=edge_index.shape[1])
            pos_logit = _edge_logits(z, edge_index)
            neg_logit = _edge_logits(z, neg)
            logits = torch.cat([pos_logit, neg_logit])
            targets = torch.cat([torch.ones_like(pos_logit),
                                 torch.zeros_like(neg_logit)])
            struct_loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = alpha * attr_loss + (1.0 - alpha) * struct_loss
        loss.backward()
        opt.step()
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print("  ep %3d  attr %.4f  struct %.4f" % (ep, attr_loss.item(),
                                                        struct_loss.item()))

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
        x_hat = model.decode_attr(z, edge_index)
        attr_err = torch.norm(x - x_hat, dim=1).cpu().numpy()
        if dense:
            a_hat = torch.sigmoid(z @ z.t())
            struct_err = torch.norm(A - a_hat, dim=1).cpu().numpy()
        else:
            struct_err = _neg_sample_struct_err(z, edge_index, n, seed)

    return {
        "learn_dom_attr": attr_err.astype(float),
        "learn_dom_struct": struct_err.astype(float),
    }


def _neg_sample_struct_err(z, edge_index, n, seed, neg_per_node=20):
    """Sampled per-node adjacency-row reconstruction error for large graphs.

    For node v: sum of (1 - A_hat)^2 over its positive edges + (A_hat)^2 over `neg_per_node`
    sampled non-neighbors, then sqrt. A sampled estimate of the dense row norm ||A_v - A_hat_v||.
    """
    g = torch.Generator(device=z.device).manual_seed(seed)
    err = torch.zeros(n, device=z.device)
    # positive contribution: (1 - sigmoid(z_u . z_v))^2 accumulated onto both endpoints
    p = torch.sigmoid(_edge_logits(z, edge_index))
    pos_term = (1.0 - p) ** 2
    err.index_add_(0, edge_index[0], pos_term)
    # negative contribution: sample neg_per_node non-neighbors per node
    src = torch.arange(n, device=z.device).repeat_interleave(neg_per_node)
    dst = torch.randint(0, n, (n * neg_per_node,), generator=g, device=z.device)
    pn = torch.sigmoid((z[src] * z[dst]).sum(dim=-1))
    err.index_add_(0, src, pn ** 2)
    return torch.sqrt(err).cpu().numpy()
