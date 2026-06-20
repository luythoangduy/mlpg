import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (from_scipy_sparse_matrix, remove_self_loops,
                                    to_undirected)

# Datasets shipped in the cloned UNPrompt repo (filenames are capitalized).
_MAT_FILES = {
    "disney": "Disney.mat",
    "reddit": "Reddit.mat",
    "amazon": "Amazon.mat",
    "facebook": "Facebook.mat",
}


def _first_key(mat, keys):
    for k in keys:
        if k in mat:
            return mat[k]
    raise KeyError("none of {} found in mat (have {})".format(
        keys, [k for k in mat if not k.startswith("__")]))


def load_mat(path: str) -> Data:
    """Load a DOMINANT/UNPrompt-schema .mat into PyG Data."""
    mat = sio.loadmat(path)
    adj = _first_key(mat, ["Network", "A", "adj"])
    attr = _first_key(mat, ["Attributes", "X", "attr"])
    label = _first_key(mat, ["Label", "gnd", "y", "ad_labels"])

    adj = sp.coo_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(adj)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)

    x = attr.todense() if sp.issparse(attr) else attr
    x = torch.tensor(np.asarray(x), dtype=torch.float)
    y = torch.tensor(np.asarray(label).ravel(), dtype=torch.long)
    y = (y != 0).long()
    return Data(x=x, edge_index=edge_index, y=y)


def load_inj_cora(root: str, seed: int = 0, n_struct_cliques: int = 15,
                  struct_clique_size: int = 15, n_context: int = 225,
                  context_sample: int = 50) -> Data:
    """Cora + standard structural (cliques) and contextual (feature swap) anomalies."""
    g = torch.Generator().manual_seed(seed)
    base = Planetoid(root=root, name="Cora")[0]
    x = base.x.clone().float()
    edge_index = to_undirected(remove_self_loops(base.edge_index)[0])
    num_nodes = x.shape[0]
    y = torch.zeros(num_nodes, dtype=torch.long)

    perm = torch.randperm(num_nodes, generator=g)
    n_struct = n_struct_cliques * struct_clique_size
    struct_nodes = perm[:n_struct]
    context_nodes = perm[n_struct:n_struct + n_context]

    # --- Structural anomalies: dense cliques among selected nodes ---
    new_edges = [edge_index]
    for c in range(n_struct_cliques):
        clique = struct_nodes[c * struct_clique_size:(c + 1) * struct_clique_size]
        if clique.numel() < 2:
            continue
        rows = clique.repeat_interleave(clique.numel())
        cols = clique.repeat(clique.numel())
        keep = rows != cols
        new_edges.append(torch.stack([rows[keep], cols[keep]], dim=0))
        y[clique] = 1
    edge_index = to_undirected(torch.cat(new_edges, dim=1))

    # --- Contextual anomalies: replace features with the farthest of a random sample ---
    for v in context_nodes.tolist():
        cand = torch.randint(0, num_nodes, (context_sample,), generator=g)
        dist = torch.norm(x[v].unsqueeze(0) - x[cand], dim=1)
        x[v] = x[cand[int(torch.argmax(dist))]]
        y[v] = 1

    return Data(x=x, edge_index=edge_index, y=y)


def load_dataset(name: str, *, unprompt_dir: str, cora_root: str,
                 books_path: str | None = None, seed: int = 0) -> Data:
    if name == "inj_cora":
        return load_inj_cora(cora_root, seed=seed)
    key = name.lower()
    if key in _MAT_FILES:
        return load_mat(os.path.join(unprompt_dir, _MAT_FILES[key]))
    if key == "books":
        if books_path is None:
            raise FileNotFoundError(
                "books.mat not provided; use an available in-repo dataset "
                "(disney, reddit, amazon, facebook).")
        return load_mat(books_path)
    raise ValueError("unknown dataset: {}".format(name))
