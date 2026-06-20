import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid
from torch_geometric.utils import (from_scipy_sparse_matrix, remove_self_loops,
                                    subgraph, to_undirected)

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


# Organic (non-injected) GADBench-scale datasets, loaded via torch_geometric downloads.
# Anomaly = the minority/positive class; all carry real (not synthetic) anomaly labels.
_HETEROPHILOUS = {"tolokers": "Tolokers", "questions": "Questions"}


def load_heterophilous(name: str, root: str) -> Data:
    """Tolokers / Questions from the heterophilous-graph benchmark as binary-label GAD.

    y is the native binary node label (Tolokers: banned workers ~22%; Questions: active
    users ~3%). Edges made undirected, self-loops removed.
    """
    ds = HeterophilousGraphDataset(root=root, name=_HETEROPHILOUS[name.lower()])
    g = ds[0]
    edge_index = to_undirected(remove_self_loops(g.edge_index)[0])
    y = (g.y.ravel() != 0).long()
    return Data(x=g.x.float(), edge_index=edge_index, y=y)


_ELLIPTIC_URL = "https://data.pyg.org/datasets/elliptic"
_ELLIPTIC_FILES = ["elliptic_txs_features.csv", "elliptic_txs_edgelist.csv",
                   "elliptic_txs_classes.csv"]


def load_elliptic(root: str) -> Data:
    """Elliptic bitcoin transaction graph; illicit=1 vs licit=0 on the *labeled* subgraph.

    Parses the raw CSVs directly with numpy (no pandas / pygod dependency), downloading
    them on first use. Unknown-class nodes (the majority) are dropped and the graph induced
    on labeled nodes, so every node carries a real fraud/non-fraud label for per-node scoring.
    The 165 transaction features are kept (timestep column dropped).
    """
    raw = os.path.join(root, "raw")
    feat_csv = os.path.join(raw, "elliptic_txs_features.csv")
    if not os.path.exists(feat_csv):
        from torch_geometric.data import download_url, extract_zip
        os.makedirs(raw, exist_ok=True)
        for f in _ELLIPTIC_FILES:
            path = download_url("%s/%s.zip" % (_ELLIPTIC_URL, f), raw)
            extract_zip(path, raw)

    feats = np.genfromtxt(feat_csv, delimiter=",", dtype=np.float64)
    tx_ids = feats[:, 0].astype(np.int64)
    x = feats[:, 2:]  # drop txId and timestep; keep the 165 features

    cls = np.genfromtxt(os.path.join(raw, "elliptic_txs_classes.csv"), delimiter=",",
                        dtype=str, skip_header=1)
    cls_map = {int(t): c for t, c in cls}
    id_to_idx = {int(t): i for i, t in enumerate(tx_ids)}

    # class "1" = illicit -> 1, "2" = licit -> 0, "unknown" -> drop
    y = np.full(len(tx_ids), -1, dtype=np.int64)
    for t, c in cls_map.items():
        if c == "1":
            y[id_to_idx[t]] = 1
        elif c == "2":
            y[id_to_idx[t]] = 0

    edges = np.genfromtxt(os.path.join(raw, "elliptic_txs_edgelist.csv"), delimiter=",",
                          dtype=np.int64, skip_header=1)
    e0 = np.array([id_to_idx[t] for t in edges[:, 0]])
    e1 = np.array([id_to_idx[t] for t in edges[:, 1]])
    edge_index = torch.tensor(np.stack([e0, e1]), dtype=torch.long)

    keep = torch.tensor(y >= 0)
    sub_ei, _ = subgraph(keep, edge_index, relabel_nodes=True, num_nodes=len(tx_ids))
    sub_ei = to_undirected(remove_self_loops(sub_ei)[0])
    x = torch.tensor(x[y >= 0], dtype=torch.float)
    return Data(x=x, edge_index=sub_ei, y=torch.tensor(y[y >= 0], dtype=torch.long))


def load_dataset(name: str, *, unprompt_dir: str, cora_root: str,
                 books_path: str | None = None, pyg_root: str | None = None,
                 seed: int = 0) -> Data:
    if name == "inj_cora":
        return load_inj_cora(cora_root, seed=seed)
    key = name.lower()
    if key in _MAT_FILES:
        return load_mat(os.path.join(unprompt_dir, _MAT_FILES[key]))
    if key in _HETEROPHILOUS:
        root = pyg_root or os.path.dirname(cora_root)
        return load_heterophilous(key, os.path.join(root, key))
    if key == "elliptic":
        root = pyg_root or os.path.dirname(cora_root)
        return load_elliptic(os.path.join(root, "elliptic"))
    if key == "books":
        if books_path is None:
            raise FileNotFoundError(
                "books.mat not provided; use an available in-repo dataset "
                "(disney, reddit, amazon, facebook).")
        return load_mat(books_path)
    raise ValueError("unknown dataset: {}".format(name))
