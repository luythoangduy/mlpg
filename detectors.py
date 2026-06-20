"""Stronger, training-free detector bank for few-shot channel identification.

Upgrades the trivial bank in `fewshot_channel.py` with stronger (but still CPU-cheap,
no gradient training) detectors spanning feature and structural anomaly hypotheses:

  feature:
    feat_pca   - PCA reconstruction error in normalized feature space (global outlierness)
    feat_lof   - Local Outlier Factor on normalized features (local density)
  structural (via spectral embedding U = top-k eigenvectors of normalized adjacency):
    struct_spec_nonsmooth - ||U_v - mean_neighbors(U)||   (structural non-smoothness)
    struct_spec_outlier   - Mahalanobis distance of U_v   (global structural outlierness)

`build_bank(data, level)` returns a dict of rank-normalized scores (higher = anomalous):
  level='trivial' -> the 5 hand-crafted channels; 'strong' -> the 4 above; 'all' -> both.

Run: python -m mlpgad.detectors
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from torch_geometric.utils import to_scipy_sparse_matrix

from mlpgad.data.loaders import load_dataset
from mlpgad.fewshot_channel import build_channels, fewshot_select

UN = r"D:\notes\graph_anomaly\UNPrompt\Datasets"
CR = r"D:\notes\graph_anomaly\data_cache\cora"
DATASETS = ["inj_cora", "disney", "reddit", "amazon", "facebook"]


def _z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def _rank(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def _csr(data):
    n = data.x.shape[0]
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return A


def _spectral_embedding(A, k=32):
    """Top-k eigenvectors of the symmetric normalized adjacency D^-1/2 A D^-1/2."""
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    Dinv = sp.diags(dinv)
    An = (Dinv @ A @ Dinv).tocsr()
    k = int(min(k, A.shape[0] - 2))
    # largest algebraic eigenvalues -> smoothest structural modes
    vals, vecs = spla.eigsh(An, k=k, which="LA")
    return vecs  # [N, k]


def _strong_channels(data):
    A = _csr(data)
    deg = np.asarray(A.sum(1)).ravel()
    dsafe = np.maximum(deg, 1)
    Xn = _z(data.x.numpy().astype(float))

    # --- feature: PCA reconstruction error ---
    c = int(min(32, Xn.shape[1] - 1, Xn.shape[0] - 1))
    c = max(1, c)
    pca = PCA(n_components=c, random_state=0)
    recon = pca.inverse_transform(pca.fit_transform(Xn))
    feat_pca = np.linalg.norm(Xn - recon, axis=1)

    # --- feature: Local Outlier Factor ---
    nn = int(min(20, Xn.shape[0] - 1))
    lof = LocalOutlierFactor(n_neighbors=max(2, nn))
    lof.fit(Xn)
    feat_lof = -lof.negative_outlier_factor_

    # --- structural: spectral embedding ---
    U = _spectral_embedding(A)
    nbr_U = (A @ U) / dsafe[:, None]
    struct_spec_nonsmooth = np.linalg.norm(U - nbr_U, axis=1)
    Uc = U - U.mean(0)
    cov = np.cov(Uc, rowvar=False) + 1e-6 * np.eye(U.shape[1])
    prec = np.linalg.pinv(cov)
    struct_spec_outlier = np.einsum("ij,jk,ik->i", Uc, prec, Uc)

    return {
        "feat_pca": feat_pca,
        "feat_lof": feat_lof,
        "struct_spec_nonsmooth": struct_spec_nonsmooth,
        "struct_spec_outlier": struct_spec_outlier,
    }


def build_bank(data, level="all"):
    bank = {}
    if level in ("trivial", "all"):
        bank.update(build_channels(data))  # already rank-normalized
    if level in ("strong", "all"):
        bank.update({k: _rank(v) for k, v in _strong_channels(data).items()})
    return bank


def _oracle(ranks, y):
    best = 0.5
    for rr in ranks.values():
        a = roc_auc_score(y, rr)
        best = max(best, a, 1.0 - a)
    return best


def _fewshot_auc(ranks, y, k, trials, rng):
    ano = np.where(y == 1)[0]
    if k >= len(ano):
        return None
    out = []
    for _ in range(trials):
        shots = rng.choice(ano, size=k, replace=False)
        name, sign = fewshot_select(ranks, shots)
        test = np.ones(len(y), dtype=bool)
        test[shots] = False
        out.append(roc_auc_score(y[test], (sign * ranks[name])[test]))
    return float(np.mean(out))


def run(k=5, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    print("%-9s | oracle(triv  strong) | few-shot k=%d (triv  strong) | strong pick"
          % ("dataset", k))
    for name in DATASETS:
        d = load_dataset(name, unprompt_dir=UN, cora_root=CR, seed=0)
        y = d.y.numpy().astype(int)
        triv = build_bank(d, "trivial")
        allb = build_bank(d, "all")
        otv, ost = _oracle(triv, y), _oracle(allb, y)
        ftv = _fewshot_auc(triv, y, k, trials, rng)
        fst = _fewshot_auc(allb, y, k, trials, rng)
        # which strong channel does few-shot favor (full-label proxy)
        best = max(((nm, max(roc_auc_score(y, rr), 1 - roc_auc_score(y, rr)))
                    for nm, rr in build_bank(d, "strong").items()),
                   key=lambda t: t[1])
        fmt = lambda v: "n/a " if v is None else "%.3f" % v
        print("%-9s |  %.3f  %.3f      |    %s  %s          | %s (%.3f)"
              % (name, otv, ost, fmt(ftv), fmt(fst), best[0], best[1]))


if __name__ == "__main__":
    run()
