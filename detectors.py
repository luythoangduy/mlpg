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

import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from torch_geometric.utils import to_scipy_sparse_matrix

from mlpgad.data.loaders import load_dataset
from mlpgad.fewshot_channel import build_channels, fewshot_select
from mlpgad.models.dominant import train_dominant

UN = r"D:\notes\graph_anomaly\UNPrompt\Datasets"
CR = r"D:\notes\graph_anomaly\data_cache\cora"
PYG = r"D:\notes\graph_anomaly\data_cache"
DATASETS = ["inj_cora", "disney", "reddit", "amazon", "facebook"]
_CACHE = os.path.join(os.path.dirname(__file__), "results", "learned_cache")


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


def learned_channels(data, name=None, seed=0, force_train=False, **kw):
    """Raw per-node learned (DOMINANT) recon errors, cached by (name, seed) when name given.

    Training is the slow part, so results are cached to results/learned_cache/<name>_<seed>.npz.
    Pass name=None (e.g. toy graphs / tests) to always train fresh without touching the cache.
    """
    if name is not None and not force_train:
        path = os.path.join(_CACHE, "%s_%d.npz" % (name, seed))
        if os.path.exists(path):
            d = np.load(path)
            return {k: d[k] for k in d.files}
    raw = train_dominant(data, seed=seed, **kw)
    if name is not None:
        os.makedirs(_CACHE, exist_ok=True)
        np.savez(os.path.join(_CACHE, "%s_%d.npz" % (name, seed)), **raw)
    return raw


def build_bank(data, level="all", name=None, seed=0, force_train=False):
    bank = {}
    if level in ("trivial", "all"):
        bank.update(build_channels(data))  # already rank-normalized
    if level in ("strong", "all"):
        bank.update({k: _rank(v) for k, v in _strong_channels(data).items()})
    if level in ("learned", "all"):
        raw = learned_channels(data, name=name, seed=seed, force_train=force_train)
        bank.update({k: _rank(v) for k, v in raw.items()})
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


def _load(name, seed=0):
    return load_dataset(name, unprompt_dir=UN, cora_root=CR, pyg_root=PYG, seed=seed)


def _best_in_group(data, level, y, name, seed):
    """Best oriented AUC and its channel within a bank level (full labels)."""
    best_nm, best_a = None, 0.5
    for nm, rr in build_bank(data, level, name=name, seed=seed).items():
        a = roc_auc_score(y, rr)
        a = max(a, 1.0 - a)
        if a > best_a:
            best_nm, best_a = nm, a
    return best_nm, best_a


def channel_table(datasets, seed=0):
    """Per-channel-group oracle: how the learned bank compares to trivial/strong."""
    print("\n== best oriented channel AUC per group (full labels) ==")
    print("%-10s | trivial        | strong         | learned        | oracle(all)"
          % "dataset")
    for name in datasets:
        d = _load(name, seed)
        y = d.y.numpy().astype(int)
        bt = _best_in_group(d, "trivial", y, name, seed)
        bs = _best_in_group(d, "strong", y, name, seed)
        bl = _best_in_group(d, "learned", y, name, seed)
        oall = _oracle(build_bank(d, "all", name=name, seed=seed), y)
        cell = lambda t: "%.3f %-9s" % (t[1], (t[0] or "-")[:9])
        print("%-10s | %s | %s | %s | %.3f"
              % (name, cell(bt), cell(bs), cell(bl), oall))


def fewshot_table(datasets, ks=(1, 3, 5, 10), trials=200, seed=0):
    """Few-shot selection over trivial vs full (trivial+strong+learned) bank."""
    rng = np.random.default_rng(seed)
    print("\n== few-shot channel selection (oriented AUC, %d draws) ==" % trials)
    print("%-10s | oracle(all) |" % "dataset" +
          "".join(" k=%-2d(triv/all)" % k for k in ks) + " | top pick (all,k=5)")
    for name in datasets:
        d = _load(name, seed)
        y = d.y.numpy().astype(int)
        triv = build_bank(d, "trivial", name=name, seed=seed)
        allb = build_bank(d, "all", name=name, seed=seed)
        line = "%-10s |    %.3f    |" % (name, _oracle(allb, y))
        picks = {}
        for k in ks:
            ft = _fewshot_auc(triv, y, k, trials, rng)
            fa = _fewshot_auc(allb, y, k, trials, rng)
            f = lambda v: "n/a " if v is None else "%.3f" % v
            line += "  %s/%s" % (f(ft), f(fa))
            if k == 5 and fa is not None:
                for _ in range(trials):
                    shots = rng.choice(np.where(y == 1)[0], size=k, replace=False)
                    cn, _s = fewshot_select(allb, shots)
                    picks[cn] = picks.get(cn, 0) + 1
        top = sorted(picks.items(), key=lambda x: -x[1])[:2]
        line += " | " + ", ".join("%s:%d%%" % (n, 100 * c // trials) for n, c in top)
        print(line)


def run(datasets=None, seed=0):
    datasets = datasets or DATASETS
    channel_table(datasets, seed=seed)
    fewshot_table(datasets, seed=seed)


if __name__ == "__main__":
    import sys
    ds = sys.argv[1:] if len(sys.argv) > 1 else None
    run(ds)
