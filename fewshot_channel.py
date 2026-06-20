"""Few-shot channel identification.

Zero-shot channel routing is unsolvable (see ROUTER*.md / MERGE.md). This tests the
tractable alternative: with k labeled anomalies per graph, *select and orient* the best
channel from a small bank of label-free candidate detectors, then evaluate on the rest.

Channel bank (all higher = more anomalous before orientation):
  feat_nonsmooth : ||x_v - mean_neighbors(x)||           (feature deviation from neighbors)
  feat_global    : ||x_v||  in normalized space          (global feature outlierness)
  struct_negdeg  : -degree                               (low-degree anomaly)
  struct_posdeg  : +degree                               (high-degree / clique anomaly)
  struct_outlier : |z| of [degree, avg_neighbor_degree]  (degree-context outlierness)

Few-shot selection: rank-transform each channel; pick the channel whose k labeled anomalies
are most extreme (max |mean_rank - 0.5|); orient by the sign of that deviation. This both
chooses the right channel and fixes inversion.

Run: python -m mlpgad.fewshot_channel
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_scipy_sparse_matrix

from mlpgad.data.loaders import load_dataset

UN = r"D:\notes\graph_anomaly\UNPrompt\Datasets"
CR = r"D:\notes\graph_anomaly\data_cache\cora"
DATASETS = ["inj_cora", "disney", "reddit", "amazon", "facebook"]


def _z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def _rank(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def build_channels(data):
    n = data.x.shape[0]
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    dsafe = np.maximum(deg, 1)
    Xn = _z(data.x.numpy().astype(float))
    chans = {
        "feat_nonsmooth": np.linalg.norm(Xn - (A @ Xn) / dsafe[:, None], axis=1),
        "feat_global": np.linalg.norm(Xn, axis=1),
        "struct_negdeg": -deg.astype(float),
        "struct_posdeg": deg.astype(float),
        "struct_outlier": np.linalg.norm(
            _z(np.stack([deg, (A @ deg) / dsafe], axis=1)), axis=1),
    }
    return {k: _rank(v) for k, v in chans.items()}


def fewshot_select(ranks, shot_idx):
    """Pick (channel, orientation) maximizing extremeness of the shots."""
    best_name, best_info, best_sign = None, -1.0, 1.0
    for name, r in ranks.items():
        m = r[shot_idx].mean()
        info = abs(m - 0.5)
        if info > best_info:
            best_name, best_info, best_sign = name, info, (1.0 if m >= 0.5 else -1.0)
    return best_name, best_sign


def oracle_channel_auc(ranks, y):
    """Upper bound: best oriented channel AUC if labels were fully known."""
    best = 0.5
    for r in ranks.items():
        name, rr = r
        a = roc_auc_score(y, rr)
        best = max(best, a, 1.0 - a)
    return best


def run(ks=(1, 3, 5, 10), trials=200, seed=0):
    rng = np.random.default_rng(seed)
    print("%-9s | %-14s |" % ("dataset", "oracle/feat") +
          "".join(" k=%-2d " % k for k in ks) + "| top picks (k=5)")
    for name in DATASETS:
        d = load_dataset(name, unprompt_dir=UN, cora_root=CR, seed=0)
        y = d.y.numpy().astype(int)
        ranks = build_channels(d)
        ano = np.where(y == 1)[0]
        oracle = oracle_channel_auc(ranks, y)
        feat_fixed = roc_auc_score(y, ranks["feat_nonsmooth"])  # naive single-channel
        line = "%-9s | orc=%.3f f=%.3f |" % (name, oracle, feat_fixed)
        pick_counter = {}
        for k in ks:
            if k >= len(ano):
                line += " k=%-2d:  n/a " % k
                continue
            aucs = []
            for _ in range(trials):
                shots = rng.choice(ano, size=k, replace=False)
                cname, sign = fewshot_select(ranks, shots)
                test = np.ones(len(y), dtype=bool)
                test[shots] = False
                score = sign * ranks[cname]
                aucs.append(roc_auc_score(y[test], score[test]))
                if k == 5:
                    pick_counter[cname] = pick_counter.get(cname, 0) + 1
            line += " %.3f" % np.mean(aucs)
        top = sorted(pick_counter.items(), key=lambda x: -x[1])[:2]
        line += " | " + ", ".join("%s:%d%%" % (n, 100 * c // trials) for n, c in top)
        print(line)


if __name__ == "__main__":
    run()
