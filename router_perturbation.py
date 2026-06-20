"""Perturbation-response channel router experiment.

Hypothesis: a channel's *competence* -- how well it detects its OWN injected anomaly
type against the graph's natural background -- predicts its *relevance* (real-anomaly AUC).
If injected signal is masked by natural variation (e.g. heavy-tailed degree), competence
drops and the channel should be down-weighted.

Two channels (label-free, higher = more anomalous):
  - struct: structural outlierness = L2 norm of z-scored [degree, avg_neighbor_degree].
  - feat:   1-hop feature non-smoothness ||x_v - mean_neighbors(x)|| in normalized space.

Run: python -m mlpgad.router_perturbation
"""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_scipy_sparse_matrix

from mlpgad.data.loaders import load_dataset

UN = r"D:\notes\graph_anomaly\UNPrompt\Datasets"
CR = r"D:\notes\graph_anomaly\data_cache\cora"
DATASETS = ["inj_cora", "disney", "reddit", "amazon", "facebook"]


def _z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def _adj_to_csr(edge_index, n):
    A = to_scipy_sparse_matrix(edge_index, num_nodes=n).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return A


def struct_score(A):
    deg = np.asarray(A.sum(1)).ravel()
    dsafe = np.maximum(deg, 1)
    avg_nbr = (A @ deg) / dsafe
    Z = _z(np.stack([deg, avg_nbr], axis=1))
    return np.linalg.norm(Z, axis=1)


def feat_score(A, Xn):
    deg = np.asarray(A.sum(1)).ravel()
    dsafe = np.maximum(deg, 1)[:, None]
    return np.linalg.norm(Xn - (A @ Xn) / dsafe, axis=1)


def inject_struct(A, frac, rng, add_edges=8):
    """Moderate structural perturbation: half densify, half sparsify."""
    n = A.shape[0]
    k = max(2, int(frac * n))
    sel = rng.permutation(n)[:k]
    dense, sparse = sel[: k // 2], sel[k // 2:]
    A = A.tolil()
    for v in dense:  # add a moderate number of random edges
        dsts = rng.integers(0, n, add_edges)
        for d in dsts:
            if d != v:
                A[v, d] = 1.0
                A[d, v] = 1.0
    for v in sparse:  # drop most incident edges
        A[v, :] = 0
        A[:, v] = 0
    A = A.tocsr()
    A.eliminate_zeros()
    labels = np.zeros(n, dtype=int)
    labels[sel] = 1
    return A, labels


def inject_feat(Xn, frac, rng):
    """Moderate feature perturbation: swap features with a random (not farthest) node."""
    n = Xn.shape[0]
    k = max(2, int(frac * n))
    sel = rng.permutation(n)[:k]
    src = rng.integers(0, n, k)
    Xp = Xn.copy()
    Xp[sel] = Xn[src]
    labels = np.zeros(n, dtype=int)
    labels[sel] = 1
    return Xp, labels


def rank(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def run():
    rng = np.random.default_rng(0)
    header = ("%-9s | real_AUC          | competence       | "
              "routed  oracle  naive  softcomp")
    print(header)
    print("%-9s | struct  feat     | struct  feat     |"
          % ("", ))
    rows = []
    for name in DATASETS:
        d = load_dataset(name, unprompt_dir=UN, cora_root=CR, seed=0)
        n = d.x.shape[0]
        y = d.y.numpy().astype(int)
        A = _adj_to_csr(d.edge_index, n)
        Xn = _z(d.x.numpy().astype(float))

        s_real = struct_score(A)
        f_real = feat_score(A, Xn)
        auc_s = roc_auc_score(y, s_real)
        auc_f = roc_auc_score(y, f_real)

        # competence: detect own injected type against natural background
        A_s, lab_s = inject_struct(A, 0.05, rng)
        comp_s = roc_auc_score(lab_s, struct_score(A_s))
        Xp_f, lab_f = inject_feat(Xn, 0.05, rng)
        comp_f = roc_auc_score(lab_f, feat_score(A, Xp_f))

        # routers (evaluated on REAL labels)
        pick = "struct" if comp_s >= comp_f else "feat"
        routed = auc_s if pick == "struct" else auc_f
        oracle = max(auc_s, auc_f)
        naive = roc_auc_score(y, rank(s_real) + rank(f_real))
        ws = max(0.0, comp_s - 0.5)
        wf = max(0.0, comp_f - 0.5)
        if ws + wf == 0:
            ws = wf = 1.0
        softcomp = roc_auc_score(y, ws * rank(s_real) + wf * rank(f_real))

        print("%-9s | %.3f  %.3f    | %.3f  %.3f    | %.3f   %.3f   %.3f  %.3f  [pick=%s]"
              % (name, auc_s, auc_f, comp_s, comp_f, routed, oracle, naive,
                 softcomp, pick))
        rows.append((name, auc_s, auc_f, comp_s, comp_f, routed, oracle, naive,
                     softcomp))

    # summary: does competence pick the better channel?
    correct = sum(1 for r in rows
                  if (r[3] >= r[4]) == (r[1] >= r[2]))
    print("\ncompetence picks better channel on %d/%d datasets" % (correct, len(rows)))
    mean_routed = np.mean([r[5] for r in rows])
    mean_oracle = np.mean([r[6] for r in rows])
    mean_naive = np.mean([r[7] for r in rows])
    mean_soft = np.mean([r[8] for r in rows])
    print("mean  routed=%.3f  oracle=%.3f  naive=%.3f  softcomp=%.3f"
          % (mean_routed, mean_oracle, mean_naive, mean_soft))
    return rows


if __name__ == "__main__":
    run()
