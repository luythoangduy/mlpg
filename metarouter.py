"""Cross-graph orientation/channel router via leave-one-graph-out (LOGO) meta-learning.

`ROUTER.md` / `ROUTER_PERTURBATION.md` showed *zero-shot, single-graph* channel routing
fails on hand-crafted statistics, and `ORIENTATION.md` showed the binding constraint is the
per-graph *sign* (the same channel inverts across graphs). This asks the natural follow-up:
given a *corpus* of graphs, can a supervised meta-learner predict the right channel and its
orientation for a held-out graph from label-free graph meta-features alone?

This is a deliberately small **probe** (n = 8 real graphs, LOGO), so the honest floor is a
**majority baseline** (predict the most common answer among the 7 training graphs). If a
1-NN meta-learner over meta-features cannot beat majority, the meta-features carry no
transferable signal -- confirming that anomaly *type/orientation* is unidentifiable from graph
structure, which is exactly why few-shot labels are needed.

Run: python -m mlpgad.metarouter
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_scipy_sparse_matrix

from mlpgad.detectors import _fewshot_auc, _load, build_bank

DATASETS = ["inj_cora", "disney", "reddit", "amazon", "facebook",
            "tolokers", "questions", "elliptic"]

# channels whose per-graph orientation we probe (defined on every graph)
SIGN_CHANNELS = ["struct_negdeg", "struct_posdeg", "feat_nonsmooth", "feat_global",
                 "learn_dom_attr"]
# canonical representative per family for the downstream router
CANON = {"structure": "struct_outlier", "feature": "feat_nonsmooth"}


def _z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def _family(name):
    return "structure" if name.startswith("struct") or name == "learn_dom_struct" \
        else "feature"


def meta_features(data):
    """Label-free graph descriptors (no use of y)."""
    n = data.x.shape[0]
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    dsafe = np.maximum(deg, 1)
    Xn = _z(data.x.numpy().astype(float))
    fns = np.linalg.norm(Xn - (A @ Xn) / dsafe[:, None], axis=1)  # feat non-smoothness
    fgl = np.linalg.norm(Xn, axis=1)                              # feat global norm

    # degree assortativity (Pearson of endpoint degrees over edges)
    ei = data.edge_index.numpy()
    du, dv = deg[ei[0]], deg[ei[1]]
    assort = np.corrcoef(du, dv)[0, 1] if len(du) > 1 and du.std() > 0 else 0.0

    def tailgap(v):
        q = np.quantile(v, [0.5, 0.99])
        rng = v.max() - v.min() + 1e-12
        return (q[1] - q[0]) / rng

    return {
        "log_n": np.log10(n),
        "log_e": np.log10(A.nnz / 2 + 1),
        "density": A.nnz / (n * (n - 1) + 1e-12),
        "mean_deg": deg.mean(),
        "deg_cv": deg.std() / (deg.mean() + 1e-12),
        "deg_skew": float(skew(deg)),
        "deg_kurt": float(kurtosis(deg)),
        "deg_assort": float(assort),
        "deg_tailgap": tailgap(deg),
        "log_featdim": np.log10(data.x.shape[1]),
        "fns_skew": float(skew(fns)),
        "fns_kurt": float(kurtosis(fns)),
        "fns_tailgap": tailgap(fns),
        "fgl_skew": float(skew(fgl)),
        "fgl_kurt": float(kurtosis(fgl)),
    }


def _channel_aucs(ranks, y):
    return {nm: roc_auc_score(y, r) for nm, r in ranks.items()}


def _collect():
    """Per graph: meta-features, oracle (channel, family, sign), per-channel signs, ranks/y."""
    rows = []
    for name in DATASETS:
        d = _load(name)
        y = d.y.numpy().astype(int)
        ranks = build_bank(d, "all", name=name)
        aucs = _channel_aucs(ranks, y)
        # oracle channel = best oriented
        o_name = max(aucs, key=lambda k: max(aucs[k], 1 - aucs[k]))
        o_sign = 1.0 if aucs[o_name] >= 0.5 else -1.0
        signs = {c: (1.0 if aucs[c] >= 0.5 else -1.0) for c in SIGN_CHANNELS}
        rows.append(dict(name=name, mf=meta_features(d), y=y, ranks=ranks,
                         aucs=aucs, o_name=o_name, o_family=_family(o_name),
                         o_sign=o_sign, signs=signs))
    return rows


def _matrix(rows):
    keys = sorted(rows[0]["mf"])
    M = np.array([[r["mf"][k] for k in keys] for r in rows])
    return M, keys


def _logo_1nn(M, i):
    """Index of nearest training graph to held-out i (standardized on the 7 train graphs)."""
    train = [j for j in range(len(M)) if j != i]
    sc = StandardScaler().fit(M[train])
    Mt = sc.transform(M)
    d = np.linalg.norm(Mt[train] - Mt[i], axis=1)
    return train[int(np.argmin(d))]


def run():
    rows = _collect()
    M, _ = _matrix(rows)
    n = len(rows)

    # ---- (A) orientation predictability: per-channel sign, 1-NN vs majority ----
    print("== (A) orientation prediction accuracy (LOGO, n=%d) ==" % n)
    print("%-16s | 1-NN meta | majority | trivial(always +)" % "channel")
    for c in SIGN_CHANNELS:
        true = np.array([r["signs"][c] for r in rows])
        nn_hit = maj_hit = pos_hit = 0
        for i in range(n):
            j = _logo_1nn(M, i)
            train = [t for t in range(n) if t != i]
            maj = 1.0 if np.mean([rows[t]["signs"][c] for t in train]) >= 0 else -1.0
            nn_hit += rows[j]["signs"][c] == true[i]
            maj_hit += maj == true[i]
            pos_hit += 1.0 == true[i]
        print("%-16s |   %d/%d     |   %d/%d    |   %d/%d"
              % (c, nn_hit, n, maj_hit, n, pos_hit, n))

    # ---- (B) channel-family predictability ----
    print("\n== (B) oracle channel-family prediction (LOGO) ==")
    fam_nn = fam_maj = 0
    for i in range(n):
        j = _logo_1nn(M, i)
        train = [t for t in range(n) if t != i]
        fams = [rows[t]["o_family"] for t in train]
        maj = max(set(fams), key=fams.count)
        fam_nn += rows[j]["o_family"] == rows[i]["o_family"]
        fam_maj += maj == rows[i]["o_family"]
    print("1-NN meta: %d/%d   majority: %d/%d" % (fam_nn, n, fam_maj, n))

    # ---- (C) downstream zero-shot AUC vs few-shot / oracle ----
    print("\n== (C) downstream AUC: zero-shot meta-router vs few-shot vs oracle ==")
    print("%-10s | naive | majority | 1NN-meta | few-shot k=5 | oracle" % "dataset")
    agg = {k: [] for k in ("naive", "majority", "nn", "fs5", "oracle")}
    for i in range(n):
        r = rows[i]
        y, ranks, aucs = r["y"], r["ranks"], r["aucs"]
        train = [t for t in range(n) if t != i]
        # naive: feat_nonsmooth, +1
        a_naive = roc_auc_score(y, ranks["feat_nonsmooth"])
        # majority: most common oracle (channel, sign) among training graphs
        votes = {}
        for t in train:
            key = (rows[t]["o_name"], rows[t]["o_sign"])
            votes[key] = votes.get(key, 0) + 1
        mname, msign = max(votes, key=votes.get)
        a_maj = roc_auc_score(y, msign * ranks[mname])
        # 1-NN meta: nearest training graph's oracle (channel, sign)
        j = _logo_1nn(M, i)
        a_nn = roc_auc_score(y, rows[j]["o_sign"] * ranks[rows[j]["o_name"]])
        # few-shot k=5 (uses 5 labels on held-out; reference, not zero-shot)
        rng = np.random.default_rng(0)
        a_fs = _fewshot_auc(ranks, y, 5, 200, rng)
        # oracle ceiling
        a_or = max(aucs[r["o_name"]], 1 - aucs[r["o_name"]])
        for k, v in zip(agg, (a_naive, a_maj, a_nn, a_fs, a_or)):
            agg[k].append(v if v is not None else np.nan)
        f = lambda v: "n/a " if v is None else "%.3f" % v
        print("%-10s | %s | %s   | %s  |   %s     | %s"
              % (r["name"], f(a_naive), f(a_maj), f(a_nn), f(a_fs), f(a_or)))
    print("%-10s | %.3f | %.3f   | %.3f  |   %.3f     | %.3f" % (
        "MEAN", *[np.nanmean(agg[k]) for k in ("naive", "majority", "nn", "fs5", "oracle")]))


if __name__ == "__main__":
    run()
