"""Orientation vs selection, and selection vs a learned combination.

Two experiments over the full channel bank (trivial + strong + learned), reusing the
few-shot setup of `fewshot_channel.py` / `detectors.py`:

  Exp A -- decompose the few-shot gain into *channel selection* and *orientation* (sign).
    For each draw of k labeled anomalies we cross {selection: few-shot | oracle} with
    {orientation: fixed +1 | few-shot | oracle}. The fixed-+1 column is the naive
    "high score = anomaly" assumption; the gap it leaves to the oracle isolates how much
    of the win is *fixing the sign* (orientation) vs *picking the channel* (selection).
    A signed-separation table then shows *which* channels invert on *which* graphs.

  Exp C -- with the same k labels, compare picking one channel+sign against a learned
    logistic combination over the whole bank (PU-style: k labelled anomalies + sampled
    pseudo-normals). Tests whether selecting a single channel is enough or a weighted mix
    helps, and how each behaves as k grows.

Run: python -m mlpgad.orientation [dataset ...]
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlpgad.detectors import _load, build_bank
from mlpgad.fewshot_channel import fewshot_select

DATASETS = ["inj_cora", "disney", "reddit", "amazon", "facebook",
            "tolokers", "questions", "elliptic"]


def _oracle_pick(ranks, y):
    """Best oriented channel under full labels: (name, sign)."""
    best_name, best_a, best_sign = None, -1.0, 1.0
    for name, r in ranks.items():
        a = roc_auc_score(y, r)
        oriented, sign = (a, 1.0) if a >= 0.5 else (1.0 - a, -1.0)
        if oriented > best_a:
            best_name, best_a, best_sign = name, oriented, sign
    return best_name, best_sign


def _oracle_sign(rank, y):
    return 1.0 if roc_auc_score(y, rank) >= 0.5 else -1.0


def _fewshot_sign(rank, shots):
    return 1.0 if rank[shots].mean() >= 0.5 else -1.0


def _auc(rank, sign, y, test):
    return roc_auc_score(y[test], (sign * rank)[test])


# --------------------------------------------------------------------------- Exp A

def decompose(datasets, ks=(1, 3, 5, 10), trials=200, seed=0):
    """Cross selection x orientation; report 6 cells + orientation/selection shares.

    Path decomposition along naive -> few/few -> oracle (the actual method's trajectory):
      orient   = (few/few - naive)  / (oracle - naive)   -- gain from few-shot fixing the SIGN
      selResid = (oracle  - few/few)/ (oracle - naive)   -- residual: few-shot's channel error
    These sum to 100%. The naive corner is (few-shot selection, fixed +1 sign); few/few adds
    few-shot orientation only (same selection). Columns few/oraOri and oraSel/+1 are shown as
    references (oracle sign on the few-shot channel; oracle channel with the naive sign).
    """
    print("\n== Exp A: orientation vs selection (oriented ROC-AUC, %d draws) ==" % trials)
    print("cells: naive(+1) | few/few | few/oraOri | oraSel/+1 | oracle   "
          "[%% of gap: orient / selResid]")
    for name in datasets:
        d = _load(name)
        y = d.y.numpy().astype(int)
        ranks = build_bank(d, "all", name=name)
        ano = np.where(y == 1)[0]
        o_name, o_sign = _oracle_pick(ranks, y)
        print("%-10s" % name)
        for k in ks:
            if k >= len(ano):
                print("   k=%-2d  n/a" % k)
                continue
            rng = np.random.default_rng(seed + k)
            acc = {c: [] for c in
                   ("naive", "few_few", "few_ora", "ora_naive", "oracle")}
            for _ in range(trials):
                shots = rng.choice(ano, size=k, replace=False)
                test = np.ones(len(y), dtype=bool)
                test[shots] = False
                fs_name, fs_sign = fewshot_select(ranks, shots)
                acc["naive"].append(_auc(ranks[fs_name], 1.0, y, test))
                acc["few_few"].append(_auc(ranks[fs_name], fs_sign, y, test))
                acc["few_ora"].append(
                    _auc(ranks[fs_name], _oracle_sign(ranks[fs_name], y), y, test))
                acc["ora_naive"].append(_auc(ranks[o_name], 1.0, y, test))
                acc["oracle"].append(_auc(ranks[o_name], o_sign, y, test))
            m = {c: float(np.mean(v)) for c, v in acc.items()}
            gap = m["oracle"] - m["naive"]
            orient = _safe_share(m["few_few"] - m["naive"], gap)
            select = _safe_share(m["oracle"] - m["few_few"], gap)
            print("   k=%-2d  %.3f | %.3f | %.3f | %.3f | %.3f   [orient %s / select %s]"
                  % (k, m["naive"], m["few_few"], m["few_ora"], m["ora_naive"],
                     m["oracle"], orient, select))


def _safe_share(num, denom):
    if abs(denom) < 1e-6:
        return " n/a"
    return "%3d%%" % int(round(100 * num / denom))


def inversion_table(datasets):
    """Signed anomaly separation (mean_rank(anomalies) - 0.5) per channel per dataset.

    Negative => anomalies sit at the LOW end of that channel => the channel inverts and
    must be flipped. Shows the per-graph sign crossover directly.
    """
    print("\n== inversion diagnostic: signed separation mean_rank(anom)-0.5 ==")
    chans = ["struct_negdeg", "struct_posdeg", "feat_nonsmooth", "feat_global",
             "learn_dom_attr", "learn_dom_struct"]
    print("%-10s | " % "dataset" + " | ".join("%-13s" % c[:13] for c in chans))
    for name in datasets:
        d = _load(name)
        y = d.y.numpy().astype(int)
        ranks = build_bank(d, "all", name=name)
        ano = y == 1
        cells = []
        for c in chans:
            s = ranks[c][ano].mean() - 0.5
            cells.append("%+0.3f" % s)
        print("%-10s | " % name + " | ".join("%-13s" % v for v in cells))


# --------------------------------------------------------------------------- Exp C

def _feature_matrix(ranks):
    names = sorted(ranks)
    F = np.stack([ranks[n] for n in names], axis=1)
    return F, names


def selection_vs_logistic(datasets, ks=(1, 3, 5, 10), trials=200, seed=0, neg_mult=10):
    """Selection (1 channel+sign) vs logistic over the bank, same k labels (PU setup)."""
    print("\n== Exp C: selection vs logistic-over-bank (ROC-AUC, %d draws) ==" % trials)
    print("%-10s |        | %s" % ("dataset",
          "  ".join("k=%-2d" % k for k in ks)))
    for name in datasets:
        d = _load(name)
        y = d.y.numpy().astype(int)
        ranks = build_bank(d, "all", name=name)
        F, _ = _feature_matrix(ranks)
        ano = np.where(y == 1)[0]
        rows = {"select": [], "logit_full": [], "logit_sel1": []}
        cols = {r: [] for r in rows}
        for k in ks:
            if k >= len(ano):
                for r in rows:
                    cols[r].append(None)
                continue
            rng = np.random.default_rng(seed + k)
            s_sel, s_lf, s_l1 = [], [], []
            for _ in range(trials):
                shots = rng.choice(ano, size=k, replace=False)
                test = np.ones(len(y), dtype=bool)
                test[shots] = False
                # pseudo-normals: sample from non-shot nodes (PU; contaminated at base rate)
                pool = np.where(test)[0]
                m = min(len(pool), neg_mult * k)
                negs = rng.choice(pool, size=m, replace=False)
                # selection
                fs_name, fs_sign = fewshot_select(ranks, shots)
                s_sel.append(_auc(ranks[fs_name], fs_sign, y, test))
                # logistic over full bank
                Xtr = np.vstack([F[shots], F[negs]])
                ytr = np.concatenate([np.ones(k), np.zeros(m)])
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(Xtr, ytr)
                p = clf.predict_proba(F)[:, 1]
                s_lf.append(roc_auc_score(y[test], p[test]))
                # logistic on the few-shot-selected single channel
                col = F[:, sorted(ranks).index(fs_name)][:, None]
                clf1 = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf1.fit(np.vstack([col[shots], col[negs]]), ytr)
                p1 = clf1.predict_proba(col)[:, 1]
                s_l1.append(roc_auc_score(y[test], p1[test]))
            cols["select"].append(float(np.mean(s_sel)))
            cols["logit_full"].append(float(np.mean(s_lf)))
            cols["logit_sel1"].append(float(np.mean(s_l1)))
        f = lambda v: "n/a " if v is None else "%.3f" % v
        for r in ("select", "logit_full", "logit_sel1"):
            print("%-10s | %-10s | %s" % (name if r == "select" else "", r,
                  "  ".join(f(v) for v in cols[r])))


def run(datasets=None):
    datasets = datasets or DATASETS
    decompose(datasets)
    inversion_table(datasets)
    selection_vs_logistic(datasets)


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
