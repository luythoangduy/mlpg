import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a).ravel()


def auc_ap(y_true, score):
    """Return (ROC-AUC, average precision) using sklearn."""
    y = _to_numpy(y_true)
    s = _to_numpy(score)
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
