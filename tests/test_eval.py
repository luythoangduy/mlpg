import numpy as np
import torch
from mlpgad.eval import auc_ap


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8])
    auc, ap = auc_ap(y, s)
    assert auc == 1.0
    assert ap == 1.0


def test_accepts_tensors():
    y = torch.tensor([0, 1, 0, 1])
    s = torch.tensor([0.2, 0.9, 0.1, 0.8])
    auc, ap = auc_ap(y, s)
    assert 0.99 <= auc <= 1.0
