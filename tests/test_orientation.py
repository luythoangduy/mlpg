import numpy as np

from mlpgad.orientation import (_feature_matrix, _fewshot_sign, _oracle_pick,
                                _oracle_sign, _safe_share)


def _toy_ranks():
    # channel "good" ranks anomalies high; "inverted" ranks them low; "noise" random
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    good = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    inv = 1.0 - good
    rng = np.random.default_rng(0)
    noise = rng.random(10)
    return {"good": good, "inverted": inv, "noise": noise}, y


def test_oracle_pick_prefers_separating_channel_and_orients():
    ranks, y = _toy_ranks()
    name, sign = _oracle_pick(ranks, y)
    assert name in ("good", "inverted")  # both perfectly separate after orientation
    # oriented AUC of the pick must be the best achievable (= 1.0 here)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, sign * ranks[name]) == 1.0


def test_oracle_and_fewshot_sign_detect_inversion():
    ranks, y = _toy_ranks()
    assert _oracle_sign(ranks["good"], y) == 1.0
    assert _oracle_sign(ranks["inverted"], y) == -1.0
    shots = np.where(y == 1)[0]
    assert _fewshot_sign(ranks["good"], shots) == 1.0      # anomalies high -> +1
    assert _fewshot_sign(ranks["inverted"], shots) == -1.0  # anomalies low  -> -1


def test_feature_matrix_shape_and_order():
    ranks, _ = _toy_ranks()
    F, names = _feature_matrix(ranks)
    assert F.shape == (10, 3)
    assert names == sorted(ranks)  # sorted for stable column indexing


def test_safe_share_handles_zero_gap():
    assert "n/a" in _safe_share(0.1, 0.0)
    assert _safe_share(0.5, 1.0).strip() == "50%"
