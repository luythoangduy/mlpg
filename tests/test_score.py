import torch
from mlpgad.score import rank_normalize


def test_monotonic_and_bounded():
    r = torch.tensor([5.0, 1.0, 3.0, 2.0])
    s = rank_normalize(r)
    # order preserved: smallest residual -> smallest score
    assert torch.argsort(s).tolist() == torch.argsort(r).tolist()
    assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0


def test_scale_invariance_of_ranking():
    r = torch.tensor([5.0, 1.0, 3.0, 2.0])
    s1 = rank_normalize(r)
    s2 = rank_normalize(r * 1000.0 + 7.0)
    assert torch.equal(torch.argsort(s1), torch.argsort(s2))
