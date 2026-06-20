import torch
from mlpgad.normalize import coordinate_normalize


def test_zero_mean_unit_std_per_column():
    x = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = coordinate_normalize(x)
    assert torch.allclose(out.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(out.std(dim=0, unbiased=False), torch.ones(2), atol=1e-5)


def test_constant_column_does_not_nan():
    x = torch.tensor([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]])
    out = coordinate_normalize(x)
    assert torch.isfinite(out).all()
