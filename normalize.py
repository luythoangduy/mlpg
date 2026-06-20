import torch


def coordinate_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-feature (column-wise) z-score normalization."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-12)
    return (x - mean) / std
