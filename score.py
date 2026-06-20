import torch


def rank_normalize(residual: torch.Tensor) -> torch.Tensor:
    """Map residuals to percentile ranks in [0, 1]; higher residual -> higher score."""
    n = residual.numel()
    order = torch.argsort(residual)
    ranks = torch.empty(n, dtype=torch.float, device=residual.device)
    ranks[order] = torch.arange(n, dtype=torch.float, device=residual.device)
    if n > 1:
        ranks = ranks / (n - 1)
    return ranks
