import torch

from mlpgad.score import rank_normalize


def _sample_mask(num_nodes, mask_rate, generator, device):
    k = max(1, int(mask_rate * num_nodes))
    return torch.randperm(num_nodes, generator=generator, device=device)[:k]


def train_one_class(model, data, *, epochs=100, lr=1e-3, mask_rate=0.5,
                    grad_clip=5.0, seed=0, device="cpu"):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = x.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    model.train()
    for _ in range(epochs):
        masked = _sample_mask(num_nodes, mask_rate, g, device)
        opt.zero_grad()
        loss = model(x, edge_index, masked).mean()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        model.update_target()
    return model


@torch.no_grad()
def score_nodes(model, data, *, mask_rate=0.5, rounds=10, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed + 10_000)
    model = model.to(device).eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = x.shape[0]
    res_sum = torch.zeros(num_nodes, device=device)
    res_cnt = torch.zeros(num_nodes, device=device)
    for _ in range(rounds):
        masked = _sample_mask(num_nodes, mask_rate, g, device)
        r = model(x, edge_index, masked)
        res_sum[masked] += r
        res_cnt[masked] += 1.0
    # nodes never masked: fall back to a single full-mask pass
    never = res_cnt == 0
    if bool(never.any()):
        idx = torch.nonzero(never, as_tuple=False).view(-1)
        res_sum[idx] += model(x, edge_index, idx)
        res_cnt[idx] += 1.0
    residual = res_sum / res_cnt.clamp(min=1.0)
    return rank_normalize(residual).cpu()
