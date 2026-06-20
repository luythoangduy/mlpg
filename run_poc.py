import argparse
import csv
import os
import statistics

import torch
import yaml

from mlpgad.data.loaders import load_dataset
from mlpgad.eval import auc_ap
from mlpgad.models.mlpgad import MLPGAD
from mlpgad.models.unprompt_baseline import UNPromptBaseline
from mlpgad.train import score_nodes, train_one_class

METHODS = {
    "unprompt": lambda d, c: UNPromptBaseline(
        d, c["hid_dim"], c["num_layers"], c["backbone"]),
    "mlpgad": lambda d, c: MLPGAD(
        d, c["hid_dim"], c["num_layers"], c["backbone"],
        target_type="mlp_frozen", mask=True),
    "mlpgad_nomask": lambda d, c: MLPGAD(
        d, c["hid_dim"], c["num_layers"], c["backbone"],
        target_type="mlp_frozen", mask=False),
    "mlpgad_gnntarget": lambda d, c: MLPGAD(
        d, c["hid_dim"], c["num_layers"], c["backbone"],
        target_type="gnn_ema", mask=True),
}


def _load(name, cfg):
    try:
        return load_dataset(
            name, unprompt_dir=cfg["unprompt_dir"], cora_root=cfg["cora_root"],
            books_path=cfg.get("books_path"), seed=0)
    except FileNotFoundError:
        if name == "books":
            fb = cfg.get("books_fallback", "reddit")
            print("[warn] books unavailable, falling back to {}".format(fb))
            return load_dataset(fb, unprompt_dir=cfg["unprompt_dir"],
                                cora_root=cfg["cora_root"], seed=0)
        raise


def run(cfg):
    rows = []
    for name in cfg["datasets"]:
        data = _load(name, cfg)
        in_dim = data.x.shape[1]
        for method, build in METHODS.items():
            aucs, aps = [], []
            for seed in cfg["seeds"]:
                torch.manual_seed(seed)
                model = build(in_dim, cfg)
                train_one_class(model, data, epochs=cfg["epochs"], lr=cfg["lr"],
                                mask_rate=cfg["mask_rate"], seed=seed)
                score = score_nodes(model, data, mask_rate=cfg["mask_rate"],
                                    rounds=cfg["score_rounds"], seed=seed)
                auc, ap = auc_ap(data.y, score)
                aucs.append(auc)
                aps.append(ap)
                rows.append({"dataset": name, "method": method, "seed": seed,
                             "auc": auc, "ap": ap})
            print("{:10s} {:18s} AUC={:.4f}+/-{:.4f}  AP={:.4f}".format(
                name, method, statistics.mean(aucs),
                statistics.pstdev(aucs), statistics.mean(aps)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mlpgad/configs/default.yaml")
    ap.add_argument("--out", default="results/poc_results.csv")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rows = run(cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "seed", "auc", "ap"])
        w.writeheader()
        w.writerows(rows)
    print("wrote {} rows to {}".format(len(rows), args.out))


if __name__ == "__main__":
    main()
