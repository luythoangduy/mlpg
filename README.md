# mlpgad — Masked Latent Predictability for Graph Anomaly Detection (PoC + investigation)

A small, pure-`torch_geometric` study testing whether a one-class **masked latent
predictability** detector (JEPA/BYOL-style: predict a node's latent target from its masked
neighborhood; predictive residual = anomaly score) is a viable, generalizable graph anomaly
detector. It is **not** a SOTA attempt — it is a cheap proof-of-concept designed to *kill or
keep* an idea quickly, plus the follow-up investigation that reframed the problem.

**TL;DR of the result:** the masked-predictability idea does **not** beat trivial baselines,
and its differentiator over UNPrompt is within noise. The investigation found the real,
high-value, open problem: **per-graph channel routing** — anomalies live in the structural
*or* feature channel depending on the graph, using the wrong one *inverts* the score, and
no simple unsupervised statistic selects the right channel across graphs.

## Findings at a glance

1. **PoC verdict ([RESULTS.md](RESULTS.md)):** the clean one-class predictability model
   fails on organic anomalies (disney AUC 0.41, inverted) and only weakly survives on
   reddit (0.575) — within noise of the UNPrompt-style baseline (0.563). Masking is the one
   component clearly validated (removing it sends inj_cora to 0.444, below random).

2. **Why it fails ([INVESTIGATION.md](INVESTIGATION.md)):** datasets carry their anomaly
   signal in **orthogonal channels**. Disney is structure-dominant — a one-line `-degree`
   detector scores **0.742**, beating every learned method; the feature channel is dead
   (0.459). Reddit is feature-dominant — `-degree` is anti-correlated (0.444). The learned
   methods merely reproduce "1-hop feature non-smoothness" (~0.59–0.66 by a 3-line numpy
   script) and only cover the feature channel.

3. **Can we route channels? ([ROUTER.md](ROUTER.md)):** per-graph channel selection is the
   high-value lever (oracle AUC 0.52–0.92, vs learned ~0.5), but a simple unsupervised
   router (kurtosis/tail-pathology) that looked perfect on 2 datasets **collapses on 5**
   (amazon 0.49, facebook 0.79 vs oracle 0.83/0.92). Choosing a channel from its own
   marginal distribution is unsolved.

4. **Perturbation-response router ([ROUTER_PERTURBATION.md](ROUTER_PERTURBATION.md)):**
   estimating channel *competence* by self-injection also **fails** — routed 0.577 <
   naive-fusion 0.622 < oracle 0.702, with counterexamples both ways (disney: high
   competence yet inverted relevance; amazon: low competence yet the relevant channel).
   Competence (detectability of injected type) does not predict relevance (real-anomaly
   channel).

### Per-graph channel AUC (learning-free channels)

| dataset  | struct (`-deg`) | feat (1-hop non-smooth) | oracle (best) | best *learned* method |
|----------|---:|---:|---:|---:|
| inj_cora | 0.261 | 0.522 | 0.522 | 0.646 |
| disney   | 0.742 | 0.664 | 0.742 | 0.545 |
| reddit   | 0.444 | 0.592 | 0.592 | 0.575 |
| amazon   | 0.491 | 0.829 | 0.829 | — |
| facebook | 0.922 | 0.482 | 0.922 | — |

The dominant channel flips per graph; where one channel is strong (facebook structure,
amazon feature) it is *far* above any learned predictability method.

## What this rules in / out

- **Out:** "make masked latent predictability better" — saturated by trivial baselines,
  single-channel (feature) only, inverts on structure-dominant graphs. The latent-vs-raw
  target distinction adds nothing.
- **Out (now tested):** *zero-shot, label-free* per-graph channel routing. Both marginal-
  statistic routing and perturbation-response competence routing fail (below naive fusion).
  On a new graph with no labels the anomaly type is unidentifiable, and choosing the wrong
  channel inverts the score.
- **In (open, tractable):** **few-shot channel identification** — with 1–5 labeled
  anomalies per graph, pick/weight the channel whose ranking they top (oracle shows
  0.59–0.92). This matches how strong generalist GAD methods (ARC, UNPrompt, AnomalyGFM)
  import external signal about what is anomalous.

## Repository layout

```
mlpgad/
  normalize.py            coordinate-wise feature z-score
  score.py                rank-normalized scoring (fixed direction, scale-free)
  eval.py                 sklearn ROC-AUC / AP
  data/loaders.py         PyG loaders: .mat datasets + injected Cora (no pygod)
  models/mlpgad.py        masked latent-predictability model (mlp_frozen / gnn_ema, mask toggle)
  models/unprompt_baseline.py   neighborhood-predictability of raw attribute (UNPrompt-style)
  train.py                one-class trainer + multi-round node scorer
  run_poc.py              grid runner {dataset x method x toggle x seed} -> CSV
  router_perturbation.py  perturbation-response channel-router experiment
  configs/default.yaml    experiment config
  tests/                  15 unit tests (pytest)
  RESULTS.md              PoC results + C1-C4 verdict
  INVESTIGATION.md        channel crossover analysis
  ROUTER.md               channel-router PoC, marginal statistics (negative)
  ROUTER_PERTURBATION.md  perturbation-response router (negative) + few-shot reframe
  docs/                   approved design spec
  results/poc_results.csv raw per-seed AUC/AP
```

## Reproduce

Requires an environment with `torch`, `torch_geometric`, `scipy`, `scikit-learn`,
`numpy`, `pyyaml`, `pytest`. Datasets (`Disney/Reddit/Amazon/Facebook.mat`) come from the
[UNPrompt repo](https://github.com/mala-lab/UNPrompt); `inj_cora` is built from Planetoid
Cora.

```bash
# unit tests
python -m pytest mlpgad/tests -q

# full PoC grid (100 epochs, 5 seeds, 3 datasets)
python -m mlpgad.run_poc --config mlpgad/configs/default.yaml --out results/poc_results.csv
```

Point `configs/default.yaml: unprompt_dir` at the folder containing the `.mat` files.

## Provenance

Built as a verification spike off a prior `CONADJEPA`/`GADJEPA` exploration. The design spec
and implementation plan are under `docs/`. Constraint throughout: `torch_geometric` only,
never import `pygod`.
