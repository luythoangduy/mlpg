# Investigation — Why does `gnn_ema` target rescue Disney but lose on Reddit?

Follow-up to the PoC `RESULTS.md`. The clue: `mlpgad_gnntarget` is above 0.5 on **all 5**
Disney seeds (0.547/0.605/0.528/0.510/0.535) yet is the **worst** method on all 5 Reddit
seeds. Stable, not noise. Investigated by stripping the question down to trivial,
learning-free baselines.

## Evidence

Dataset composition:
- Disney: 124 nodes, 6 anomalies, anomalies are **low-degree** (mean deg 2.83 vs 5.53).
- Reddit: 10984 nodes, 366 anomalies, anomalies near-normal degree (12.4 vs 15.4).
- `str_anomaly_label` and `attr_anomaly_label` are degenerate (identical to `Label`) — no
  usable type split, so channels were probed directly.

Learning-free channel AUCs:

| signal (no model) | Disney | Reddit |
|---|---:|---:|
| `-degree` (structural) | **0.742** | 0.444 (anti-correlated) |
| feature L2-norm | 0.459 | 0.576 |
| 1-hop feature non-smoothness | 0.664 | 0.592 |
| naive equal rank-fusion (struct+feat) | **0.770** | 0.552 |
| oracle (best single channel) | 0.736 | 0.592 |

Our learned methods, for reference: Disney best = 0.545 (`gnn_ema`); Reddit best = 0.575
(`mlpgad`).

## Mechanism (resolved)

**The two datasets carry their anomaly signal in orthogonal channels.**

- **Disney is structure-dominant.** A one-line `-degree` detector (0.742) beats every
  learned method. The feature channel is dead/inverted (L2-norm 0.459). `gnn_ema` "wins"
  on Disney only because message passing leaks degree/neighborhood structure into its
  target embedding, so its residual partially tracks the structural signal. The pure-
  feature targets (`mlpgad` frozen-MLP, `unprompt` raw-attribute) see only features, where
  there is no signal → they invert.

- **Reddit is feature-dominant.** `-degree` is anti-correlated (0.444); the signal is
  feature non-smoothness (0.592). The pure-feature target matches it (0.575). The GNN
  target smooths the feature deviation away → worst (0.549).

So the Disney↔Reddit crossover is not a tuning artifact: the target that wins is whichever
channel (feature vs structure) the dataset's anomalies live in.

## Two hard truths

1. **The learned predictability methods do not beat trivial baselines.** "Masked latent
   predictability" reproduces, at best, "1-hop feature non-smoothness" (~0.59-0.66 by a
   3-line numpy script). It is not learning anything beyond that.

2. **No fixed recipe wins both.** Naive equal fusion gets Disney 0.770 but *hurts* Reddit
   (0.552 < the feature-only 0.592) because the structural channel is anti-correlated there
   and poisons the sum. The oracle that picks the right channel per dataset (Disney 0.736,
   Reddit 0.592) is the achievable ceiling — but it requires knowing which channel to
   trust.

## Reframed research direction

The interesting, generalizable problem is **not** "make predictability better" — that
signal is saturated by trivial baselines and only covers feature anomalies. The real open
problem surfaced here:

> **Per-graph adaptive channel selection / weighting.** Anomalies live in different
> channels (structural vs feature) on different graphs, and using the wrong channel
> *inverts* the score. A generalist detector must infer, ideally zero-shot per graph, which
> channel carries the signal and weight structure vs feature accordingly.

This also explains why feature-predictability generalists (UNPrompt-style) are fragile: they
are a single-channel (feature) detector and will invert on structure-dominant graphs like
Disney. The contribution worth pursuing is a principled, data-driven channel router, not
another predictability head.

## Reproduce

```
# per-seed breakdown
python -c "import csv,collections; ..."   # see results/poc_results.csv
# channel baselines (the table above) are computed inline from UNPrompt/Datasets/*.mat
```
