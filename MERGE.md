# Merge vs Select — does keeping both channels help?

Tests the natural alternative to routing: instead of *selecting* one channel per graph,
*merge* both. If the structural and feature channels caught complementary anomaly subsets,
a merge could exceed the best single channel ("super-oracle") and sidestep the unsolved
selection problem.

Channels (label-free, higher = anomalous): `struct` = L2 of z-scored
`[degree, avg_neighbor_degree]`; `feat` = 1-hop feature non-smoothness. Merges over the rank-
transformed channels: `naive` = mean rank, `max` = elementwise max rank, `oracle_w` = best
`w*rank_struct + (1-w)*rank_feat` swept on **real labels** (the complementarity ceiling).

## Results (ROC-AUC)

| dataset  | struct | feat  | select = max(single) | naive merge | max merge | oracle_w merge |
|----------|-------:|------:|---------------------:|------------:|----------:|---------------:|
| inj_cora | 0.720  | 0.522 | 0.720 | 0.667 | 0.711 | 0.725 |
| disney   | 0.333  | 0.664 | 0.664 | 0.498 | 0.465 | 0.664 |
| reddit   | 0.460  | 0.592 | 0.592 | 0.554 | 0.527 | 0.594 |
| amazon   | 0.536  | 0.829 | 0.829 | 0.746 | 0.764 | 0.829 |
| facebook | 0.706  | 0.482 | 0.706 | 0.647 | 0.552 | 0.706 |

## Findings

1. **No complementarity.** `oracle_w merge` ≈ `select` on every dataset (gain ≤ 0.005). Even
   with real labels chosen to optimize the mix, the best weighted merge cannot beat the best
   single channel — the optimal weight just collapses onto the stronger channel. The two
   channels are redundant/competing for the real anomalies, not complementary: each
   dataset's anomalies live almost entirely in one channel, and the other channel is noise
   or *anti-correlated on those same nodes*.

2. **Every label-free merge is worse than select.** `naive` and `max` merges are below
   `select` on all 5 datasets, because there is always a weak/anti channel that dilutes the
   strong one. `max` merge (flag if any channel fires) is also poor: each channel's natural
   false positives (degree hubs in `struct`, feature-diverse nodes in `feat`) pollute the
   union.

## Conclusion

Keeping both channels and merging does **not** escape the hard problem. With no
complementarity, the task reduces exactly to *selection* — and selection is label-free
unsolvable (see `ROUTER.md`, `ROUTER_PERTURBATION.md`). Merge therefore cannot help here;
few-shot channel identification remains the only tractable path.

Caveat: these benchmarks inject a single anomaly type per dataset, so by construction the
channels cannot be complementary. On a graph that genuinely mixes structural and feature
anomalies, merging would matter — but that case is not represented in this benchmark, so it
cannot be claimed from this evidence.
