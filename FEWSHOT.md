# Few-Shot Channel Identification — positive result

Zero-shot channel routing is unsolvable and merging is dominated by selection
(`ROUTER.md`, `ROUTER_PERTURBATION.md`, `MERGE.md`). This tests the remaining tractable
path: with **k labeled anomalies per graph**, select *and orient* the best channel from a
small bank of label-free candidate detectors, then evaluate on the rest.

Channel bank (rank-transformed; higher = more anomalous before orientation):
`feat_nonsmooth`, `feat_global`, `struct_negdeg` (−degree), `struct_posdeg` (+degree),
`struct_outlier` (|z| of [degree, avg-neighbor-degree]).

Selection: pick the channel whose k shots are most extreme (`max |mean_rank − 0.5|`); orient
by the sign of that deviation (this fixes inversion). Evaluate the oriented channel on all
nodes except the k shots. 200 random shot draws per (dataset, k).

## Results (ROC-AUC; oracle = best oriented single channel with full labels)

| dataset  | oracle | feat (naive) | k=1 | k=3 | k=5 | k=10 | few-shot pick (k=5) |
|----------|-------:|-------------:|----:|----:|----:|-----:|---------------------|
| inj_cora | 0.742  | 0.522 | 0.564 | 0.677 | 0.679 | 0.709 | struct_outlier / posdeg |
| disney   | 0.736  | 0.664 | 0.653 | 0.598 | 0.515 | n/a   | struct_neg / posdeg |
| reddit   | 0.592  | 0.592 | 0.531 | 0.527 | 0.546 | 0.553 | feat_global / nonsmooth |
| amazon   | 0.892  | 0.829 | 0.752 | 0.850 | 0.874 | 0.885 | feat_global (85%) |
| facebook | 0.921  | 0.482 | 0.856 | 0.921 | 0.920 | 0.920 | struct_negdeg (60%) |

## Findings

1. **A handful of labels recovers near-oracle AUC where zero-shot inverts.** facebook: naive
   feature scoring is useless (0.482), but with k=3 labels few-shot reaches **0.921 = oracle**
   — it identifies that structure is the channel *and* fixes the orientation. This is exactly
   the failure mode that killed every zero-shot router.

2. **Few-shot can beat the hand-picked default.** On amazon it discovers `feat_global`
   (oracle 0.892) over the default `feat_nonsmooth` (0.829), reaching 0.885 at k=10.

3. **k = 3–5 is usually enough; k = 1 is noisy.** Performance rises monotonically with k on
   inj_cora, amazon, facebook and plateaus near oracle by k = 3–10.

4. **Where it does not help, honestly:** reddit's ceiling is low (oracle 0.592 = naive feat),
   so few-shot only adds sampling noise (~0.53–0.55). disney has just 6 anomalies, so k = 5
   leaves a single test positive — its AUC there (0.515) is an evaluation artifact, not a
   method failure.

## Conclusion

**Few-shot channel identification is the working direction.** With 3–5 labeled anomalies per
graph it selects and orients the right detector, recovering near-oracle AUC and curing the
inversion that makes zero-shot routing impossible. The largest gains appear precisely on
structure-dominant graphs (facebook 0.48 → 0.92) where single-channel and zero-shot methods
fail.

Caveats / next steps:
- The bank here is **trivial hand-crafted detectors**. The real contribution would be a
  few-shot router over a bank of *stronger* detectors (learned structural reconstruction,
  feature predictability, spectral) — few-shot then picks among genuine hypotheses, not toy
  scores.
- Validate on a larger benchmark (GADBench-scale) and against few-shot GAD baselines
  (ARC few-shot, UNPrompt few-shot, AnomalyGFM few-shot prompt tuning).
- A weighting variant (logistic regression over channel scores) may help when anomalies
  genuinely span channels, but needs more than k = 1–5 positives to avoid overfitting.
