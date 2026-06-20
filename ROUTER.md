# Channel Router PoC — can we pick the right channel per graph, zero-shot?

Follow-up to `INVESTIGATION.md`, which showed anomalies live in different channels
(structure vs feature) on different graphs, and that *oracle* per-graph channel selection
far exceeds any learned predictability method. Question: is there a **label-free** statistic
that selects/weights the right channel per graph?

Channels (hand-crafted, no learning, higher = more anomalous):
- `struct` = `-degree` (low-degree as anomaly).
- `feat` = 1-hop feature non-smoothness `||x_v - mean_neighbors(x)||` in normalized space.

Routers tried:
- `gated` = drop any channel with excess kurtosis > 50 (pathological tail), rank-mean the rest.
- `soft`  = weight each channel by `1 / log(10 + kurtosis)`, rank-fuse.

## Results (ROC-AUC)

| dataset  | struct | feat  | oracle (best single) | gated | soft  |
|----------|-------:|------:|---------------------:|------:|------:|
| inj_cora | 0.261  | 0.522 | 0.522 | 0.522 | 0.435 |
| disney   | 0.742  | 0.664 | 0.742 | 0.769 | 0.770 |
| reddit   | 0.444  | 0.592 | 0.592 | 0.592 | 0.593 |
| amazon   | 0.491  | 0.829 | 0.829 | **0.492** | 0.651 |
| facebook | 0.922  | 0.482 | 0.922 | **0.791** | 0.780 |

For reference, the best *learned* method from the PoC was ~0.55 (disney) / ~0.58 (reddit).

## Findings

1. **Per-graph channel dominance flips wildly, and the prize is large.** Oracle ranges
   0.52–0.92. On facebook structure wins (0.922 vs 0.482); on amazon feature wins
   (0.829 vs 0.491); disney/reddit/inj_cora are milder. Where one channel is strong, it is
   *far* above any learned predictability method (~0.5). So **selecting the right channel
   is the high-value lever**, not improving predictability.

2. **The n=2 "win" was overfit.** On Disney+Reddit alone, kurtosis-gating looked great
   (matched/beat oracle). Expanding to 5 datasets breaks it:
   - **amazon:** struct kurtosis is mild (14), so the gate keeps it and naive-fuses a
     useless struct (0.491) into a strong feat (0.829) → collapses to **0.492**.
   - **facebook:** symmetric failure — fusing weak feat into strong struct gives 0.791 vs
     oracle 0.922.

3. **Tail-pathology is not a channel-quality proxy.** A channel can be useless *without*
   being pathologically tailed (amazon struct kurt=14, facebook feat kurt mild). Kurtosis
   only catches the special case of power-law degree hubs (reddit struct kurt=685). The
   marginal-distribution statistics tried earlier each fail somewhere:
   - tail/separation stats (top-gap, max-z) always prefer the feature channel;
   - kurtosis always prefers the structural channel;
   - neither tracks actual channel AUC across all 5 datasets.

## Conclusion

**Simple unsupervised routing on channel marginals does not generalize.** The value of
per-graph channel selection is real and large (oracle 0.83–0.92 where learned methods sit
at ~0.5), but choosing the channel from its own score distribution is unsolved: channel
quality (does this channel separate a small minority outlier set?) is not captured by tail
shape or spread.

## Next experiment (genuinely open, not yet done)

Estimate channel **quality** by *perturbation response* rather than marginal shape:
inject synthetic structural anomalies and, separately, synthetic feature anomalies (cheap,
CONAD-style); measure how strongly each channel's score separates its *own* injected type
(label-free, since we know the injected labels). A channel that cannot even detect its own
injected anomaly type is unreliable for the real data and should be down-weighted; the mix
of injected-type detectability also hints at which channel the real graph rewards. This
turns routing into a self-supervised, per-graph calibration problem and is the contribution
worth validating — on a *multi-graph* benchmark (≥ the 5 here, ideally GADBench-scale),
because n=2 demonstrably overfits.
