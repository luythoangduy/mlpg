# Upgrading the Detector Bank — negative on injected benchmarks

`FEWSHOT.md` showed few-shot channel selection works, but over a bank of *trivial*
hand-crafted detectors, and flagged "use a stronger bank" as the next step. This adds four
stronger, still training-free detectors and asks: does a stronger bank raise the oracle
ceiling (and thus the few-shot result)?

Strong detectors added (`detectors.py`):
- `feat_pca` — PCA reconstruction error in normalized feature space.
- `feat_lof` — Local Outlier Factor on normalized features.
- `struct_spec_nonsmooth` — non-smoothness of the spectral embedding (top-k eigenvectors of
  the normalized adjacency) across neighbors.
- `struct_spec_outlier` — Mahalanobis distance in the spectral embedding.

## Results (oracle = best oriented single channel; few-shot k=5, 200 draws)

| dataset  | oracle trivial | oracle +strong | few-shot trivial | few-shot +strong | best strong detector |
|----------|---:|---:|---:|---:|---|
| inj_cora | 0.742 | 0.742 | 0.683 | 0.663 | struct_spec_nonsmooth (0.612) |
| disney   | 0.736 | 0.736 | 0.514 | 0.478 | struct_spec_nonsmooth (0.643) |
| reddit   | 0.592 | 0.592 | 0.539 | 0.539 | feat_lof (0.574) |
| amazon   | 0.892 | 0.892 | 0.874 | 0.856 | feat_lof (0.711) |
| facebook | 0.921 | 0.921 | 0.921 | 0.908 | struct_spec_outlier (0.879) |

## Findings

1. **The strong bank does not raise the oracle on any dataset.** `oracle(+strong)` equals
   `oracle(trivial)` everywhere — every strong detector is weaker than or equal to the best
   trivial one (best strong: 0.61–0.88, all below the trivial oracle on the same graph).

2. **Few-shot is slightly worse with the bigger bank.** Adding channels that don't help only
   increases the chance of selecting a noisy channel from k=5 labels, so `few-shot(+strong) ≤
   few-shot(trivial)` on every dataset.

3. **The bottleneck is the benchmark, not the bank.** These datasets inject *trivial*
   anomalies — structural = degree/density changes, contextual = feature swaps — which the
   degree and 1-hop non-smoothness scores target directly. PCA / LOF / spectral detectors
   capture subtler patterns that simply are not present, so they cannot beat the trivial
   detectors that match the injection process. reddit is the one organic-ish case and its
   ceiling is low (0.592) for *all* detectors — genuinely hard, and not unlocked by a richer
   bank either.

## Conclusion

On injected-anomaly benchmarks, the channel oracle is capped by trivial detectors because
the injection itself is trivial; a stronger training-free bank cannot help, and dilutes
few-shot selection. The lever for higher absolute AUC is therefore **not "more/stronger
detectors" on these graphs** — it is **harder, real-world benchmarks** (and, there, possibly
learned detectors), where subtle anomalies exist for stronger detectors to exploit. The
few-shot *selection* result from `FEWSHOT.md` stands; this experiment delimits where bank
sophistication can and cannot move the ceiling.
