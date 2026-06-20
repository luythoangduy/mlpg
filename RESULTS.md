# MLP-GAD PoC — Results & Verdict

Protocol: 100 epochs, 5 seeds, full-batch, CPU. Score = rank-normalized predictive
residual. Metric = ROC-AUC (mean ± std over 5 seeds). Datasets: `inj_cora` (injected
sanity), `disney` (organic, 124 nodes), `reddit` (organic, 10984 nodes; substitute for
books, which is not shipped in the UNPrompt repo).

## AUC (mean ± std)

| dataset  | unprompt | mlpgad (default) | mlpgad_nomask | mlpgad_gnntarget |
|----------|---------:|-----------------:|--------------:|-----------------:|
| inj_cora | 0.642 ± 0.011 | **0.646 ± 0.011** | 0.444 ± 0.011 | 0.470 ± 0.045 |
| disney   | 0.434 ± 0.069 | 0.414 ± 0.073 | 0.378 ± 0.089 | **0.545 ± 0.032** |
| reddit   | 0.563 ± 0.007 | **0.575 ± 0.007** | 0.568 ± 0.002 | 0.549 ± 0.013 |

(`mlpgad` = default: mlp_frozen target, mask on. Full numbers in `results/poc_results.csv`.)

## Criteria verdict (default `mlpgad`)

- **C1 (alive on organic, > 0.55 on disney AND reddit): FAIL.** reddit 0.575 passes, but
  disney 0.414 is below 0.55 — in fact below random.
- **C2 (no sign inversion; > 0.5 on all 3 with one fixed direction): FAIL.** disney 0.414
  inverts. Rank-calibration fixes score scale/direction but cannot fix a *true* inversion
  where anomalies genuinely have lower predictive residual than normals.
- **C3 (sanity inj_cora > 0.65): MARGINAL FAIL.** 0.646, essentially at the threshold.
- **C4 (mask ≥ nomask on ≥ 2/3): PASS (3/3).** inj_cora 0.646≥0.444, disney 0.414≥0.378,
  reddit 0.575≥0.568. Masking (anti-smoothing) helps consistently.

## Interpretation

1. **The masking / anti-smoothing diagnosis is validated (C4, 3/3).** Removing the mask
   collapses inj_cora to 0.444 (below random) — the context trivially copying the node's
   own features destroys the signal. Masking is necessary.

2. **The differentiator over UNPrompt is negligible.** `mlpgad` (EMA/frozen latent target)
   tracks `unprompt` (raw-attribute target) within noise on every dataset
   (0.646 vs 0.642; 0.575 vs 0.563; 0.414 vs 0.434). The "latent target instead of raw
   attribute" idea does **not** add measurable value at this scale.

3. **The predictability signal still inverts on some organic anomalies.** On disney both
   neighborhood-predictability methods (unprompt, mlpgad) land below 0.5. Only the EMA-GNN
   target (`mlpgad_gnntarget`, 0.545) avoids inversion there — but it loses inj_cora
   (0.470). No single configuration wins across all three datasets. Note disney is tiny
   (124 nodes, ~6 anomalies, std ≈ 0.07), so its result is closer to anecdote than
   evidence; reddit (10984 nodes) is the reliable organic signal, and there the idea is
   weakly alive (0.575) but not better than the UNPrompt baseline.

## Decision

**Do not proceed with the differentiation plan as framed.** The clean one-class
masked-latent-predictability PoC fails C1 and C2, and its latent-target delta over
UNPrompt is within noise. What the PoC *did* establish cheaply:

- Masking is essential (C4) — keep it in any future design.
- Raw-attribute vs latent-EMA target makes no difference — drop that as a "novelty" axis.
- Predictability-residual inverts on a subset of organic anomalies; rank-calibration does
  not rescue it. The next idea must attack the inversion directly (e.g. EMA-GNN target was
  the only thing that helped on disney — worth isolating why), not just recalibrate.

This is a negative-but-useful result: it rules out the proposed differentiator before any
expensive generalist build.
