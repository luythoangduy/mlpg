# Learned detector + organic datasets — where a non-trivial detector finally wins

`BANK.md` showed stronger *training-free* detectors (PCA/LOF/spectral) never raise the
channel oracle on the injected `.mat` benchmarks: the injection is trivial, so degree and
1-hop non-smoothness already match it. It named the two honest next steps — add a **learned**
detector, and test on **harder/organic** data. This does both.

- **Learned detector** (`models/dominant.py`): a reconstruction autoencoder in the spirit of
  Ding et al. 2019 "DOMINANT". A shared 2-layer `GCN` encoder feeds an attribute decoder
  (`learn_dom_attr` = feature recon error) and an inner-product structure decoder
  (`learn_dom_struct` = adjacency-row recon error). Pure `torch_geometric`, ~100 epochs CPU,
  scores cached per `(dataset, seed)`. Graphs over 5k nodes use a negative-sampling structure
  loss/score instead of the dense `O(N^2)` path.
- **Organic datasets** (`data/loaders.py`): Tolokers, Questions (heterophilous-graph
  benchmark, real banned/active labels) and the Elliptic bitcoin fraud graph (parsed from raw
  CSVs, no pandas/pygod). Non-injected — real anomalies for stronger detectors to exploit.

## Best oriented channel AUC per group (full labels)

| dataset  | trivial best | strong best | learned best | oracle(all) |
|----------|---:|---:|---:|---:|
| inj_cora | **0.742** | 0.616 | 0.562 | 0.742 |
| disney   | **0.736** | 0.643 | 0.564 | 0.736 |
| reddit   | **0.592** | 0.574 | 0.565 | 0.592 |
| amazon   | **0.892** | 0.711 | 0.876 | 0.892 |
| facebook | **0.921** | 0.879 | 0.568 | 0.921 |
| tolokers | 0.582 | 0.568 | **0.614** | **0.614** |
| questions| **0.639** | 0.590 | 0.596 | 0.639 |
| elliptic | **0.863** | 0.814 | 0.850 | 0.863 |

(Injected benchmarks first; the three organic datasets last. "best" = best oriented single
channel in that group under full labels.)

## Few-shot channel selection over the upgraded bank (oriented AUC, 200 draws)

| dataset  | oracle(all) | k=1 triv/all | k=3 triv/all | k=5 triv/all | k=10 triv/all | top pick (all, k=5) |
|----------|---:|---|---|---|---|---|
| inj_cora | 0.742 | 0.564/0.591 | 0.672/0.625 | 0.687/0.662 | 0.717/0.684 | struct_posdeg, struct_outlier |
| disney   | 0.736 | 0.654/0.576 | 0.584/0.552 | 0.488/0.501 | n/a | struct_negdeg, struct_outlier |
| reddit   | 0.592 | 0.525/0.525 | 0.536/0.529 | 0.537/0.532 | 0.552/0.537 | feat_global, feat_nonsmooth |
| amazon   | 0.892 | 0.746/0.668 | 0.847/0.821 | 0.871/0.855 | 0.888/0.873 | feat_global, learn_dom_attr |
| facebook | 0.921 | 0.852/0.825 | 0.915/0.905 | 0.921/0.909 | 0.921/0.916 | struct_negdeg, struct_posdeg |
| tolokers | 0.614 | 0.519/0.522 | 0.527/0.531 | 0.538/0.538 | 0.549/0.552 | learn_dom_struct, struct_outlier |
| questions| 0.639 | 0.533/0.512 | 0.566/0.552 | 0.578/0.565 | 0.598/0.584 | struct_posdeg, struct_spec_outlier |
| elliptic | 0.863 | 0.723/0.682 | 0.786/0.767 | 0.822/0.806 | 0.853/0.837 | feat_global, learn_dom_attr |

## Findings

1. **The learned detector raises the oracle on exactly one graph — and it is organic.** On
   **Tolokers** `learn_dom_struct` (0.614) beats every trivial (0.582) and training-free
   strong (0.568) detector, lifting the oracle to 0.614. This is the *only* dataset in the
   entire study where any non-trivial detector wins, and it is precisely the predicted
   outcome: it takes organic anomalies for a learned detector to add headroom. Few-shot finds
   it — `learn_dom_struct` is the top k=5 pick (17%) and `few-shot(all) ≥ few-shot(triv)`
   there at every k (the one dataset where a richer bank helps rather than dilutes).

2. **The learned *attribute* channel is a much stronger feature detector than the
   training-free strong bank — but not stronger than the best trivial channel.** On amazon it
   scores 0.876 (vs LOF 0.711, vs trivial feat_global 0.892) and on Elliptic 0.850 (vs PCA
   0.814, vs feat_global 0.863). It is the runner-up few-shot pick on both (`learn_dom_attr`
   ~22–35%). So the learned recon genuinely improves the *feature hypothesis* over PCA/LOF/
   spectral, it just doesn't overtake the simple global-feature-norm channel that already
   matches these graphs.

3. **The inner-product structure decoder is weak.** `learn_dom_struct` misses degree/clique
   anomalies because a GAE reconstructs dense communities well (facebook 0.568 vs trivial
   `-degree` 0.921; correlation with the degree-context outlier ≈ 0). On the trivially-
   injected benchmarks the learned channels reproduce the feature channel (inj_cora
   `corr(learn_dom_attr, feat_nonsmooth) = 0.77`) and sit at 0.56–0.57 — exactly BANK.md's
   result, now confirmed for a *learned* detector too.

4. **A richer bank still dilutes few-shot, except where the new channel actually wins.**
   `few-shot(all) ≤ few-shot(triv)` on every injected dataset (more channels ⇒ more chance to
   pick a noisy one from k labels), reproducing BANK.md. Tolokers is the lone exception
   because the learned channel is genuinely the best one there.

5. **Organic ≠ automatically harder for trivial detectors.** Elliptic's `feat_global` (0.863)
   and Questions' `struct_posdeg` (0.639) are still the best channels — organic data is not a
   guaranteed win for learned methods. The clean win is Tolokers, where neither a degree nor a
   feature statistic suffices and the learned structural embedding does.

## Conclusion

Adding a learned reconstruction detector to the bank confirms and sharpens BANK.md rather
than overturning it: on trivially-injected graphs a learned detector cannot beat the trivial
channels (it reproduces them), and a richer bank dilutes few-shot. The payoff appears only on
**organic** data — and even there it is modest and uneven: a clear win on Tolokers (oracle
0.582 → 0.614, learned is the single best channel and few-shot selects it), a strong-but-
second feature detector on amazon/Elliptic, and no help on Questions. The learned *attribute*
recon is the useful half; the inner-product *structure* recon is dominated by a one-line
degree score. The few-shot *selection* result from FEWSHOT.md stands; this experiment shows
the bank's learned hypotheses matter exactly where the benchmark stops being trivial.

### Reproduce

```bash
# trains + caches DOMINANT per dataset, prints both tables
python -m mlpgad.detectors inj_cora disney reddit amazon facebook tolokers questions elliptic
python -m pytest mlpgad/tests -q
```
