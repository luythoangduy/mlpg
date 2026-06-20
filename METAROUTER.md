# Cross-graph meta-router — can orientation be learned across graphs? (decisive negative)

`ROUTER.md` / `ROUTER_PERTURBATION.md` killed *zero-shot, single-graph* channel routing on
hand-crafted statistics; `ORIENTATION.md` showed the binding constraint is the per-graph
*sign* (the same channel inverts across graphs). The open question those left: with a *corpus*
of graphs, can a **supervised** meta-learner predict the right channel and its orientation for
a held-out graph from label-free meta-features? This turns the failed unsupervised router into
a cross-graph learning problem — the natural "did it fail only because it was unsupervised?"
test, and the lever the Feb-2026 zero-shot MoE routers bet on.

**Probe (honest scope):** 8 real graphs, leave-one-graph-out (LOGO). n = 8 is tiny, so the
floor is a **majority baseline** (predict the most common answer among the 7 training graphs)
and the only feasible learner is **1-NN** over standardized meta-features (15 label-free graph
descriptors: degree skew/kurtosis/CV/assortativity/tail-gap, feature non-smoothness and
global-norm skew/kurtosis/tail-gap, size/density/feature-dim). Script: `metarouter.py`.

## (A) Orientation prediction accuracy (LOGO, n = 8)

Predict each channel's per-graph sign (+1/−1) for the held-out graph.

| channel | 1-NN meta | majority | always +1 |
|---|---:|---:|---:|
| struct_negdeg | 4/8 | **5/8** | 3/8 |
| struct_posdeg | 4/8 | **5/8** | 5/8 |
| feat_nonsmooth | 3/8 | **5/8** | 5/8 |
| feat_global | **5/8** | 0/8 | 4/8 |
| learn_dom_attr | **5/8** | 0/8 | 4/8 |

**1-NN does not reliably beat majority.** On 3/5 channels it is *worse*; on the two where it
"wins" (feat_global, learn_dom_attr) majority scores 0/8 because the sign is split 4/4 across
graphs — i.e. the target is balanced and 1-NN's 5/8 is barely above chance. There is **no
transferable orientation signal** in the meta-features: whether a channel ranks anomalies high
or low on a new graph cannot be predicted from that graph's structure.

## (B) Oracle channel-family prediction (LOGO)

Predict whether the graph's best channel is structural or feature.

| | 1-NN meta | majority |
|---|---:|---:|
| family accuracy | **6/8** | 5/8 |

Channel *family* is marginally more predictable than orientation (one extra graph over
majority) — weak, and anecdotal at n = 8, but directionally consistent: *which channel* carries
a little cross-graph signal; *which direction* carries essentially none.

## (C) Downstream zero-shot AUC vs few-shot vs oracle

| dataset | naive | majority | 1-NN meta | few-shot k=5 | oracle |
|---|---:|---:|---:|---:|---:|
| inj_cora | 0.522 | 0.262 | 0.262 | 0.660 | 0.742 |
| disney   | 0.664 | 0.270 | 0.736 | 0.515 | 0.736 |
| reddit   | 0.592 | 0.553 | 0.553 | 0.533 | 0.592 |
| amazon   | 0.829 | 0.508 | 0.108 | 0.863 | 0.892 |
| facebook | 0.482 | 0.079 | 0.921 | 0.910 | 0.921 |
| tolokers | 0.425 | 0.560 | 0.440 | 0.538 | 0.614 |
| questions| 0.504 | 0.363 | 0.504 | 0.566 | 0.639 |
| elliptic | 0.224 | 0.317 | 0.224 | 0.806 | 0.863 |
| **MEAN** | **0.530** | **0.364** | **0.469** | **0.674** | **0.750** |

Both zero-shot meta-routers are **below even the naive fixed feature detector** (majority
0.364, 1-NN 0.469 vs naive 0.530) and far below few-shot (0.674). 1-NN has catastrophic
inversions where the nearest training graph carries the wrong channel/sign (amazon 0.108,
inj_cora 0.262) — the high-variance failure of trusting one neighbor when orientation does not
transfer. few-shot, using just 5 labels on the held-out graph, closes most of the
naive→oracle gap that the meta-router cannot touch.

## Conclusion

**Cross-graph meta-learning does not rescue zero-shot channel/orientation routing.** Even with
supervision over a corpus, (i) orientation is unpredictable from graph meta-features (1-NN ≤
majority, balanced 4/4 signs), (ii) channel-family is only marginally predictable, and (iii)
downstream both meta-routers are worse than a naive fixed detector and far below few-shot. This
extends `ROUTER_PERTURBATION.md`'s "the anomaly type is unidentifiable on an unlabeled graph"
from single-graph to the cross-graph setting, and pinpoints **orientation** as the
specifically unlearnable part. It is the strongest statement yet of why a handful of target
labels is necessary: the sign that few-shot recovers is information that simply is not present
in the graph, on its own or by analogy to other graphs.

**Caveats.** n = 8 (LOGO) is a probe, not a benchmark; 1-NN is the only viable learner at this
size. A GADBench-scale corpus could move channel-family predictability. But the *orientation*
target being balanced (majority 0/8 on feat_global/learn_dom_attr) is a structural fact, not a
sample-size artifact: there is no canonical direction for a channel across graphs, so no
meta-feature can encode one. A richer learner on more graphs would still face a sign that is
defined only by the (hidden) anomaly semantics.

### Reproduce

```bash
python -m mlpgad.metarouter
python -m pytest mlpgad/tests/test_metarouter.py -q
```
