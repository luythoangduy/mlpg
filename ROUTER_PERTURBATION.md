# Perturbation-Response Router — decisive negative

Tests the open idea proposed at the end of `ROUTER.md`: estimate a channel's **competence**
(how well it detects its OWN synthetic injected anomaly type against the graph's natural
background) and use it as a label-free proxy for **relevance** (real-anomaly AUC), to route
between a structural and a feature channel per graph.

Setup (script: `router_perturbation.py`, all numpy, no training, 5 datasets):
- `struct` channel = L2 norm of z-scored `[degree, avg_neighbor_degree]` (outlierness).
- `feat` channel = 1-hop feature non-smoothness.
- Competence = AUC of a channel detecting its own moderate injection (5% nodes; structural
  densify/sparsify, or feature swap with a random node) against the real background.

## Results

| dataset  | real struct | real feat | comp struct | comp feat | router pick | routed | oracle |
|----------|---:|---:|---:|---:|:--:|---:|---:|
| inj_cora | 0.720 | 0.522 | 0.784 | 0.520 | struct | 0.720 | 0.720 |
| disney   | 0.333 | 0.664 | 0.907 | 0.607 | struct | 0.333 | 0.664 |
| reddit   | 0.460 | 0.592 | 0.648 | 0.972 | feat   | 0.592 | 0.592 |
| amazon   | 0.536 | 0.829 | 0.643 | 0.529 | struct | 0.536 | 0.829 |
| facebook | 0.706 | 0.482 | 0.642 | 0.462 | struct | 0.706 | 0.706 |

Means: **routed 0.577 < naive-fusion 0.622 < oracle 0.702.** Competence picks the better
channel on only 3/5 datasets, and the competence-weighted soft fusion (0.596) is also below
naive fusion.

## Why it fails — counterexamples in both directions

1. **High competence, inverted relevance (disney).** Struct competence is the highest in
   the table (0.907) — injected degree changes are trivially detectable on a tiny tight
   graph — yet the struct channel's real AUC is 0.333 (inverted). Detectability of an
   injected type says nothing about whether real anomalies live in that channel.

2. **Low competence, high relevance (amazon).** Feature is the correct channel (real AUC
   0.829), but a *moderate* feature injection is masked by amazon's large natural feature
   variance, so feat competence is only 0.529 → the router down-weights the channel that
   actually matters.

No single injection strength fixes both: the failures point in opposite directions, so this
is a structural flaw of the competence proxy, not a tuning issue.

## Secondary finding — "channel" is not well-defined

Changing only the structural channel definition (`-degree` in `ROUTER.md` vs. `|z|` of
`[degree, avg_neighbor_degree]` here) flips real AUCs drastically: inj_cora 0.26 -> 0.72,
disney 0.74 -> 0.33, facebook 0.92 -> 0.71. There are many structural sub-channels and which
one is relevant also flips per graph. The routing space is larger and messier than two
channels.

## Conclusion and reframe

Both label-free routing attempts now fail: marginal-distribution statistics (`ROUTER.md`)
and self-injection competence (here). The likely reason is fundamental: **on a brand-new
graph with no labels, the anomaly type is unidentifiable** — nothing in the unlabeled graph
distinguishes "structural anomalies are the targets" from "feature anomalies are the
targets," and choosing wrong inverts the score.

This matches why every strong generalist GAD method imports *external* information about what
counts as anomalous: ARC uses a few in-context normal examples, UNPrompt/AnomalyGFM use
cross-graph pre-training, few-shot variants use a handful of labels. The realistic, tractable
problem is therefore **few-shot channel identification**: with as few as 1-5 labeled
anomalies per graph, pick/weight the channel whose ranking they top — which the oracle row
shows would yield 0.59-0.92. Pure zero-shot single-graph channel routing appears to be a
dead end and should be abandoned in favor of the few-shot framing.
