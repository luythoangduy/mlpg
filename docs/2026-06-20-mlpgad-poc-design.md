# MLP-GAD PoC — Masked Latent Predictability for Graph Anomaly Detection

**Date:** 2026-06-20
**Status:** Design approved, pending spec review
**Type:** Proof-of-concept verification experiment

## 1. Goal

Verify whether a **one-class predictive-residual** graph anomaly detector — "normality =
predictability of a node's latent content from its masked neighborhood, in a normalized
latent space" — survives on **organic** anomaly datasets, where the existing CONADJEPA
model fails (disney AUC 0.30–0.41, i.e. below random).

This is a **proof-of-concept**, not a SOTA-beating attempt. It answers a binary question:
*does the predictability signal, done cleanly (no pseudo-anomaly margin, anchored latent
target, masked context, rank-calibrated score), rank organic anomalies above normals
consistently?* If yes → proceed to differentiate vs UNPrompt. If no → stop early.

### Background / motivation (from prior analysis)

CONADJEPA's JEPA head failed for three diagnosed root causes:
1. **Margin loss** on synthetic CONAD anomalies → overfits the injection distribution,
   fails on organic anomalies.
2. **Predictability inversion under message passing** → well-connected anomalies get
   smoothed and become *more* predictable (graph-target AUC 0.418, disney 0.30–0.41,
   both below random).
3. **Self-referential, unanchored target** (feature branch = BYOL loop with no
   reconstruction anchor) → high seed variance (±0.097).

This PoC removes the margin, anchors the target, masks the context (anti-smoothing), and
rank-calibrates the score — each toggle maps to one diagnosed cause.

### Novelty honesty

At node level, "predict a masked node from its neighborhood" is the same family as
**UNPrompt** (IJCAI'25) and **TAM** (NeurIPS'23). The PoC's delta over UNPrompt is:
*latent EMA target instead of raw attribute* + *masked context* + *rank calibration*.
The PoC does **not** claim novelty yet — it establishes whether the signal is alive.

## 2. Constraints

- **`torch_geometric` only.** No PyGOD library dependency (neither models, loaders, nor
  metrics). Data files may be loaded directly (`torch.load` / file download) and converted
  to PyG `Data`, but the `pygod` package is never imported.
- UNPrompt is **DGL-based**; it is cloned only as a data source (Disney/Weibo) and a
  correctness reference. The UNPrompt scoring mechanism is **reimplemented in PyG** so all
  methods run in one harness (Approach A).
- Light datasets, 100 epochs, CPU-friendly, full-batch.

## 3. Model — MLP-GAD (Masked Latent Predictability)

One-class. All PyG.

| Component | Spec |
|---|---|
| Input normalization | Coordinate-wise z-score per feature dimension |
| Context encoder `f_θ` | 2-layer GNN (GraphSAGE default), runs on graph with **target node v's features masked** (mask token) |
| Target encoder `f_ξ` | **Per-node MLP** on clean `x` (no aggregation → preserves node individuality), **EMA** of an online MLP, stop-grad |
| Predictor `g_φ` | MLP: `z_ctx[v]` → predicted target latent of v |
| Residual | `r_v = 1 − cos(g_φ(z_ctx[v]), sg(f_ξ(x)[v]))` |

**Training (100 epochs):** minimize `mean_v r_v` over all nodes (one-class assumption:
bulk is normal). EMA-update target each step. Grad clip. **No margin, no pseudo-anomaly
injection.** Collapse avoided by EMA+predictor asymmetry (BYOL-style) plus the masked
prediction task being non-trivial.

**Scoring:** `r_v` → **rank-normalize** to percentile ⇒ fixed direction, scale-free.
Anomaly = high residual rank.

### Ablation toggles (same model, config flags)

| Toggle | Tests diagnosed cause |
|---|---|
| `mask` on/off | anti-smoothing effect (cause 2) |
| `target` = MLP-latent vs GNN | target smoothing leakage (cause 2) |
| `rank_calib` on/off | sign stability across datasets (calibration) |

## 4. Harness & repository layout

New project, no pygod:

```
mlpgad/
  data/
    loaders.py        # inj_cora (Planetoid + injection), disney, books → PyG Data
    normalize.py      # coordinate-wise z-score
  models/
    mlpgad.py         # Section 3 model (context/target/predictor, EMA, mask)
    unprompt_baseline.py  # UNPrompt-score reimplemented in PyG, single-graph mode
  train.py            # one-class 100-epoch loop, EMA, grad clip
  score.py            # residual → rank-normalize
  eval.py             # AUC/AP via sklearn (NOT pygod.metric)
  run_poc.py          # grid {dataset × method × toggle × seed} → CSV
  configs/            # yaml per dataset
../UNPrompt/          # cloned for Disney/Weibo data + reference only (DGL, not run by us)
```

### Datasets (3)

- **inj_cora** — built in PyG: Planetoid Cora + standard injection (structural cliques +
  contextual feature swap). Sanity check; model must exceed 0.5 here.
- **disney** — taken from the cloned UNPrompt repo, converted to PyG `Data`.
- **books** — downloaded as a raw data file (URL + `torch.load`), converted to PyG `Data`.
  **Fallback:** if books sourcing is blocked, substitute **weibo** (present in UNPrompt repo).

### Baselines (same harness, single-graph, same seed/split)

1. Random = 0.5 (reference line).
2. **UNPrompt-score reimplemented** — neighborhood-predictability of the **raw normalized
   attribute**: context = neighborhood with center node excluded/masked (intrinsic to
   UNPrompt), target = raw normalized attribute, **no EMA, no latent target**. This is the
   prior-art point; the PoC's delta is EMA latent target instead of raw attribute.
   Single-graph mode.
3. **MLP-GAD (PoC)** + toggles.
4. Reference only (not a main column): CONADJEPA's old organic numbers (disney 0.30–0.41)
   for qualitative contrast — different harness.

### Protocol

100 epochs, 5 seeds, full-batch, CPU. Report **AUC, AP (mean ± std)** per dataset.

## 5. Success criteria (pre-registered, falsifiable)

- **C1 (alive):** on disney **and** books, mean AUC **> 0.55** (CONADJEPA is below 0.5).
- **C2 (no sign inversion):** a single fixed `score_direction` yields AUC > 0.5 on all 3
  datasets (rank-calibration should guarantee this).
- **C3 (sanity):** inj_cora mean AUC **> 0.65**.
- **C4 (anti-smoothing works):** `mask`-on ≥ `mask`-off on ≥ 2 of 3 datasets.

**Decision:** C1 + C2 pass → idea worth pursuing; proceed to differentiate vs UNPrompt.
Otherwise → diagnosis wrong or one-class predictability insufficient; stop early.

## 6. Execution

After this spec + an implementation plan, code is delegated to **codex** (skill
`codex:rescue` / Agent) task-by-task (loaders → model → train/score → eval → run), each
reviewed before moving on.

## 7. Out of scope (YAGNI)

- Generalist / zero-shot cross-graph transfer (that is the *next* phase if PoC passes).
- Dynamic / temporal graphs.
- Reconstruction decoder, contrastive heads, uncertainty weighting (all removed).
- Beating UNPrompt's reported numbers (PoC only checks the signal is alive).
- Mini-batching / neighbor sampling (datasets are light, full-batch).
```
