# Self-Organizing Neuron Unit — End-to-End Spec v2

**Date:** 2026-07-23
**Status:** 🧪 Design spec — not yet implemented
**Scope:** Incorporates cross-domain research (SORN, Physarum, clonal selection, SMGrNN, self-referential weight matrices) and a stress test of the v1 design.

Design law unchanged: structure and content factorized, one unit type with four toggleable properties, fixed-capacity pool + masking for GPU batchability.

---

## 0. Per-Unit State

* `a` — activation
* `e` — eligibility trace
* edge list, each edge tracking: weight `w_k`, rolling mean update `μ_k`, rolling variance `σ_k²` over a window of `T` steps
* gene vector `g = [λ_edge, τ_w, τ_Δ, clone_affinity_threshold, η]` — local hyperparameters, the target of the self-evolving pathway
* `f` — local fitness/affinity signal (recent local prediction error, inverted)

**Engineering constraint carried over from v1:** pre-allocate a max-capacity pool, mask dead/ungrown units. Do not literally resize tensors mid-training.

---

## 1. Self-Connecting (Structure Pathway)

### Primary rule — instability-triggered growth (from SMGrNN, arXiv:2512.12713)

For edge `k`, over a sliding window of `T` steps:

```
μ_k = mean(Δw_k over window)
σ_k² = var(Δw_k over window)
```

Flag edge `k` unstable if:

```
|μ_k| < 0.5 · σ_k   AND   σ_k² > λ_edge · |μ_k|
```

On flag: insert a relay node on a parallel two-step path (i → relay → j), small-magnitude init weights, keep the original edge. This targets capacity where gradient signal is persistently undecided, not just where activity is high.

### Secondary — random exploratory growth

Bernoulli trigger (prob `p_rand`) proposing `ρ_rand · N` new random edges, to escape local structural traps that edge-driven growth alone can't reach (pure reactive growth only edits edges that already exist).

### Pruning — mandatory, not optional

Remove edges where `|w_k| ≤ τ_w AND |μ_k| ≤ τ_Δ`; delete only a random fraction `η_prune` of the flagged set per prune step (period `s` steps), not all at once. Remove orphan nodes (zero in- or out-degree) after pruning.

Evidence this is load-bearing: a growth-only ablation with pruning disabled matched growth+prune reward almost exactly but at ~100x the parameters (6,692 vs 84 on CartPole; two orders of magnitude on the other two benchmark tasks tested).

### Alternative edge-weight rule worth a head-to-head, not a default — Physarum flux rule

Edge conductance tracks recent signal flux and decays with disuse, same shape as the SMGrNN mean/variance rule but derived from slime-mold transport dynamics instead of gradient statistics. Has a real convergence proof (converges to the shortest path for any topology, any initial condition) — but only for specific values of the model's power-law exponent and zero tube-saturation; it is not a universal guarantee. A differentiable version exists (usable as a trainable layer), so this is implementable, not just an analogy.

### Homeostasis (SORN) — required if the structure pathway is Hebbian-driven

Synaptic normalization (keep each unit's total afferent weight roughly constant) + intrinsic plasticity (adjust firing threshold to hold a target average activity). Without this pairing, Hebbian-driven growth is the exact mechanism that either bursts or dies out.

---

## 2. Self-Learning (Content Pathway)

Split by pathway, unchanged from the prior turn's reasoning, sharpened:

* **Content:** gated delta-rule (`S_t = S_{t-1} + β_t(v_t − S_{t-1}k_t) ⊗ k_t`), already validated in-project for associative recall (MQAR K=32, ~100% vs GRU's 13%). Reuse the existing chunk-parallel WY-form implementation.
* **Structure-pathway plasticity:** local three-factor Hebbian (`Δw = η · pre · post · M`).
* **Modulator M:** per-unit local prediction error — reuse the already-validated non-backprop predictive coding mechanism (within ~4pp of backprop), not a newly invented reward signal. RM-SORN is direct precedent for pairing local Hebbian plasticity with exactly this kind of modulator.

---

## 3. Self-Evolving

### Primary — self-referential update, not population mutation

The gene vector `g` is updated by the same delta-rule mechanism already validated for content learning, applied reflexively to `g` itself (outer-product/delta-rule self-modification, per Irie/Schlag/Schmidhuber's scalable self-referential weight matrix). Target signal: recent trend in the unit's own local prediction error. This is in-lifetime, differentiable, single-run — and reuses code already proven on this project, rather than requiring new machinery.

### Comparison arm only — NEAT-style population mutation

Slower, offline, population-level; keep as a baseline to compare against in the ablation ladder, not as the shipped default.

> **Flag:** self-referential update of a small gene vector (rather than a full weight matrix) is untested territory — no literature precedent found for this specific target. This is the single highest-uncertainty piece of the whole spec.

---

## 4. Self-Replicating

### Primary — clonal selection (affinity-proportional cloning + hypermutation)

```
clone_rate(unit) ∝ f(unit)              # local affinity/fitness, above clone_affinity_threshold
mutation_scale(clone) ∝ 1 / f(parent)   # well-performing parents mutate less
```

Clone copies parent's gene vector `g` with hypermutation applied only to the clone, wires locally using the same edge-growth rule as the connectivity pathway. This is the standard CSA mechanism (de Castro & Von Zuben), not an ad hoc spawn-on-error-threshold rule.

**Death:** units with sustained low affinity and stalled updates are orphan-removed, symmetric to the edge/node pruning above.

**Cap:** population budget bounded by the fixed-capacity pool.

---

## 5. Stress Test — Cross-Pathway Interaction Risks

These did not show up when auditing any one pathway alone; they emerge only from the combination.

**(a) Structure chasing a moving target.** Content-pathway learning (delta-rule) continuously shifts activations; the structure pathway reads those same activations as "coactivity" or "instability." Structure could keep rewiring against representations that haven't settled, never converging. *Mitigation:* gate structural edits to a slower clock than content learning — periodic (every `s` steps), not continuous, as SMGrNN already does. Consider requiring a local-loss-plateau signal before allowing large structural edits.

**(b) Undefined priority when signals collide.** If local prediction error drives both edge growth and clone-triggering, a single noisy/high-error unit can grow new connections AND spawn a mutated clone in the same window — compounding rather than isolating the response to error. Currently undecided — needs an explicit tie-break rule before implementation, not left implicit. *Candidate:* growth is the default response to instability; replication requires sustained (not instantaneous) high affinity over a longer window, so the two operate on different timescales rather than the same trigger.

**(c) Homeostasis vs. replication.** Synaptic normalization assumes a roughly fixed unit count. Replication changes that count, which shifts every unit's normalization target, which can suppress the very activity/fitness signal replication is reading. None of the source mechanisms have this problem individually — SORN doesn't replicate units; clonal selection algorithms don't carry synaptic homeostasis. This is a genuinely new instability mode created by the combination, not inherited from any one part. *Mitigation:* re-normalize on a schedule tied to population-change events, not continuously — and treat this as a first-class thing to look for in the ablation below, not an assumed non-issue.

**(d) Ablation confound.** Because properties are toggled together, "growth improves stability" (SMGrNN's finding) could get misattributed to the full system if replication rides along untested. The two mechanisms that change the size of the state space (connect, replicate) are the actual novel risk — not just weight values — and need to be isolated from each other, not just from the static baseline.

---

## 6. Revised Ablation Ladder

Linear ladder (as before), plus a small factorial insert to catch (c) and (d):

```
static
  → +connect (SMGrNN rule)
  → +connect +content-learn (delta-rule)
  → +connect +content-learn +structure-learn (Hebbian + M)
  → +...+evolve (self-referential)
  → +...+replicate (clonal)
```

**Insert before the full ladder:** a 2×2 factorial on `{connect, replicate}` alone (content-learn off, evolve off) — isolates whether population-size change interacts badly with structural growth before either is combined with anything else.

**Protocol** (unchanged from prior discipline):

* Non-stationary task (a fixed task gives structural/replication mechanisms nothing to earn their keep over a static net)
* 6-8 seeds, matched step budget
* Per-seed trajectories reported, not means (delayed/unreliable escapes are a known failure mode in this project's own findings)
* Hysteresis on grow/prune/clone thresholds to prevent flapping

---

## 7. Open / Deferred, Honestly Flagged

* Physarum flux rule vs. SMGrNN instability rule as the primary edge-growth law — genuinely open, worth a small head-to-head before committing to one.
* Priority rule for simultaneous growth/replication triggers (risk b) — undecided, must be fixed before implementation, not discovered empirically after the fact.
* Self-referential gene-vector update — no precedent at this scale/target; treat as the riskiest single component and test it in isolation first.

---

## References

* Lazar, Pipa & Triesch, "SORN: a self-organizing recurrent neural network," *Frontiers in Computational Neuroscience*, 2009. [frontiersin.org/articles/10.3389/neuro.10.023.2009/full](https://www.frontiersin.org/articles/10.3389/neuro.10.023.2009/full)
* Aswolinskiy & Pipa, "RM-SORN: a reward-modulated self-organizing recurrent neural network" (via Science.gov topic aggregation). [science.gov/topicpages/s/self-organizing+recurrent+neural.html](https://www.science.gov/topicpages/s/self-organizing+recurrent+neural.html)
* Tero, Kobayashi & Nakagaki (2007) Physarum tube-adaptation model; Bonifaci, Mehlhorn & Varma, "Physarum Can Compute Shortest Paths," *J. Theoretical Biology* 309, 2012. [arxiv.org/abs/1106.0423](https://arxiv.org/abs/1106.0423)
* Straszak & Vishnoi, "Physarum Powered Differentiable Linear Programming Layers and Applications." [arxiv.org/pdf/2004.14539](https://arxiv.org/pdf/2004.14539)
* de Castro & Von Zuben, "The Clonal Selection Algorithm with Engineering Applications," GECCO 2000. [en.wikipedia.org/wiki/Clonal_selection_algorithm](https://en.wikipedia.org/wiki/Clonal_selection_algorithm)
* Jia & Zhou, "Self-Motivated Growing Neural Network for Adaptive Architecture via Local Structural Plasticity" (SMGrNN), arXiv:2512.12713, Dec 2025. [arxiv.org/pdf/2512.12713](https://arxiv.org/pdf/2512.12713)
* Imam & Cleland (2020), neurogenesis-inspired lifelong learning on Intel Loihi. [pmc.ncbi.nlm.nih.gov/articles/PMC10194827](https://pmc.ncbi.nlm.nih.gov/articles/PMC10194827/)
* Irie, Schlag, Csordás & Schmidhuber, "A Modern Self-Referential Weight Matrix That Learns to Modify Itself," ICML 2022. [proceedings.mlr.press/v162/irie22b/irie22b.pdf](https://proceedings.mlr.press/v162/irie22b/irie22b.pdf)
* Kirsch & Schmidhuber, "Self-Referential Meta Learning," 2022. [openreview.net/pdf?id=adt25bANyfB](https://openreview.net/pdf?id=adt25bANyfB)
