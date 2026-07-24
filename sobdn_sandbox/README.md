# SOBDN Sandbox

A separate, deliberately unstable research sandbox for the "Self-Organizing
Bio-Digital Neuron" idea from a design conversation with Gemini: neurons
that grow their own topology via mitosis/apoptosis/chemotaxis, wired by
reward-modulated (dopamine/pain) plasticity instead of backprop.

This is **not** part of the production framework at the repo root
(`neuron.py` / `circuit.py`, tested and GPU-accelerated). It intentionally
lives in its own directory so it can be broken, rewritten, or deleted
without touching that code. The goal here isn't a finished feature -- it's
finding out where this specific idea actually holds up and where it hits a
wall, the same way the other experiments in `experiments/` each test one
concrete hypothesis.

See `FINDINGS.md` for what actually happened when this was run.

## What's here

- `engine.py` -- the core World/Neuron/Agent simulation. A rewrite, not a
  port, of the original design sketch -- see the module docstring for the
  concrete bugs the original had (list-index synapse targets that rot on
  apoptosis, spike propagation that silently only ran for 4 hardcoded
  neurons, O(N^2) neighbor search, hand-coded sensing that bypassed the
  entire "structure emerges from space" premise, etc.) and how each is
  fixed here.
- `run_experiment.py` -- baseline run: seed a small interneuron cloud
  around an embodied agent in a shared 3D voxel space, let it grow/learn
  under its own rules, see what happens. No hand-holding.
- `scaling_test.py` -- isolated performance benchmark: cycles/sec vs
  population size, spatial-hash neighbor search vs naive O(N^2), directly
  testing the original design's own prediction that "10,000 neurons will
  lag."
- `learning_wall_experiment.py` -- population held at a fixed cap
  (so per-cycle cost stays cheap) for a much longer horizon, to isolate
  one specific question from the population-growth wall: given a stable
  population, does reward-modulated growth ever actually learn to
  approach food, or does it stay a random walk indefinitely? Also tracks
  whether a sensor->motor path is merely *present* vs. actually *live*
  (see FINDINGS.md), and supports a `fixed_spontaneity` override.
- `spontaneity_ablation.py` -- runs the learning-wall experiment twice,
  identical seed, with spontaneity heritable/selected vs. pinned to a
  constant -- a direct test of whether individual-level selection eroding
  population-level exploration is the actual bottleneck (see FINDINGS.md;
  short answer: no, it isn't).
- `outputs/` -- logs, plots, and profiler output from actual runs.

## Running it

```bash
cd sobdn_sandbox
python3 run_experiment.py          # baseline: free-growing population
python3 scaling_test.py            # performance wall
python3 learning_wall_experiment.py  # learning wall (population capped)
python3 spontaneity_ablation.py    # does fixing exploration noise help?
```

Each script prints periodic status lines, explicit pass/fail-style
criteria at the end, and saves a plot to `outputs/`.

## What's deliberately out of scope for this first pass

The original design conversation also covered a topographic visual
cortex (retinotopic mapping + lateral inhibition) and a "Memory Protein"
framed as a CaMKII analog. Both are left out here. The memory/adaptation
variable that *is* implemented (`Neuron.mp` in `engine.py`) is labeled
for what it mechanically is -- spike-frequency adaptation, a second
slower leaky variable gating the firing threshold -- rather than given a
biological name it doesn't need to justify its existence. Vision is a
separate, later experiment if the core growth/learning loop here turns
out to be worth building on.
