# FINDINGS: Chasing the SOBDN Idea

Honest writeup of what happened when the "brain grows its own hardware"
idea was actually built and run, not just designed in a chat. Three
scripts, three separate questions. Numbers below are from real runs
(`outputs/*.log`, `outputs/*.png`), not projections.

## TL;DR

The idea is buildable and the mechanism is genuinely alive -- neurons
grow, wire, spike, move a body, and reward-modulated plasticity is
mathematically doing something. But it hits two real walls fast:

1. **A performance wall.** Pure-Python/NumPy, per-neuron-per-synapse
   simulation becomes impractical (<20 cycles/s) somewhere around a few
   hundred neurons, not the "10,000 neurons" the original design chat
   guessed at -- an order of magnitude earlier than predicted.
2. **A carrying-capacity wall.** The nutrient-based population control
   doesn't actually cap population at this grid size/regen rate within
   thousands of cycles -- it grows roughly exponentially, which is also
   what drives wall #1.

And one clean negative result:

3. **Undirected structural growth reliably builds a *connected* graph,
   but not reliably a *functional* one.** Over a full 20,000-cycle run
   with population deliberately held stable at 150, distance-to-food got
   48% *worse* from the first half of the run to the second half, and
   zero food was ever eaten. Whether that's "needs more scale" or "needs
   a fundamentally different growth rule" is still open -- see below --
   but at the scale actually tested, the answer is no, it doesn't learn.

## Two bugs found by running the code, not by reading it

Both of these were introduced during the rewrite, caught by actually
executing the sandbox and looking at *why* nothing was happening, not by
inspection. Recorded here because they're the kind of failure that looks
identical to "the idea doesn't work" from the outside -- exactly the
failure mode the original Gemini code had (see `engine.py` docstring,
bug #2: a script that runs to completion and prints a nice narrative
while doing nothing).

- **Growth restricted to interneurons.** First pass only let
  `kind == "inter"` neurons initiate new synapses. Sensors could never
  grow an outgoing axon, so a sensor->motor bridge was structurally
  impossible no matter how long it ran. 2000 cycles, population 449,
  2398 edges grown, zero connectivity from any sensor. Fixed by letting
  every neuron kind attempt growth; only *mitosis* stays interneuron-only
  (sensors/motors are body-anchored, not a population that should
  multiply).

- **Sensory drive too weak to ever cross threshold.** With the initial
  gain constants, net sensor input at a typical 15-20 voxel range came
  out to ~0.08 -- and because food scent and hazard scent were combined
  with comparable weight, at some spawn geometries they nearly
  cancelled. Steady-state membrane potential asymptotes *below*
  threshold for a constant sub-threshold input, so this isn't "hasn't
  fired yet," it's "mathematically will never fire." Confirmed directly:
  zero fires anywhere in a 60-neuron population over 100 sampled cycles.
  Fixed by raising sensor gain and reducing the hazard term's
  cancellation weight (`SENSOR_GAIN`, `HAZARD_SENSOR_FACTOR` in
  `engine.py`).

## Wall: topology forms, function doesn't (without help)

Even after both fixes, with a real sensor->motor path present (1-2
hops, confirmed via BFS) and sensors firing, the motors never fired.
Diagnosis: every single presynaptic neuron feeding either motor had a
**0% fire rate** over a 100-cycle sample, despite the path existing.
Population-wide mean fire rate was 4.85%. Random undirected growth had
built a *connected* graph, but nothing forced any specific edge on that
path to involve a neuron excitable enough to relay a ~0.05-0.2 weight
signal through the leak/threshold dynamics.

This is the single most interesting finding: **structural connectivity
and functional signal-passing are different properties**, and the
original design's growth rule only targets the first one. Real
developing nervous systems solve this with spontaneous activity --
retinal waves before the eyes open, embryonic motor twitching -- that
gives circuits something to Hebb-strengthen before any task-relevant
signal exists. The original Gemini V0.1 sketch actually had a crude
version of this (`if random.random() < 0.1: n.fired = True`) and it was
dropped in the "Genesis" rewrite; re-adding it here as a per-neuron,
genome-controlled `spontaneity` rate (mutable, so selection can shape it)
is what got the agent moving at all. Without it, the baseline run's
motor fire count was 0/100 for every window sampled, forever, regardless
of population size.

## Wall: population does not reach homeostasis

`run_experiment.py`, uncapped population, seed=1:

| cycle | population | edges | mean energy | dist-to-food | step time |
|------:|-----------:|------:|------------:|-------------:|----------:|
| 500   | 43         | 233   | 11.0        | 20.8          | 4.5 ms |
| 1000  | 104        | 552   | 12.8        | 26.8          | 13.7 ms |
| 1500  | 220        | 1170  | 12.8        | 28.6          | 19.2 ms |
| 2000  | 445        | 2301  | 11.7        | 27.6          | 43.1 ms |
| 2500  | 786        | 3984  | 10.9        | 37.0          | 101.5 ms |
| 3000  | 1283       | 6303  | 10.5        | 36.5          | 142.7 ms |
| 3500  | 1899       | 9486  | 10.2        | 36.2          | 257.8 ms |
| 4000  | 2696       | 13411 | 9.9         | 33.2          | 585.8 ms |

(Run manually stopped after cycle 4000 -- population and per-cycle cost
were still compounding with no sign of leveling off, and the trend was
already unambiguous; see `outputs/baseline_run.log`.)

Population growth is compounding (~1.5-2x every 500 cycles) with no sign
of leveling off before the run became impractically slow to continue.
Mean energy per neuron stays roughly flat around 10-13 the whole time --
the nutrient field is providing *just enough* ambient income to keep
mitosis firing steadily, because a 40^3 grid gives a freely-expanding
population plenty of unclaimed nutrient to spread into. The carrying
capacity mechanism is real (crowding a voxel does split its nutrient),
but at this grid size relative to population density, the population
never gets dense enough anywhere to actually feel it before the
simulation itself becomes the bottleneck. `food_eaten` stayed at 0 the
entire run; distance-to-food drifted between 20 and 37 with no
directional trend -- consistent with an undirected random walk, not
learned approach.

## Wall: performance (quantified against the original chat's own prediction)

The design conversation predicted "if you run this with 10,000 neurons,
Python will lag." `scaling_test.py` measures this directly, holding
population fixed (`freeze_population=True`) to isolate per-cycle cost
from the growth-runaway above.

Spatial-hash neighbor search (the fix for the original's O(N^2) growth
search):

| population | edges | ms/cycle | cycles/s |
|-----------:|------:|---------:|---------:|
| 55         | 218   | 8.7      | 115.7 |
| 105        | 425   | 17.9     | 55.9 |
| 205        | 858   | 52.7     | 19.0  *(below 20 cycles/s)* |
| 405        | 1699  | 153.6    | 6.5 |
| 805        | 3207  | 414.0    | 2.4 |
| 1605       | 6108  | 1609.0   | 0.6 |

The actual wall shows up around **200-400 neurons**, not 10,000 -- a full
order of magnitude earlier than the original design chat's own guess.
Note this benchmark seeds all neurons in one instantaneous dense cluster
(worst case for any spatial partition), whereas the organically-grown
population in `run_experiment.py` spreads out gradually and is
measurably cheaper at matched population size (786 neurons: 101.5 ms/cycle
organically grown vs. ~414 ms/cycle synthetically clustered here) --
so real usage sits somewhere between these two curves, but the
qualitative wall is the same order of magnitude either way. The
fixed-bucket-size spatial hash degenerates toward the naive scan when
everything is packed into a handful of voxels; an adaptive/octree
partition would close part of this gap but not all of it, since the
underlying cost is still O(edges) Python-level dict operations for the
reward-modulated weight update every cycle.

Matched-size comparison against a naive O(N^2) neighbor scan (both
benchmarks use the same instantaneous dense-cluster seeding):

| population | hashed | naive | speedup |
|-----------:|-------:|------:|--------:|
| 55         | 8.6 ms   | 9.0 ms   | 1.04x |
| 105        | 17.7 ms  | 18.8 ms  | 1.06x |
| 205        | 55.4 ms  | 52.5 ms  | 0.95x |
| 405        | 164.0 ms | 157.6 ms | 0.96x |
| 805        | 464.4 ms | 490.1 ms | 1.06x |

That is not a typo: the spatial hash gives essentially **no speedup at
all** in this seeding regime -- confirming the mechanism suspected above.
With `BUCKET_SIZE = 2.0` voxels and every neuron spawned within a ~2-voxel
Gaussian cloud of the same point, nearly the entire population lands in
a handful of buckets, so "check this neuron's bucket and its neighbors"
degenerates into "check almost everyone," same as the naive scan. A
spatial hash is only as good as the assumption that points are
spread out relative to its bucket size; a real, organically-grown
population *is* spread out (see the 786-neuron comparison above, where
the organic run is ~4x cheaper than either clustered benchmark at a
similar size), so the fix is real and matters in practice -- but a
synthetic worst-case benchmark like this one is exactly the wrong way to
show that, which is itself worth knowing before trusting a
microbenchmark over an end-to-end one.

## Answered: does it ever learn, given a stable population? No -- not at this scale.

`learning_wall_experiment.py` holds population at a hard cap of 150 (so
per-cycle cost stays cheap: 75.8 cycles/s average, whole run took 264s)
and runs for 20,000 cycles specifically to separate "population never
stabilizes" from "the learning rule doesn't work."

Completed run, seed=7:

| | first half | second half | verdict |
|---|---:|---:|---|
| mean distance-to-food | 20.85 | 30.90 | **got worse** |
| food eaten | 0 | 0 | no change |
| direct sensor->motor synapses | 0 | 0 | never formed |
| mean genome `v_thresh` | 0.877 | 0.863 | flat |
| mean genome `spontaneity` | 0.0137 | 0.0111 | **decreased** |

Zero food eaten across all 20,000 cycles. Distance-to-food did not
plateau noisily around a constant -- it trended *up* 48% from first half
to second half, i.e. the agent ended up further from food on average
later in the run than earlier. No direct sensor->motor synapse ever
formed in 20,000 cycles (only indirect multi-hop paths, and per the
baseline run those paths mostly don't carry live signal either -- see
above).

The genome drift is the most interesting secondary result: mean
`spontaneity` *fell* by ~19% over the run, the opposite direction you'd
want for bootstrapping exploration. This makes sense once you track the
individual incentives: `spontaneity` costs a neuron `FIRE_COST` in
energy every time it triggers a spike that wouldn't otherwise have
happened, but the reward payoff for that spike is rare, stochastic, and
often credited to whichever neuron happens to be spatially near the
agent when reward lands -- not reliably the neuron that actually fired.
Individual-level energy selection can therefore work *against* the
population-level exploration the mechanism depends on to ever discover
a useful circuit in the first place. That's a real tension in the
design, not a tuning accident: the same reward-modulated Hebbian rule
that (correctly) requires local coincidence to assign credit also has no
way to reward "useful variance" versus "wasted energy," so cheap,
cautious, quiet neurons are individually favored right up until the
population-level task actually needs a noisy one.

Reading this honestly: the credit-assignment chain here is long --
scent -> sensor spike -> Hebbian coincidence with a downstream neuron ->
that neuron eventually relaying to a motor -> motor movement -> distance
change -> reward deposited -> diffused back to the *specific* synapses
that were eligible -- and every link has to work in the same
few-hundred-cycle window for a single reward event to teach anything.
With only 3 sensors, 2 motors, and a population capped at 150, the odds
of that whole chain lining up by chance are low, and nothing in the
design increases those odds over time (no directed exploration bias
toward under-explored regions, no curriculum, no increasing plasticity
for young/novel synapses). This is very plausibly a scale-and-shaping
problem rather than a fundamental impossibility -- but "plausibly fixable
with more scale" is exactly the kind of claim this sandbox exists to
pressure-test rather than assert.

## What this suggests

- The reward-modulated (three-factor) plasticity piece is the strongest
  part of the original idea and it is implemented faithfully here --
  it just hasn't been given a chain short/easy enough to demonstrate it
  yet.
- The structural growth piece reliably does *something* (a connectome
  forms, population responds to energy) but "something" is not the same
  as "something useful," and nothing in the mechanism as designed
  closes that gap on its own.
- The performance ceiling is real and arrives early. Any next step that
  wants more population or more cycles needs vectorization (NumPy arrays
  over the whole population instead of a Python dict of objects) before
  it needs more biological features -- the same jump this repo's own
  `neuron.py` already made once, from a NumPy prototype to a
  GPU-accelerated one.

## If continuing this sandbox

In rough order of expected information gained per unit effort:

1. Let `learning_wall_experiment.py` run substantially longer (or with a
   larger population cap) before concluding the credit-assignment chain
   can't close -- 20,000 cycles with 150 neurons and 3 sensors is still a
   small sample of a large search space.
2. Add a small directed-exploration bias (e.g. higher initial synapse
   weight, or extra plasticity for young synapses) and see whether that
   alone is enough to bridge the gap -- this isolates "needs help
   bootstrapping" from "needs a different mechanism entirely."
3. Vectorize the neuron population into NumPy arrays (state per neuron
   as array columns, synapses as a sparse adjacency structure) before
   scaling population further -- the current dict-of-objects design is
   the direct cause of the performance wall, independent of any
   biological question.
4. Only after 1-3: revisit the visual cortex / "Memory Protein" pieces
   that were deliberately left out of this pass.
