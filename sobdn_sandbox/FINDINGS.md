# FINDINGS: Chasing the SOBDN Idea

Honest writeup of what happened when the "brain grows its own hardware"
idea was actually built and run, not just designed in a chat. Three
scripts, three separate questions. Numbers below are from real runs
(`outputs/*.log`, `outputs/*.png`), not projections.

## TL;DR

The idea is buildable and the mechanism is genuinely alive -- neurons
grow, wire, spike, move a body, and reward-modulated plasticity is
mathematically doing something. It hits real walls, but a review pass
(see "Follow-up" below) found that one of them was partly this
implementation rather than the mechanism, and traced a second one to a
more specific cause than originally stated:

1. **A performance wall -- real, but ~1.8x-4x of it was implementation
   overhead, not the mechanism.** The <20 cycles/s ceiling originally
   measured around 200 neurons was genuinely there, but profiling found
   most of the cost was numpy dispatch overhead on tiny per-candidate
   operations, not the neighbor-search algorithm itself (spatial hash
   vs. naive O(N^2) made almost no difference, which was the tell). Two
   small, behavior-preserving fixes pushed the wall out by 1.8x-4x+
   depending on scale. "200-400 neurons is where the idea breaks down"
   is no longer a claim the numbers support -- see point 2 below in the
   Follow-up section.
2. **A carrying-capacity wall.** The nutrient-based population control
   doesn't actually cap population at this grid size/regen rate within
   thousands of cycles -- it grows roughly exponentially, which is also
   what drove the original performance measurements.
3. **Connectivity forms almost immediately; liveness essentially never
   does.** A sensor->motor path exists at 100% of logged checkpoints in
   every run tested, often within the first few hundred cycles -- but
   it carries a real, live signal at only ~30-40% of checkpoints, most
   of those barely above the detection threshold, and this pattern
   holds whether or not exploration noise is artificially kept high
   (see point 3 in Follow-up). This, not population size, looks like
   the actual bottleneck.
4. **Net result: it doesn't learn, at the scale actually tested.** Over
   a full 20,000-cycle run with population held stable at 150,
   distance-to-food got worse (not just flat) from the first half of the
   run to the second, and zero food was ever eaten -- reproduced twice,
   with and without heritable exploration noise. Whether that's "needs
   more scale" or "needs a fundamentally different growth rule" is still
   open, but "needs more noise" has now been tested and ruled out.

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

`run_experiment.py`, uncapped population, seed=1 (numbers below are from
the post-optimization engine -- see Follow-up point 2 -- so step times
are ~3x faster than they were when this was first measured; the
population/edges/energy/distance trajectory itself is bit-identical
either way):

| cycle | population | edges | mean energy | dist-to-food | path (hops, live) | step time |
|------:|-----------:|------:|------------:|-------------:|:------------------:|----------:|
| 500   | 43         | 233   | 11.0        | 20.8          | 1, 0.000 | 2.5 ms |
| 1000  | 104        | 552   | 12.8        | 26.8          | 1, 0.000 | 8.4 ms |
| 1500  | 220        | 1170  | 12.8        | 28.6          | 1, 0.000 | 6.4 ms |
| 2000  | 445        | 2301  | 11.7        | 27.6          | 1, 0.000 | 15.6 ms |
| 2500  | 786        | 3984  | 10.9        | 37.0          | 1, 0.000 | 31.7 ms |
| 3000  | 1283       | 6303  | 10.5        | 36.5          | 1, 0.000 | 50.4 ms |
| 3500  | 1899       | 9486  | 10.2        | 36.2          | 1, 0.000 | 87.1 ms |
| 4000  | 2696       | 13411 | 9.9         | 33.2          | 1, 0.000 | 186.0 ms |

This run now completes cleanly (it originally had to be stopped early --
see Follow-up point 2 for why). The direct sensor->motor edge this run
keeps finding is present and unchanging at 1 hop from cycle 500 onward,
and its liveness reads exactly **0.000 at every single one of the 8
checkpoints** -- the starkest version of the "sprawling, not reaching"
pattern in this whole document: population grew 63x, edges grew 58x,
and the one connection that would matter never once fired end-to-end.

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

## Follow-up: three things worth checking rather than trusting

A review of this document caught three real gaps -- one plot that hadn't
been verified against what it actually measures, one conclusion drawn
from a benchmark without profiling it first, and one mechanism-level
hypothesis (the spontaneity tragedy-of-the-commons) that was diagnosed
but never actually tested by removing it. All three were checked, not
just discussed.

### 1. The hop-count metric was real BFS, but not diagnostic on its own

The concern: a "sensor->motor hops" panel pinned at exactly 1.00 for
every logged cycle looked like it might be measuring something trivial
(e.g. "length of a single synapse," which actually would be 1 by
definition) rather than a real path check.

Checked: `sensorimotor_path()` is a genuine BFS from the sensor set to
the motor set, not a tautology -- confirmed by reading it and by the
capped run below, where it reports 2 hops, not 1. But the underlying
instinct was right for a different reason than expected: in *this*
rewrite, sensors and motors are body-anchored close together (a few
units apart, moving together with the agent), unlike the original
design's fixed sensors-at-X=0/motors-at-X=29 layout. That makes a 1-hop
path geometrically cheap to form almost by chance, independent of
whether the mechanism is doing anything intelligent -- so hop-count
alone was never going to be very diagnostic here, correct concern, just
not the "trivial by definition" mechanism suspected.

The actual gap: liveness (does the path carry a real signal) was
checked by hand once during debugging (0% fire rate on relay neurons)
but was never tracked over time or checked at all in the capped
(`learning_wall_experiment.py`) run -- the FINDINGS text above says
"only indirect multi-hop paths" for that run without ever having
verified it. Fixed: `sensorimotor_path()` now returns the actual path,
and a new `path_liveness()` reports the mean firing rate of the
interior relay neurons (or the eligibility trace, for a direct edge).

Rerunning the full 20,000-cycle capped experiment with this tracked
(`outputs/spontaneity_ablation.log`, run A): a 2-hop path is present at
**every single one of the 40 logged checkpoints** -- confirming the
"cheap to form geometrically" read -- and liveness is 0.000 at all but
a handful of them (two small blips of 0.002-0.003 around cycles
12,500-15,000, and one flicker of 0.162 at the very last checkpoint).
Overall, 40% of logged checkpoints cross a >0.01 liveness threshold at
all, but essentially none of that shows up as improved food-seeking (see
point 3). So the precise version of the original finding is: a path
forms almost immediately and persists throughout, it flickers live more
often than a single spot-check would suggest, but that intermittent
liveness isn't translating into anything the reward signal can lock
onto. "Sprawling, not reaching" was the right call, with "sometimes
twitching" added as a real nuance the single fire-rate spot-check had
missed.

### 2. The spatial hash really doesn't help here -- and it isn't neighbor search that's slow

The concern: hashed and naive O(N^2) neighbor search overlapping almost
exactly in `scaling_test.png` suggests the ~200-400 neuron performance
wall might belong to this implementation, not the underlying mechanism
-- worth profiling before treating the scaling numbers as a verdict on
the idea itself.

Checked, with `cProfile` at population 800 (`outputs/profile_cumulative.txt`):
confirmed directly. The full-grid chemical diffusion step (the other
plausible fixed-cost suspect) doesn't even appear in the top 15 by
cumulative time. What dominates is `_attempt_growth`'s candidate-scoring
loop: 91% of total per-cycle time, and within it, the two largest
individual costs were `_voxel()`'s `np.clip` on a 3-element array
(called ~27,700 times/cycle, 5.4s of the 13.9s profiled window) and
`np.linalg.norm` on 3-element vectors (~69,600 calls/cycle, 3.8s),
computed twice per candidate -- once to filter it in `_nearby_neurons`,
again to score it in `_attempt_growth`. Neither cost is inherent
arithmetic; both are numpy's generic-dispatch overhead applied to
operations tiny enough that plain Python is faster.

Fixed both (pure-Python clip/truncate in `_voxel`; a `math.sqrt`-based
`_dist()` helper; `_nearby_neurons` now returns `(neuron, distance)`
pairs so the distance is computed once, not twice) and reconfirmed
**bit-identical simulation output** on a 300-cycle deterministic replay
before trusting the numbers. Result, same population sizes as the
original scaling test:

| population | before | after | speedup |
|-----------:|-------:|------:|--------:|
| 55         | 8.6 ms   | 4.8 ms   | 1.8x |
| 205        | 55.4 ms  | 18.7 ms  | 3.0x |
| 805        | 464.4 ms | 138.1 ms | 3.4x |
| 1605       | 1947.0 ms| 477.8 ms | 4.1x |

So: **the wall was real, but a meaningful chunk of it -- and a growing
share as population increases -- was this implementation, not the
mechanism.** Two small, behavior-preserving fixes bought 1.8x-4x+,
pushing the usability threshold (20 cycles/s) from ~200 neurons to
somewhere between 200-800. Re-profiling after the fix shows the new
dominant cost is the raw number of Python-level candidate evaluations
per growth attempt (still ~46,000/cycle at population 800) -- which is
a real cost of this benchmark's dense-cluster seeding combined with a
dict-of-Python-objects architecture, not evidence about the SOBDN
mechanism's intrinsic compute demand. That distinction still hasn't
fully resolved -- it would take the vectorization pass already listed
below to find out where the wall sits once the implementation stops
being the bottleneck -- but "200-400 neurons is where the *idea* breaks
down" is no longer a claim these numbers support.

### 3. The spontaneity ablation: does removing the tragedy-of-the-commons actually help?

The proposal: if individual-level energy selection eroding
population-level spontaneity is really what's smothering exploration,
pinning `spontaneity` to a fixed constant (heritable no longer, immune
to selection) should let the capped 20,000-cycle test learn where the
original didn't. If it still doesn't learn, the blocker is deeper --
most likely the missing live bridge from point 1 above.

Added `fixed_spontaneity` as a `World` constructor override (applied at
spawn and at mitosis, bypassing mutation), and reran the identical
seed=7, population-150, 20,000-cycle experiment with it pinned at 0.02
(comfortably above the ~0.011 the original run's selection converged
to). Both conditions logged with the same path-liveness tracking from
point 1, for a direct comparison.

Completed, same seed=7 for both:

| metric                          | A: evolvable | B: fixed at 0.02 |
|----------------------------------|--------------:|------------------:|
| dist-to-food, first half         | 20.85         | 27.87             |
| dist-to-food, second half        | 30.90         | 37.43             |
| change (negative = improved)     | +10.04        | +9.55             |
| total food eaten                 | 0             | 0                 |
| % logged cycles with ANY path    | 100.0%        | 100.0%            |
| % logged cycles with LIVE path   | 40.0%         | 30.0%             |

**Verdict: fixing spontaneity did not rescue learning.** Both
conditions get worse by essentially the same amount (+10.04 vs +9.55),
neither ever eats food, and the fixed-spontaneity run's live-path
fraction is if anything slightly *lower* (30% vs 40%) despite constant
exploration noise. One real difference did show up: run B is the only
one of the two that ever grew a direct (1-hop) sensor->motor synapse at
all -- it appears at cycle 12,500 and its weight climbs from the
default ~0.05-0.2 spawn range to 0.205 by the end, a small but genuine
sign of reward-modulated reinforcement happening on *some* edge. It
just never turns into eaten food or a shrinking distance-to-food trend.

So: the tragedy-of-the-commons dynamic is real (spontaneity does drift
down under individual selection, as documented above), but it is not
*the* bottleneck -- removing it doesn't unlock learning. That leaves the
missing live bridge (point 1) as the better-supported explanation:
guaranteed exploration noise gives you more scattered activity, not a
more reliable path from sensors to motors, and reward-modulated Hebbian
plasticity has nothing to reinforce when the path that would matter
almost never fires end-to-end. One caveat worth being honest about:
this is a single seed per condition, not a multi-seed statistical
comparison -- the 40% vs 30% live-path gap and the "which one is worse"
comparison could partly be seed noise rather than a real effect of the
spontaneity condition. The "neither learns, neither eats" result is
robust across both runs; the finer-grained comparisons between them are
suggestive, not proven.

## Second follow-up round: a control, a prediction, and the cheapest fix

Three more things checked -- one to verify a claim made above rather than
just assert it, one to test whether a design parallel actually holds, and
one to try the single cheapest, most standard fix available before
concluding anything further.

### 4. Zero-reward control: was the 0.205 weight really reward-driven?

The claim above ("a small but genuine sign of reward-modulated
reinforcement") was inference, not proof. Checked directly: reran the
identical fixed-spontaneity(0.02) condition with a new `disable_reward`
override that pins reward and pain to exactly 0.0 every cycle, making the
weight-update line (`dw = LR * trace * reward`) a mathematical no-op for
every synapse. Confirmed the chem field stays bit-exactly zero for 20,000
cycles. Result: **max direct sensor->motor weight ever observed: 0.0000**
-- no direct edge even formed in this run, let alone reached a weight
above the 0.2 ceiling that random initial growth alone can produce. That
confirms the claim: the 0.205 in the real run required reward-modulated
plasticity to happen at all; it isn't an artifact of structural growth.

Side effect worth flagging on its own: this run's distance-to-food
*improved* (16.13 -> 13.69) with reward completely off. That's not
learning -- there is no mechanism left that could produce it -- it's this
single random-walk trajectory happening to end up nearer a food position
that (since it's never eaten in this run) never moves from its seed=7
starting location. This is a useful caution: "distance-to-food improved"
is a noisy enough signal on a single trajectory that a true null
condition can pass it by chance. `food_eaten` is the more trustworthy
metric precisely because it can't be won by luck the same way -- and
it's zero in every condition tested in this document, including this one.

### 5. Does mp_beta erode under selection the way spontaneity did?

The hypothesis: mp_beta (the Memory Protein / intrinsic-adaptation gene)
is heritable and mutable exactly like spontaneity was, so individual
energy selection should erode it the same way. Reran the
evolvable-spontaneity condition (bit-reproducing the original run) with
mp_beta/mp_gamma now tracked. Result: mean mp_beta went from -0.1257 to
-0.1458 -- **more negative, i.e. more facilitating, the opposite
direction** from what the spontaneity parallel predicted.

Reading this honestly rather than declaring victory either way: the
magnitude is small relative to mp_beta's allowed range ([-0.6, 0.3]), and
this is one seed, so this could be ordinary genetic drift rather than
selection pushing in a real direction -- unlike spontaneity's erosion,
which was a clear, consistent ~19% relative move in one direction. A
plausible mechanistic reason the two genes would diverge: `spontaneity`
manufactures firing (and its `FIRE_COST`) out of nothing, a pure gamble
that's individually expensive with a rare payoff; `mp_beta` only shapes
how *already-arriving* input converts into a spike, so it doesn't create
new costly events on its own the way spontaneity does, and has less
reason to be punished the same way. Net effect on the plan: the specific
empirical prediction wasn't confirmed, but the underlying point stands on
different grounds -- if selection isn't clearly shaping mp_beta in either
direction (as opposed to clearly eroding it), evolutionary search isn't
reliably finding a good value for it either, which is itself an argument
for the proper fix (a fixed regulatory *rule* every neuron runs, per
Turrigiano, rather than a heritable *value* left to drift) over trusting
mutation and selection to land somewhere useful.

### 6. The cheapest fix: continuous potential-based reward shaping

Implemented Ng, Harada & Russell (1999) properly: `reward = max(prev_dist
- dist, 0) * GAIN`, `pain = max(dist - prev_dist, 0) * GAIN` -- moving
away deposits into the pain channel instead of being silently discarded,
so it's genuinely signed, not just a reward-only floor (an earlier draft
of this fix clamped the negative case to zero, which is a different and
weaker rule than what the shaping-invariance proof actually covers;
caught and fixed before running, confirmed via a bit-identical-output
regression check on the non-shaping code path). Tested on top of
fixed-spontaneity(0.02) for a clean, one-variable comparison against the
existing binary-shaping baseline:

| condition | 1st half dist | 2nd half dist | change | food eaten | % live path |
|---|---:|---:|---:|---:|---:|
| B: binary shaping (baseline) | 27.87 | 37.43 | +9.55 | 0 | 30% |
| C: continuous shaping | 18.66 | 21.73 | **+3.07** | 0 | **60%** |
| D: no reward (null control) | 16.13 | 13.69 | -2.44 (noise, see above) | 0 | 45% |

Continuous shaping is a real, measurable improvement over binary shaping
on both axes that matter -- distance-to-food degrades roughly a third as
much, and the fraction of checkpoints with a *live* path doubles to 60%,
the highest of any condition tested in this document. Denser reward
genuinely produces more live signal and less-bad drift. But it does not
solve the underlying problem: distance still trends up, not down, and
**food eaten is still exactly zero** -- across all four conditions in
this table, and every 20,000-cycle run in this entire document. That
last fact is the one result in this whole investigation that isn't
sensitive to which single trajectory got lucky: no amount of noise
tuning or reward-density tuning tried so far has produced one single
successful food-reaching event.

## Third follow-up round: a replication, a reconciliation, and a clean isolation

A second review pass caught one thing to reconcile (a plot whose trend
line and summary table read in opposite directions), one thing that
wasn't actually checkable from its own chart (fixed in round two's plot
but not yet re-rendered for the runs already committed), and one design
improvement on the mp_beta test: pinning spontaneity alone doesn't
isolate mp_beta, because v_thresh is still free to absorb the same
selection pressure. All three addressed.

### 7. The v_thresh pattern is real, consistent, and worth naming on its own

Pulled the exact numbers across every run in this document rather than
eyeballing plots:

| run | spontaneity | v_thresh: first -> last | delta |
|---|---|---:|---:|
| A (evolvable, original) | evolvable | 0.877 -> 0.863 | -0.014 |
| A2 (evolvable, replica) | evolvable | 0.877 -> 0.863 | -0.014 |
| B (fixed, binary shaping) | fixed 0.02 | 0.882 -> 0.923 | **+0.041** |
| C (fixed, continuous shaping) | fixed 0.02 | 0.875 -> 0.912 | **+0.037** |
| D (fixed, no reward) | fixed 0.02 | 0.870 -> 0.910 | **+0.040** |

This is a real, three-for-three replicated pattern, and it is more
decisive than any of the reward-related findings in this document: every
run with spontaneity fixed shows v_thresh climbing by a consistent
+0.037 to +0.041, *regardless of reward regime* (binary, continuous, or
none at all) -- confirming it's a metabolic/selection effect, not a
reward-learning effect. Every run with spontaneity evolvable shows
v_thresh flat-to-slightly-down, identically between A and its replica
A2. Read together with spontaneity's own erosion, this looks like one
underlying "reduce firing cost" selection pressure with more than one
lever to express itself through -- when spontaneity is free to fall, it
absorbs the pressure; when it's pinned, v_thresh climbs instead.

### 8. mp_beta, properly isolated

Pinning spontaneity alone doesn't isolate mp_beta -- v_thresh is still
free in that setup, and per point 7 it's the more consistent absorber of
this pressure. The existing mp_beta numbers with only spontaneity fixed
are, honestly, mixed rather than clean: D (spontaneity fixed, no reward)
moved mp_beta slightly *more* facilitating (-0.1295 -> -0.1326, delta
-0.0031), while C (spontaneity fixed, continuous shaping) moved it
notably *less* facilitating (-0.1277 -> -0.1167, delta +0.0110) -- the
same "spontaneity fixed" condition producing opposite-sign mp_beta drift
depending on reward regime. Not a clean signal.

Added `fixed_v_thresh` alongside `fixed_spontaneity` and reran with
*both* pinned (spontaneity=0.02, v_thresh=0.88), leaving mp_beta and
mp_gamma as the only evolvable excitability-related genes -- a clean,
unconfounded test. Result: **mp_beta barely moved: -0.1227 -> -0.1234,
a delta of -0.0007** -- an order of magnitude smaller than v_thresh's
consistent ~+0.04 climb, and smaller than any of the mixed
spontaneity-only-fixed results above. With the two known-mobile levers
locked down and mp_beta genuinely the last one standing, it still didn't
move. That's real evidence against trait substitution reaching mp_beta,
not just an absence of evidence for it -- supporting the original
mechanistic read (mp_beta shapes existing input into spikes rather than
manufacturing costly spikes from nothing, so it doesn't carry the same
per-spike energy tax spontaneity and, apparently, v_thresh's absence
does) over "it just hadn't had its turn yet."

The genome-drift panel in `learning_wall_experiment.py`'s `plot()` only
showed spontaneity and v_thresh even after mp_beta/mp_gamma were added to
`history` -- a real gap (a claim in this document that wasn't checkable
against its own chart). Fixed: the plot is now 3x3 with a dedicated
mp_beta/mp_gamma panel, and the A2/C/D charts already committed were
regenerated (deterministically, same numbers) so every chart in this
document is now self-verifying against the text.

### 9. Reconciling the continuous-shaping trend line vs. the summary table

The continuous-shaping plot's dashed trend line and the first/second-half
table row read in opposite directions (trend line improving, table
worse) -- both are real, correctly-computed statistics that can
legitimately disagree on a series this non-monotonic. The trend line is
an ordinary-least-squares fit through all 40 logged points; the table
compares the *mean* of the first 20 points to the *mean* of the last 20.
C's distance series swings 23.68 -> 23.08 -> 12.88 -> 21.90 -> 26.33 ->
25.71 -> 15.03 -> 18.36 across its 8 printed checkpoints (finer-grained
underneath) -- a regression line anchored near the high early points and
the lower late points can slope down even while the second-half *mean*
sits above the first-half mean, because two large mid-run spikes
(26.33, 25.71) pull the second-half average up without moving the
endpoint-driven regression slope the same way. Neither statistic is
wrong; they're answering different questions ("what's the linear trend"
vs. "was the back half worse on average"), and on a curve this noisy
they don't have to agree. Between the two, the first/second-half mean is
the one used for the verdicts in this document, since a single OLS slope
on 40 points this non-monotonic is more sensitive to exactly which
points happen to sit at the ends -- but this is a real ambiguity worth
flagging rather than silently picking one, and neither is as trustworthy
as `food_eaten`, which is unambiguous by construction.

Worth noting on its own: `n_direct_edges` stays at 0 throughout C's run
-- continuous shaping's liveness gain (30% -> 60%) came entirely from
strengthening the existing multi-hop path, not from growing a shortcut.
Of five conditions with full instrumentation (A, A2, B, C, D), only B
ever produced a direct edge at all. That's worth being honest about: the
0.205 direct-edge weight from B is one idiosyncratic route to a small
improvement, not a generalizable mechanism -- C found a *different*
route (strengthening length-4 relay) to a bigger improvement on the same
metrics. "Grow a direct edge" was never load-bearing in the story; it
just happened to be the visible artifact in one run.

## What this suggests

- The reward-modulated (three-factor) plasticity piece is the strongest
  part of the original idea and it is implemented faithfully here --
  it just hasn't been given a chain short/easy enough to demonstrate it
  yet, and the ablation shows *more noise alone doesn't shorten that
  chain*. Denser reward (continuous shaping) helps on every measurable
  axis except the one that actually matters (food eaten stays zero) --
  and it did so by strengthening an existing multi-hop relay, not by
  growing a shortcut: of five fully-instrumented runs, only one (B) ever
  grew a direct sensor->motor edge at all, so that 0.205 weight was one
  idiosyncratic route to a small improvement, not a generalizable
  mechanism worth relying on.
- The structural growth piece reliably does *something* (a connectome
  forms, population responds to energy) but "something" is not the same
  as "something useful." The precise gap, now measured rather than
  inferred: paths form almost immediately (100% of checkpoints, every
  run) and are live only ~30-40% of the time, mostly barely above
  threshold -- growth solves "wire something up," not "wire up
  something that fires."
- The performance ceiling is real but was partly an artifact of this
  implementation, not the mechanism -- a lesson in itself: don't trust a
  scaling number as a verdict on an idea until you've profiled it.
  Vectorizing the rest of the way (NumPy arrays over the whole
  population instead of a Python dict of objects) is still the right
  next move before scaling population further, and would also settle
  the still-open question of where the wall sits once implementation
  stops being the confound -- the same jump this repo's own `neuron.py`
  already made once, from a NumPy prototype to a GPU-accelerated one.
- The tragedy-of-the-commons genome-drift finding was real (spontaneity
  measurably falls under individual selection) but turned out to be a
  secondary effect, not the bottleneck -- a good reminder that a
  mechanistically clean story can still be the wrong explanation until
  you've actually removed the thing you think is causing it. Following it
  further turned up something broader than the original framing: it isn't
  just that spontaneity erodes, it's that **one underlying "reduce firing
  cost" selection pressure has more than one lever to express itself
  through**. Pin spontaneity and v_thresh climbs instead -- a clean,
  three-for-three replication (runs B, C, D) that holds regardless of
  reward regime, which is itself the evidence that it's a metabolic
  effect, not a reward-learning one. Only once *both* spontaneity and
  v_thresh are pinned does the pressure run out of levers, and mp_beta,
  tested in exactly that fully-isolated condition, stayed essentially
  flat (delta -0.0007, an order of magnitude below v_thresh's climb) --
  real evidence it's structurally different (it shapes existing input
  into spikes rather than manufacturing costly ones from nothing) rather
  than a gene that just hadn't had its turn yet. The general caution:
  don't call a heritable trait "safe from selection pressure" without
  locking down its neighbors first, since the pressure will happily move
  house.
- Across every condition tested -- evolvable spontaneity, fixed
  spontaneity, reward disabled, continuous shaping, two different seeds'
  worth of runs -- **food eaten never once left zero.** Distance-to-food
  trends are a noisy enough metric that a true null (zero-reward)
  condition passed them by chance in one run; food-eaten is not, and it's
  the cleanest single fact this whole document has produced.

## If continuing this sandbox

In rough order of expected information gained per unit effort, updated
after both rounds of follow-up checks:

1. **Chase the liveness gap directly -- it's the one lever that's moved
   the needle so far.** Continuous shaping alone (a ~15-line change) cut
   the distance-to-food degradation to a third of baseline and doubled
   the live-path fraction to 60%, without fixing the underlying problem.
   Next candidates, same spirit: stronger initial synapse weights
   (currently 0.05-0.2, weak relative to thresholds ~0.6-1.2), extra
   plasticity for young/novel synapses (the literal silent-synapse
   unsilencing mechanism -- Isaac/Nicoll/Malenka 1995 -- rather than a
   generic weight boost, since a generic boost would also inflate dead
   ends that were never near reward), or Turrigiano-style per-neuron
   homeostatic thresholding instead of leaving excitability to a
   heritable gene under uncertain selection -- no longer just a
   theoretical preference: the trait-substitution result above (points
   7-8) is direct evidence that heritable excitability genes don't stay
   put under selection, they migrate to whichever one is left
   unconstrained, which is exactly the failure mode a fixed regulatory
   rule sidesteps.
2. Finish the vectorization pass (NumPy arrays over the population)
   before scaling further -- both to remove the remaining implementation
   confound from any future scaling claim, and because it would make
   experiments like these (currently ~1.5-2 minutes per 20,000-cycle run
   at population 150) cheap enough to run across multiple seeds instead
   of one, which every ablation in this document genuinely needed and
   didn't get.
3. Let `learning_wall_experiment.py` run substantially longer or with a
   larger population cap -- 20,000 cycles with 150 neurons and 3 sensors,
   several times now, is still a small sample of a large search space,
   especially now that vectorization would make longer runs cheap.
4. Only after 1-3: revisit the visual cortex / "Memory Protein" pieces
   that were deliberately left out of this pass -- though point 5 below
   is a cheap, standalone way to sanity-check the population's state
   before investing more in relay mechanisms specifically.
5. **A linear-readout decoding probe**, orthogonal to 1-4: freeze a
   snapshot of population firing rates and fit a cheap linear/logistic
   classifier from firing rates -> correct steering direction. If it
   decodes well above chance, the sensory information is present in the
   population and the bottleneck really is relay/credit-assignment
   (consistent with everything above). If it doesn't decode at all, no
   amount of relay-strengthening fixes anything -- the population isn't
   representing the task yet, which would be a different and more
   fundamental finding. Not yet built here.
