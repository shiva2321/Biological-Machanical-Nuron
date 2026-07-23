"""
Verifies (or refutes) the growth/prune population-balance claims documented
in calibrate_structure_thresholds.py's "KNOWN LIMITATIONS" section and
structure_pathway.py's StructuralPlasticityConfig docstring.

This is the script that found, independently and empirically rather than
by inspection alone, that:

  1. The (now fixed) joint-vs-marginal threshold calibration bug was real:
     with tau_w/tau_delta picked as independent marginal p10s, pruning
     fired ~1% of the time against growth's ~10%, and a circuit grew
     monotonically to its max_neurons cap regardless of input rate.
  2. That fix (joint calibration against growth's own measured flag rate)
     helps substantially at the reference input regime the calibration
     script itself uses, but does NOT transfer to a circuit driven with
     materially different input statistics -- pruning falls badly behind
     again at higher input activity, even with the corrected thresholds.
  3. Even at the reference regime, growth does not appear self-limiting:
     given far more pool headroom, population still grows monotonically
     to the (larger) cap rather than settling at an equilibrium size,
     because each relay insertion nets +1 node / +2 edges (the original
     edge is kept, per SONU_SPEC_V2.md Section 1) while each successful
     prune nets only -1 edge -- a structural, not just rate, imbalance.

Run: PYTHONPATH=. python3 experiments/verify_structure_pathway_growth_prune_balance.py
"""

import numpy as np

from circuit import NeuralCircuit
from structure_pathway import SelfConnectingPathway, StructuralPlasticityConfig


def run(on_p, n_steps, num_neurons=15, input_channels=10, max_neurons=45, seed=0, verbose=True):
    """
    Drives a circuit with Bernoulli(on_p) input for n_steps (or until
    max_neurons is hit) using the pathway's shipped defaults, and reports
    how many relay-growth vs prune events fired.
    """
    rng = np.random.default_rng(seed)
    circuit = NeuralCircuit(
        num_neurons=num_neurons, input_channels=input_channels, dt=1.0, max_delay=3,
        max_neurons=max_neurons,
        neuron_params={'tau_m': 15.0, 'theta_base': -58.0},
    )
    made = set()
    while len(made) < 25:
        i, j = rng.integers(0, num_neurons, size=2)
        if i != j and (i, j) not in made:
            circuit.connect(int(i), int(j), weight=0.5, delay=1)  # matches relay_init_scale * conn_weight_max
            made.add((i, j))

    cfg = StructuralPlasticityConfig(window_T=50, prune_period_s=100, seed=seed)  # shipped defaults, untouched
    pathway = SelfConnectingPathway(circuit, cfg)

    grown_total, pruned_total = 0, 0
    hit_cap_at = None
    for t in range(n_steps):
        input_spikes = (rng.random(input_channels) < on_p).astype(np.float64)
        I_ext = rng.uniform(5.0, 20.0, size=circuit.num_neurons)
        output_spikes = circuit.step(input_spikes, I_ext=I_ext, learning=True)
        events = pathway.step(output_spikes)
        grown_total += len(events['grown_relay'])
        pruned_total += len(events['pruned'])
        if circuit.num_neurons >= max_neurons:
            hit_cap_at = t
            break

    if verbose:
        cap_note = f" (hit cap at step {hit_cap_at})" if hit_cap_at is not None else ""
        print(f"  on_p={on_p:.2f}, max_neurons={max_neurons}: neurons {num_neurons} -> "
              f"{circuit.num_neurons}{cap_note}, grown={grown_total}, pruned={pruned_total}")
    return circuit.num_neurons, grown_total, pruned_total, hit_cap_at


if __name__ == "__main__":
    print("Part 1: growth/prune balance across input regimes (reference on_p=0.15, cap=45)")
    print("-" * 78)
    for on_p in (0.15, 0.35, 0.55, 0.75):
        for seed in (0, 1):
            run(on_p, n_steps=3000, max_neurons=45, seed=seed)

    print("\nPart 2: does growth self-limit given far more headroom? (cap=150)")
    print("-" * 78)
    for on_p in (0.15, 0.75):
        for seed in (0, 1):
            run(on_p, n_steps=6000, max_neurons=150, seed=seed)

    print("\nIf growth were tracking task demand rather than racing a wall, higher-cap")
    print("runs at the SAME on_p should settle below the cap, not just take longer to")
    print("reach a bigger one. As of this pathway's current design, they don't -- see")
    print("this file's module docstring and calibrate_structure_thresholds.py's")
    print("'KNOWN LIMITATIONS' section for what that does and doesn't mean.")
