"""
Empirical calibration for structure_pathway.py's growth/prune thresholds.

SONU_SPEC_V2.md Section 1 gives the SMGrNN instability rule and the prune
rule as formulas only -- lambda_edge, tau_w, and tau_delta have no numeric
defaults in the spec (SMGrNN's own paper values were tuned on gradient-
descent weight updates in a different substrate entirely, so porting its
numbers here wouldn't mean anything even if we had them). Since this
codebase's Δw source is STDP, not gradient descent (see structure_pathway.py
adaptation #1), this script measures the STDP-driven Δw distribution this
substrate actually produces and derives threshold defaults from that
measurement, rather than guessing blind.

Method:
  1. Build a mid-sized circuit with sparse random internal connectivity,
     initialized at the SAME weight scale growth actually produces
     (relay_init_scale * conn_weight_max -- see structure_pathway.py
     _insert_relay/_random_growth), not an arbitrary stronger range: tau_w
     needs to separate "freshly grown edge that got reinforced" from
     "freshly grown edge nothing reinforced," and those only look different
     starting from the same init point. Drive it with Poisson-ish input for
     several thousand steps, with structure_pathway's connection-STDP (incl.
     passive weight decay) running but growth/prune/homeostasis DISABLED
     (prune_period_s set past the run length) -- this measures the "natural"
     Δw statistics an edge produces before the structure pathway has done
     anything to it, which is what growth/prune thresholds need to be tuned
     against.
  2. Periodically snapshot every edge's (mu_k, sigma_k^2, |weight|) once its
     window is full, matching structure_pathway._instability_growth's own
     computation exactly, and pool the samples across all edges and time.
  3. Report percentiles and propose defaults:
       - tau_w:      a low percentile of |weight| (small enough to be a
                      believable "prune candidate", not just any weight)
       - tau_delta:  a low percentile of |mu_k| (below the noise floor of
                      an edge that is meaningfully still drifting)
       - lambda_edge: chosen so the instability flag fires for a plausible
                      minority of edge-windows (SMGrNN's own ablation
                      targets sparse, occasional flags -- 84 params on
                      CartPole after growth+prune -- not "flag everything")

Run: PYTHONPATH=. python3 calibrate_structure_thresholds.py
"""

import numpy as np

from circuit import NeuralCircuit
from structure_pathway import SelfConnectingPathway, StructuralPlasticityConfig


def run_calibration(
    num_neurons=20,
    input_channels=12,
    n_edges=40,
    n_steps=4000,
    window_T=50,
    conn_a_plus=0.05,
    conn_a_minus=0.05,
    input_rate=0.15,
    snapshot_every=10,
    seed=0,
):
    rng = np.random.default_rng(seed)

    circuit = NeuralCircuit(
        num_neurons=num_neurons,
        input_channels=input_channels,
        dt=1.0,
        max_delay=5,
        # +1 only to satisfy SelfConnectingPathway's growth-room guard; growth
        # never fires in this run (prune_period_s is set past n_steps below),
        # so the spare slot is unused.
        max_neurons=num_neurons + 1,
        neuron_params={'tau_m': 15.0, 'theta_base': -58.0},
    )

    cfg = StructuralPlasticityConfig(
        window_T=window_T,
        prune_period_s=n_steps + 1,  # never fires: measure raw STDP Δw, not this run's own structural response
        conn_a_plus=conn_a_plus,
        conn_a_minus=conn_a_minus,
        seed=seed,
    )

    # Sparse random internal connectivity, initialized at the SAME weight
    # scale a freshly grown relay/random edge actually starts at -- see
    # module docstring for why this matters for tau_w specifically.
    init_w = cfg.relay_init_scale * cfg.conn_weight_max
    made = set()
    while len(made) < n_edges:
        i, j = rng.integers(0, num_neurons, size=2)
        if i != j and (i, j) not in made:
            circuit.connect(int(i), int(j), weight=init_w, delay=1)
            made.add((i, j))

    pathway = SelfConnectingPathway(circuit, cfg)

    mus, sigma2s, abs_weights = [], [], []

    for t in range(n_steps):
        input_spikes = (rng.random(input_channels) < input_rate).astype(np.float64)
        I_ext = rng.uniform(5.0, 20.0, size=circuit.num_neurons)
        output_spikes = circuit.step(input_spikes, I_ext=I_ext, learning=True)
        pathway.step(output_spikes)

        if t % snapshot_every != 0:
            continue
        for source_id in range(circuit.num_neurons):
            for connection in circuit.connections[source_id]:
                history = pathway._edge_history.get(connection.conn_id)
                if history is None or len(history) < cfg.window_T:
                    continue
                mus.append(float(np.mean(history)))
                sigma2s.append(float(np.var(history)))
                abs_weights.append(abs(connection.weight))

    return np.array(mus), np.array(sigma2s), np.array(abs_weights)


def propose_thresholds(mus, sigma2s, abs_weights):
    abs_mus = np.abs(mus)
    ratio = sigma2s / np.maximum(abs_mus, 1e-12)  # sigma_k^2 / |mu_k|, what lambda_edge thresholds

    def pct(arr, p):
        return float(np.percentile(arr, p)) if len(arr) else float('nan')

    print(f"Samples collected: {len(mus)} (edge, window) observations\n")

    print("abs(mu_k)   [prune uses tau_delta against this]")
    for p in (5, 10, 25, 50, 75, 90):
        print(f"  p{p:>2}: {pct(abs_mus, p):.6f}")

    print("\nabs(weight) [prune uses tau_w against this]")
    for p in (5, 10, 25, 50, 75, 90):
        print(f"  p{p:>2}: {pct(abs_weights, p):.6f}")

    print("\nsigma_k^2 / |mu_k|  [lambda_edge thresholds this]")
    for p in (50, 75, 80, 90, 95, 99):
        print(f"  p{p:>2}: {pct(ratio, p):.4f}")

    tau_delta = pct(abs_mus, 10)
    tau_w = pct(abs_weights, 10)
    lambda_edge = pct(ratio, 90)  # flag roughly the top decile as unstable

    print("\n" + "="*60)
    print("PROPOSED DEFAULTS (grounded in this run; not final -- sanity check before use)")
    print("="*60)
    print(f"  tau_delta   = {tau_delta:.4f}   (p10 of |mu_k|)")
    print(f"  tau_w       = {tau_w:.4f}   (p10 of |weight|)")
    print(f"  lambda_edge = {lambda_edge:.4f}   (p90 of sigma_k^2 / |mu_k| -> flags ~top decile)")
    return tau_delta, tau_w, lambda_edge


if __name__ == "__main__":
    print("Running calibration simulation (drives a 20-neuron circuit with 40 "
          "random internal edges for 4000 steps; may take a little while on CPU)...\n")
    mus, sigma2s, abs_weights = run_calibration()
    propose_thresholds(mus, sigma2s, abs_weights)
