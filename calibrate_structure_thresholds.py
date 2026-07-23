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
       - lambda_edge: chosen so the instability flag fires for a plausible
                      minority of edge-windows (SMGrNN's own ablation
                      targets sparse, occasional flags -- 84 params on
                      CartPole after growth+prune -- not "flag everything")
       - tau_w, tau_delta: calibrated JOINTLY (find_joint_prune_thresholds)
                      against growth's own measured flag rate, NOT as
                      independent marginal percentiles. This was a real
                      bug in an earlier version of this script: the prune
                      rule needs |w_k| <= tau_w AND |mu_k| <= tau_delta at
                      once, so picking each as its own marginal p10
                      produces a joint rate around p10*p10 (~1%) --
                      roughly 7-10x weaker than growth's ~10% rate at the
                      calibrated lambda_edge. Verified empirically (see
                      PR discussion): with pruning that starved, a circuit
                      grows monotonically to whatever max_neurons cap is
                      configured, on a fixed schedule, regardless of task.

KNOWN LIMITATIONS, not fixed by the joint-rate calibration above (flagged
honestly rather than silently left for someone else to rediscover):

  - These thresholds are calibrated against ONE input regime (this
    script's on_p=0.15-equivalent reference run) and do NOT automatically
    transfer to a circuit driven with different input statistics. Verified
    empirically: at higher input activity, even the joint-calibrated
    thresholds fall badly behind growth again (pruned/grown ratios like
    1/30 rather than the ~25/30 seen at the reference rate). Recalibrate
    per-task (re-run this script with your actual circuit/stimulus) rather
    than trusting the shipped defaults on a materially different input
    regime.
  - Even at the reference input regime, growth does not appear to be
    self-limiting at all: given 10x more pool headroom (max_neurons=150
    instead of 45), it still grows monotonically to the new cap rather
    than settling at some smaller equilibrium size, even though pruning
    keeps much closer pace there (~84% of growth's event count). This
    looks structural, not just a rate-matching problem: each relay
    insertion nets +1 node and +2 edges (the original edge is kept, per
    spec Section 1), while each successful prune nets -1 edge -- so even
    at perfectly matched per-edge flag RATES, growth and pruning are not
    matched in raw population-flux terms. Matching flag rates was
    necessary but is not sufficient for population size to reach a
    stable equilibrium; that needs its own follow-up (e.g. rate-limiting
    relay insertions per cycle, or increasing pruning's aggressiveness
    relative to growth's, not just its trigger rate) and is explicitly
    NOT attempted here.

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


def growth_flag_rate(mus, sigma2s, lambda_edge):
    """Fraction of (edge, window) samples the instability rule itself would flag at this lambda_edge."""
    abs_mus = np.abs(mus)
    sigma = np.sqrt(sigma2s)
    flagged = (abs_mus < 0.5 * sigma) & (sigma2s > lambda_edge * abs_mus)
    return float(np.mean(flagged)) if len(mus) else float('nan')


def find_joint_prune_thresholds(abs_mus, abs_weights, target_rate, p_grid=None):
    """
    Search a single shared percentile p, applied to both marginals at once
    (tau_w = percentile(|weight|, p), tau_delta = percentile(|mu|, p)), for
    the value whose JOINT (AND) flag rate is closest to target_rate.

    This replaces picking each threshold's own marginal percentile in
    isolation (e.g. "p10 of each"): the prune rule requires BOTH
    conditions at once, so independently-chosen p10 marginals produce a
    joint rate near p10 * p10 (~1%), not p10 (~10%) -- found empirically
    (see PR discussion) to starve pruning relative to growth badly enough
    that population size just races to whatever cap is configured.
    """
    if p_grid is None:
        p_grid = np.arange(1, 91)
    best = None
    for p in p_grid:
        tau_w = float(np.percentile(abs_weights, p))
        tau_delta = float(np.percentile(abs_mus, p))
        rate = float(np.mean((abs_weights <= tau_w) & (abs_mus <= tau_delta)))
        if best is None or abs(rate - target_rate) < abs(best[2] - target_rate):
            best = (tau_w, tau_delta, rate, int(p))
    return best  # (tau_w, tau_delta, achieved_joint_rate, percentile_used)


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

    lambda_edge = pct(ratio, 90)  # flag roughly the top decile as unstable
    growth_rate = growth_flag_rate(mus, sigma2s, lambda_edge)

    naive_tau_delta, naive_tau_w = pct(abs_mus, 10), pct(abs_weights, 10)
    naive_joint_rate = float(np.mean((abs_weights <= naive_tau_w) & (abs_mus <= naive_tau_delta)))

    tau_w, tau_delta, joint_rate, p_used = find_joint_prune_thresholds(abs_mus, abs_weights, target_rate=growth_rate)

    print("\n" + "="*60)
    print("GROWTH vs PRUNE FLAG RATE (what actually governs the population's growth/prune balance)")
    print("="*60)
    print(f"  growth flag rate @ lambda_edge={lambda_edge:.4f}:                    {growth_rate:.4f}")
    print(f"  prune flag rate  @ naive independent p10/p10 (tau_w={naive_tau_w:.4f}, tau_delta={naive_tau_delta:.6f}): "
          f"{naive_joint_rate:.4f}  <- ~{growth_rate / max(naive_joint_rate, 1e-9):.0f}x weaker than growth; this is the bug")
    print(f"  prune flag rate  @ joint-calibrated  (tau_w={tau_w:.4f}, tau_delta={tau_delta:.6f}, p{p_used}): "
          f"{joint_rate:.4f}  <- matched to growth's rate")

    print("\n" + "="*60)
    print("PROPOSED DEFAULTS (grounded in this run; not final -- sanity check before use)")
    print("="*60)
    print(f"  tau_delta   = {tau_delta:.6f}   (joint-calibrated against growth's own flag rate, not an independent marginal)")
    print(f"  tau_w       = {tau_w:.4f}   (joint-calibrated against growth's own flag rate, not an independent marginal)")
    print(f"  lambda_edge = {lambda_edge:.4f}   (p90 of sigma_k^2 / |mu_k| -> flags ~top decile)")
    return tau_delta, tau_w, lambda_edge


if __name__ == "__main__":
    print("Running calibration simulation (drives a 20-neuron circuit with 40 "
          "random internal edges for 4000 steps; may take a little while on CPU)...\n")
    mus, sigma2s, abs_weights = run_calibration()
    propose_thresholds(mus, sigma2s, abs_weights)
