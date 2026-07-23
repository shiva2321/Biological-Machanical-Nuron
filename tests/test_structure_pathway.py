"""
Smoke tests for the self-connecting structure pathway (structure_pathway.py),
implementing SONU_SPEC_V2.md Section 1 in isolation (the "+connect" ablation
ladder rung -- see SONU_SPEC_V2.md Section 6 -- with nothing else layered on).

Run directly: python tests/test_structure_pathway.py
Or via pytest: pytest tests/test_structure_pathway.py
"""

from collections import deque

import numpy as np

from circuit import NeuralCircuit
from structure_pathway import SelfConnectingPathway, StructuralPlasticityConfig


def _make_circuit(num_neurons=6, input_channels=4, max_neurons=20, max_delay=5):
    return NeuralCircuit(
        num_neurons=num_neurons,
        input_channels=input_channels,
        dt=1.0,
        max_delay=max_delay,
        max_neurons=max_neurons,
        neuron_params={'tau_m': 10.0, 'theta_base': -60.0},
    )


def test_connection_stdp_updates_weight_and_history():
    print("="*70)
    print("TEST 1: internal-connection STDP produces real Δw")
    print("="*70)

    circuit = _make_circuit(num_neurons=3, input_channels=2, max_neurons=4)
    conn = circuit.connect(0, 1, weight=1.0, delay=1)
    pathway = SelfConnectingPathway(circuit, StructuralPlasticityConfig(window_T=5, prune_period_s=10**9))

    initial_weight = conn.weight
    for t in range(30):
        input_spikes = np.array([1.0, 0.0]) if t % 3 == 0 else np.array([0.0, 0.0])
        I_ext = np.array([25.0, 25.0, 0.0])
        output_spikes = circuit.step(input_spikes, I_ext=I_ext, learning=True)
        pathway.step(output_spikes)

    assert conn.conn_id in pathway._edge_history, "edge should be tracked after connect()"
    assert len(pathway._edge_history[conn.conn_id]) > 0, "edge should have accumulated Δw history"
    print(f"  weight: {initial_weight:.4f} -> {conn.weight:.4f}")
    print(f"  history samples: {len(pathway._edge_history[conn.conn_id])}")
    print("PASS: connection STDP produces trackable delta-w\n")


def test_instability_growth_inserts_relay():
    print("="*70)
    print("TEST 2: SMGrNN instability flag triggers relay insertion")
    print("="*70)

    circuit = _make_circuit(num_neurons=3, input_channels=2, max_neurons=6)
    conn = circuit.connect(0, 2, weight=1.0, delay=1)
    cfg = StructuralPlasticityConfig(window_T=10, prune_period_s=10**9)
    pathway = SelfConnectingPathway(circuit, cfg)

    # Engineer an oscillating, near-zero-mean Δw history: exactly the
    # "persistently undecided" signature SMGrNN's rule is built to catch,
    # as opposed to an edge that's merely idle (small mean, small variance).
    oscillating = [3.0, -3.0] * (cfg.window_T // 2)
    pathway._edge_history[conn.conn_id] = deque(oscillating, maxlen=cfg.window_T)

    mu, sigma = np.mean(oscillating), np.std(oscillating)
    assert abs(mu) < 0.5 * sigma and sigma ** 2 > cfg.lambda_edge * abs(mu), \
        "test setup bug: engineered history should itself satisfy the flag condition"

    neurons_before = circuit.num_neurons
    grown = pathway._instability_growth()

    assert len(grown) == 1, f"expected exactly one relay insertion, got {grown}"
    source_id, relay_id, target_id = grown[0]
    assert (source_id, target_id) == (0, 2)
    assert circuit.num_neurons == neurons_before + 1
    assert circuit.neuron_active[relay_id] is True
    assert relay_id in pathway._relay_ids

    conn_ids_from_0 = {c.conn_id for c in circuit.connections[0] if c.active}
    assert conn.conn_id in conn_ids_from_0, "original edge must be kept, not replaced"
    relay_out = [c for c in circuit.connections[relay_id] if c.active]
    assert len(relay_out) == 1 and relay_out[0].target_id == target_id

    relay_weights = circuit.neurons[relay_id].get_weights()
    assert np.allclose(relay_weights, 0.0), "relay's external input weights should be zero-initialized"

    print(f"  relay {relay_id} inserted on path {source_id} -> {relay_id} -> {target_id}")
    print("PASS: instability-triggered growth works\n")


def test_capacity_cap_is_respected():
    print("="*70)
    print("TEST 3: growth respects max_neurons")
    print("="*70)

    circuit = _make_circuit(num_neurons=2, input_channels=2, max_neurons=2)  # zero growth room
    idx = circuit.add_neuron()
    assert idx is None, "add_neuron() must return None once max_neurons is reached"
    assert circuit.num_neurons == 2
    print("PASS: pool cap respected (add_neuron returns None at capacity)\n")


def test_extreme_lambda_edge_does_not_disable_growth_but_the_flag_does():
    print("="*70)
    print("TEST 3b: lambda_edge cannot disable growth at mu~0; enable_instability_growth=False can")
    print("="*70)

    # An exactly-zero-mean, nonzero-variance history. This makes the bug
    # airtight rather than a matter of picking "big enough" numbers: when
    # mu_k == 0.0 exactly, the flag condition's right-hand side
    # (lambda_edge * |mu_k|) is 0 * lambda_edge = 0 for EVERY finite
    # lambda_edge, so `sigma_k^2 > lambda_edge * |mu_k|` collapses to
    # `sigma_k^2 > 0`, which is lambda_edge-independent by construction --
    # not just true for the lambda_edge value this test happens to try.
    exactly_zero_mean = [3.0, -3.0] * 10  # mean == 0.0 exactly, real variance

    circuit = _make_circuit(num_neurons=3, input_channels=2, max_neurons=6)
    conn = circuit.connect(0, 2, weight=1.0, delay=1)

    huge_lambda_cfg = StructuralPlasticityConfig(window_T=20, lambda_edge=1e12)
    pathway = SelfConnectingPathway(circuit, huge_lambda_cfg)
    pathway._edge_history[conn.conn_id] = deque(exactly_zero_mean, maxlen=20)
    mu, sigma = np.mean(exactly_zero_mean), np.std(exactly_zero_mean)
    assert mu == 0.0 and sigma > 0.0
    assert abs(mu) < 0.5 * sigma and sigma ** 2 > huge_lambda_cfg.lambda_edge * abs(mu), \
        "lambda_edge=1e12 should still fail to block this history (that's the bug)"
    grown = pathway._instability_growth()
    assert len(grown) == 1, "confirms lambda_edge alone cannot disable growth at mu=0"

    # Same history that just proved it triggers growth above -- but with the
    # flag off instead of an extreme threshold. enable_connection_stdp is
    # also disabled here so the seeded history isn't overwritten by real
    # (decay-only, zero-spike) STDP updates during the step() loop below;
    # that isolates exactly one variable: does enable_instability_growth
    # block a history that would otherwise trigger growth? Note
    # max_neurons == num_neurons is fine here (no ValueError), which is
    # itself part of the fix: a growth-free arm shouldn't be forced to
    # allocate pool room it will never use.
    circuit2 = _make_circuit(num_neurons=3, input_channels=2, max_neurons=3)
    conn2 = circuit2.connect(0, 2, weight=1.0, delay=1)
    disabled_cfg = StructuralPlasticityConfig(window_T=20, prune_period_s=20,
                                               enable_connection_stdp=False,
                                               enable_instability_growth=False)
    pathway2 = SelfConnectingPathway(circuit2, disabled_cfg)
    pathway2._edge_history[conn2.conn_id] = deque(exactly_zero_mean, maxlen=20)

    events = None
    for _ in range(disabled_cfg.prune_period_s):
        output_spikes = np.zeros(circuit2.num_neurons, dtype=bool)
        events = pathway2.step(output_spikes)  # drives _step_count to a prune_period_s boundary

    assert events['grown_relay'] == [], "enable_instability_growth=False must block growth through the public step() path"
    assert circuit2.num_neurons == 3, "no relay neuron should have been added"
    print("PASS: enable_instability_growth=False is the correct disable lever, not lambda_edge\n")


def test_prune_removes_weak_stable_edges():
    print("="*70)
    print("TEST 4: mandatory pruning removes weak, stable edges")
    print("="*70)

    circuit = _make_circuit(num_neurons=3, input_channels=2, max_neurons=4)
    weak_conn = circuit.connect(0, 1, weight=0.01, delay=1)
    strong_conn = circuit.connect(0, 2, weight=5.0, delay=1)

    cfg = StructuralPlasticityConfig(window_T=10, tau_w=0.05, tau_delta=0.005, prune_fraction_eta=1.0)
    pathway = SelfConnectingPathway(circuit, cfg)

    quiet_history = deque([0.0] * cfg.window_T, maxlen=cfg.window_T)
    pathway._edge_history[weak_conn.conn_id] = deque(quiet_history, maxlen=cfg.window_T)
    pathway._edge_history[strong_conn.conn_id] = deque(quiet_history, maxlen=cfg.window_T)

    pruned = pathway._prune()

    assert (0, 1) in pruned, "weak, stable edge should be pruned"
    assert weak_conn.active is False
    assert strong_conn.active is True, "strong edge should survive despite also being Δw-stable"
    print(f"  pruned: {pruned}")
    print("PASS: pruning targets weight AND stability jointly, not just one\n")


def test_orphan_relay_removed_after_hop_pruned():
    print("="*70)
    print("TEST 5: orphaned relay is removed; original topology is untouched")
    print("="*70)

    circuit = _make_circuit(num_neurons=3, input_channels=2, max_neurons=6)
    # Deliberately weak, out-degree-0 original edge -- shaped like an "orphan"
    # by SMGrNN's literal rule, but it is NOT a relay, so it must survive.
    original_conn = circuit.connect(0, 1, weight=0.01, delay=1)
    cfg = StructuralPlasticityConfig(window_T=5)
    pathway = SelfConnectingPathway(circuit, cfg)

    relay_id = pathway._insert_relay(0, 2, base_delay=1)
    assert relay_id is not None

    # Simulate a prune cycle that removed the relay's outgoing hop.
    relay_conn = circuit.connections[relay_id][0]
    relay_conn.active = False

    removed = pathway._remove_orphans()
    assert relay_id in removed
    assert circuit.neuron_active[relay_id] is False

    assert circuit.neuron_active[1] is True
    assert original_conn.active is True
    print(f"  removed orphaned relay {relay_id}; original neuron 1 (out-degree 0 by design) left untouched")
    print("PASS: orphan removal is correctly scoped to relay neurons only\n")


def test_synaptic_normalization_holds_afferent_sum():
    print("="*70)
    print("TEST 6: SORN synaptic normalization pulls afferent sum back to target")
    print("="*70)

    circuit = _make_circuit(num_neurons=3, input_channels=2, max_neurons=4)
    c1 = circuit.connect(0, 2, weight=2.0, delay=1)
    c2 = circuit.connect(1, 2, weight=2.0, delay=1)
    pathway = SelfConnectingPathway(circuit)

    target = pathway._afferent_targets[2]
    assert np.isclose(target, 4.0)

    # Perturb away from the captured target, as STDP would over time.
    c1.weight = 9.0
    c2.weight = 9.0
    assert np.isclose(pathway._afferent_sum(2), 18.0)

    pathway._synaptic_normalization()

    new_sum = pathway._afferent_sum(2)
    assert np.isclose(new_sum, target, atol=1e-6), f"expected afferent sum back at {target}, got {new_sum}"
    assert np.isclose(c1.weight, c2.weight), "relative proportions between edges should be preserved"
    print(f"  afferent sum: 18.0 -> {new_sum:.4f} (target {target:.4f})")
    print("PASS: synaptic normalization restores each neuron's own afferent budget\n")


def test_end_to_end_smoke_run_stays_consistent():
    print("="*70)
    print("TEST 7: end-to-end run stays internally consistent")
    print("="*70)

    circuit = _make_circuit(num_neurons=8, input_channels=6, max_neurons=24, max_delay=4)
    circuit.connect_chain(weight=0.5, delay=1)
    circuit.connect(0, 5, weight=0.3, delay=2)

    cfg = StructuralPlasticityConfig(window_T=20, prune_period_s=50, seed=0)
    pathway = SelfConnectingPathway(circuit, cfg)

    rng = np.random.default_rng(0)
    total_events = {'grown_relay': 0, 'grown_random': 0, 'pruned': 0, 'orphans_removed': 0}
    start_neurons = circuit.num_neurons

    for t in range(600):
        input_spikes = (rng.random(6) < 0.2).astype(np.float64)
        I_ext = rng.uniform(0.0, 20.0, size=circuit.num_neurons)
        output_spikes = circuit.step(input_spikes, I_ext=I_ext, learning=True)
        events = pathway.step(output_spikes)
        for k in total_events:
            total_events[k] += len(events[k])

        assert circuit.num_neurons <= circuit.max_neurons, "pool cap must never be exceeded"
        assert len(circuit.neuron_active) == circuit.num_neurons
        assert len(circuit.connections) == circuit.num_neurons

    print(f"  ran 600 steps; neurons {start_neurons} -> {circuit.num_neurons} (cap {circuit.max_neurons})")
    print(f"  events over run: {total_events}")
    print("PASS: 600-step run completes with all pool invariants intact\n")


if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("STRUCTURE PATHWAY TESTS (SONU_SPEC_V2.md Section 1, +connect rung)")
    print("="*70)
    print()

    test_connection_stdp_updates_weight_and_history()
    test_instability_growth_inserts_relay()
    test_capacity_cap_is_respected()
    test_extreme_lambda_edge_does_not_disable_growth_but_the_flag_does()
    test_prune_removes_weak_stable_edges()
    test_orphan_relay_removed_after_hop_pruned()
    test_synaptic_normalization_holds_afferent_sum()
    test_end_to_end_smoke_run_stays_consistent()

    print("="*70)
    print("ALL STRUCTURE PATHWAY TESTS PASSED")
    print("="*70)
