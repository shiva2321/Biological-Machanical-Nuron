"""
Self-Connecting (Structure) Pathway — SONU_SPEC_V2.md, Section 1.

Wraps a NeuralCircuit and manages structural plasticity in isolation from
the other three SONU properties (self-learning's content pathway,
self-evolving, self-replicating). This is the "+connect" rung of the
spec's ablation ladder (SONU_SPEC_V2.md Section 6) and nothing past it:

    static -> +connect (SMGrNN rule)   <-- this module
      -> +connect +content-learn (delta-rule)
      -> +connect +content-learn +structure-learn (Hebbian + M)
      -> +...+evolve (self-referential)
      -> +...+replicate (clonal)

It was built first, deliberately, because it's the one piece of the spec
with real ablation numbers behind it (SMGrNN's growth+prune vs growth-only
comparison) and the only piece that doesn't depend on resolving the spec's
still-open questions (Section 5b's growth/replication tie-break, Section
5c's homeostasis-vs-replication interaction, Section 3's self-referential
gene-vector update) -- those only matter once self-replicating and
self-evolving are layered on top, which hasn't happened yet.

Substrate adaptations from the spec text (each is a real decision, flagged
here rather than silently made):

1. Δw source. Section 1's instability criterion (|mu_k| < 0.5*sigma_k and
   sigma_k^2 > lambda_edge*|mu_k|) was defined by SMGrNN over gradient-
   descent weight updates. This codebase has no gradient steps -- learning
   is STDP-driven. The criterion itself only needs *a* per-edge time series
   of weight-update magnitude, so Delta w_k here is "the STDP update
   applied to edge k this step," not a gradient step. Same formula, same
   threshold shape, different source signal.

2. Base learning rule for internal connections. Before this module,
   BiologicalNeuron.weights (external input -> neuron) already learned via
   STDP, but NeuralCircuit.Connection (neuron -> neuron) weights were
   static -- see the original README's "Known Limitations" ("STDP only on
   external inputs, not internal connections") and "Future Enhancements"
   ("STDP on Internal Connections", "Structural Plasticity: Dynamic
   connection creation/pruning"), i.e. exactly this. This module adds
   pairwise STDP to internal connections, reusing each neuron's existing
   `post_trace` (BiologicalNeuron already decays and increments it on
   spike; it doubles as that neuron's own pre/post synaptic trace for any
   connection it is the source or target of). This is the minimal base
   rule for the "+connect" rung; Section 2's three-factor Hebbian
   (Delta w = eta * pre * post * M, M = local prediction error) is a later
   ladder rung ("+structure-learn") and is intentionally not built here --
   building it now would mean depending on the predictive-coding modulator
   before it's been added to this codebase, which defeats the point of
   testing structure-pathway-only first.

2b. Passive weight decay on internal connections, mirroring
   BiologicalNeuron.stdp()'s own weight_decay convention (applied
   unconditionally every step: new_w = (w + stdp_delta) * (1 - decay)).
   Discovered as a genuine gap during threshold calibration, not
   speculative: without it, an edge that never gets a coincident pre/post
   spike (most obviously a freshly grown relay hop nothing reinforces yet)
   simply sits frozen at its init weight forever, since LTP/LTD only fire
   on spike coincidence. Mandatory pruning (Section 1: |w_k| <= tau_w AND
   |mu_k| <= tau_delta) would then have no way to distinguish "actively
   reinforced" from "never reinforced" -- both look identical (frozen) to
   the prune check. Decay gives unreinforced edges a slow, low-variance
   downward drift: small and stable, so it correctly avoids tripping the
   *instability* flag (which needs high variance relative to the mean),
   while eventually crossing the prune thresholds.

3. Fixed-capacity pool. Section 0's engineering constraint ("pre-allocate
   a max-capacity pool, mask dead/ungrown units, do not literally resize
   tensors mid-training") was written for a batched-tensor substrate (one
   weight matrix for the whole population). This circuit is object-
   oriented instead: each BiologicalNeuron owns its own small, independently
   -sized tensors that never change shape, and neuron-to-neuron connections
   are plain Python objects, not a stacked tensor. Densely pre-allocating
   every unused pool slot up front would burn GPU memory for idle neurons
   that batching was meant to avoid in the first place. So the constraint
   is enforced as (a) a hard `max_neurons` cap on growth
   (NeuralCircuit.add_neuron returns None past it) and (b) mask-don't-
   delete for pruned connections and orphaned neurons -- never a resize.

4. Relay neuron input weights. A freshly grown BiologicalNeuron gets
   Xavier-initialized external input weights like any other neuron, which
   would make a "relay" an independent input-driven feature detector
   competing with relaying the i->j signal, not a clean two-hop pass-
   through. This module zeroes a relay's external weights at insertion so
   the new i -> relay -> j path starts out driven by the relayed signal.
   STDP is left running afterward, so a relay can still develop its own
   input tuning over time -- it's just not seeded with one.

5. Orphan-removal scope. Section 1 says to remove orphan nodes (zero in-
   or out-degree) after pruning. In a pure structural-growth graph that's
   unambiguous. Here it isn't: every neuron, including the circuit's
   original ones, also has an independent external-input pathway, so zero
   *internal* degree does not mean zero activity -- a chain's head neuron
   is legitimately internal-in-degree 0 by original design. Applying
   Section 1's orphan rule to every neuron would silently deactivate the
   user's original topology the first time this pathway runs. Orphan-
   removal is therefore scoped to neurons this pathway itself created
   (relays): they have no independent input tuning (see #4) and exist
   solely to mediate one two-hop path, so losing either hop really does
   leave them doing nothing.

6. Scope note on mixed-purpose circuits. This pathway applies STDP to,
   tracks history for, and can prune *every* connection present on the
   circuit it's attached to, including ones added before construction.
   Connection.weight is clipped into [conn_weight_min, conn_weight_max]
   (default [0, 10], matching neuron.py's own default bounds -- an
   all-excitatory, Dale's-law-style convention). A circuit that also
   carries hand-set negative-weight connections (e.g.
   NeuralCircuit.connect_lateral_inhibition) will have those clipped
   non-negative the first time a coincident spike drives a nonzero
   update -- silently breaking the inhibition. Don't attach this pathway
   to a circuit that mixes STDP-managed excitatory edges with statically-
   set inhibitory ones; give inhibition its own circuit/demo instead.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from circuit import Connection, NeuralCircuit


@dataclass
class StructuralPlasticityConfig:
    """
    Hyperparameters for the structure pathway. The spec (Section 1) gives
    formulas, not numbers -- SMGrNN's own paper values were tuned on
    gradient-descent updates in a different substrate, so they wouldn't
    transfer even if we had them (see structure_pathway.py adaptation #1).

    lambda_edge, tau_w, and tau_delta below are NOT guesses: they're taken
    from calibrate_structure_thresholds.py's p10/p90 measurements of this
    module's actual STDP-driven Δw distribution (20-neuron circuit, 40
    edges initialized at relay_init_scale, 4000 steps -- see that script
    for the full method). Re-run it and update these if conn_a_plus/
    conn_a_minus/relay_init_scale/circuit scale change materially, since
    the numbers are only as good as how representative that run is of your
    actual circuit.

    To build an ablation-isolation arm (e.g. "STDP-only, no structural
    change" as a middle rung between a bare circuit and the full pathway),
    use the enable_* flags below -- do NOT try to threshold a mechanism
    off by cranking lambda_edge/tau_w/tau_delta to an extreme. That does
    not work: the instability check is `sigma_k^2 > lambda_edge * |mu_k|`,
    and as |mu_k| -> 0 (common for an edge whose LTP/LTD roughly balance,
    or that's merely decaying quietly) the right-hand side shrinks toward
    zero right along with it, regardless of how large lambda_edge is set.
    Any edge with mu_k near zero and any nonzero variance still trips the
    flag at *any* lambda_edge. There is no threshold value that reliably
    disables growth from this rule alone -- set enable_instability_growth
    = False instead.
    """
    window_T: int = 50          # steps of Δw history required before an edge is eligible for growth/prune checks
    lambda_edge: float = 10.0   # instability flag: sigma_k^2 > lambda_edge * |mu_k|  (calibrated: p90 of sigma_k^2/|mu_k| ~= 10.5)
    tau_w: float = 0.16         # prune candidate: |w_k| <= tau_w                    (calibrated: p10 of |weight| ~= 0.16)
    tau_delta: float = 0.0002   # prune candidate: |mu_k| <= tau_delta               (calibrated: p10 of |mu_k| ~= 0.0002)
    prune_period_s: int = 200   # structural-edit ("slow") clock, in circuit steps
    prune_fraction_eta: float = 0.3   # only prune this fraction of the flagged set per cycle
    p_rand: float = 0.05        # Bernoulli chance of a random-growth event per structural cycle
    rho_rand: float = 0.02      # random edges proposed per trigger = rho_rand * active neuron count
    relay_init_scale: float = 0.05    # relay/random edge init weight = scale * conn_weight_max
    conn_a_plus: float = 0.05   # LTP rate for internal-connection STDP
    conn_a_minus: float = 0.05  # LTD rate for internal-connection STDP
    conn_weight_decay: float = 0.0001   # matches BiologicalNeuron.weight_decay's convention/scale
    conn_weight_min: float = 0.0
    conn_weight_max: float = 10.0
    seed: Optional[int] = None

    # Explicit, orthogonal per-mechanism kill switches for building ablation
    # arms. All default True (full pathway). Prefer these over threshold
    # extremes -- see the lambda_edge note above for why that doesn't work.
    enable_connection_stdp: bool = True         # False => internal connections never change at all (frozen topology + weights)
    enable_instability_growth: bool = True      # False => SMGrNN relay insertion never fires
    enable_random_growth: bool = True           # False => secondary Bernoulli exploratory growth never fires
    enable_pruning: bool = True                 # False => mandatory pruning never fires
    enable_synaptic_normalization: bool = True  # False => SORN synaptic normalization never fires


class SelfConnectingPathway:
    """
    Structure-pathway-only manager, layered on top of an existing
    NeuralCircuit rather than baked into it, so it can be toggled on/off
    independently -- matching the spec's "one unit type, four toggleable
    properties" design law (SONU_SPEC_V2.md Section 0) and the ablation
    ladder's requirement to isolate +connect from everything else.

    Building ablation arms (e.g. for the Section 6 ladder): use
    StructuralPlasticityConfig's enable_* flags, not extreme threshold
    values -- lambda_edge in particular cannot be raised high enough to
    reliably disable instability growth (see the config docstring). For
    example, an "STDP-only, frozen topology" arm to compare against a bare
    circuit and the full pathway:
        StructuralPlasticityConfig(enable_instability_growth=False,
                                    enable_random_growth=False,
                                    enable_pruning=False)

    Usage:
        circuit = NeuralCircuit(num_neurons=20, input_channels=10, max_neurons=60)
        pathway = SelfConnectingPathway(circuit)
        for t in range(n_steps):
            output_spikes = circuit.step(input_spikes[t])
            events = pathway.step(output_spikes)   # {'grown_relay': [...], 'grown_random': [...],
                                                     #  'pruned': [...], 'orphans_removed': [...]}
    """

    def __init__(self, circuit: NeuralCircuit, config: Optional[StructuralPlasticityConfig] = None):
        self.circuit = circuit
        self.cfg = config or StructuralPlasticityConfig()

        # Only instability growth calls circuit.add_neuron() (random growth
        # only adds edges between existing neurons), so pool room is only
        # required when it's actually enabled -- a growth-free ablation arm
        # (enable_instability_growth=False) shouldn't be forced to allocate
        # capacity it will never use.
        if self.cfg.enable_instability_growth and circuit.max_neurons <= circuit.num_neurons:
            raise ValueError(
                "circuit.max_neurons must exceed circuit.num_neurons when "
                "enable_instability_growth is True; either construct the circuit "
                "with NeuralCircuit(..., max_neurons=<capacity>) or set "
                "config.enable_instability_growth = False for a growth-free run."
            )
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.default_rng(self.cfg.seed)

        self._edge_history: Dict[int, Deque[float]] = {}
        self._step_count = 0
        self._relay_ids: Set[int] = set()

        # SORN synaptic normalization target: each neuron's *own* initial
        # afferent (incoming internal-connection) weight sum, not a shared
        # global constant -- different neurons can legitimately start with
        # very different in-degree.
        self._afferent_targets: Dict[int, float] = {
            i: self._afferent_sum(i) for i in range(circuit.num_neurons)
        }

    # ---- per-step: connection STDP + history bookkeeping (fast clock) ----

    def step(self, output_spikes: np.ndarray) -> Dict[str, list]:
        """
        Call once per circuit.step(), passing the spikes it just returned.

        Connection STDP and Δw-history recording run every call (the
        "fast," content-learning-rate clock). Growth, pruning, orphan
        removal, and synaptic normalization only run every
        cfg.prune_period_s calls -- a deliberately slower clock, per the
        spec's stress test (Section 5a): gating structural edits to a
        slower cadence than the plasticity that drives them is what stops
        structure from chasing activations that haven't settled yet.
        """
        self._apply_connection_stdp(output_spikes)
        self._step_count += 1

        events: Dict[str, list] = {
            'grown_relay': [], 'grown_random': [], 'pruned': [], 'orphans_removed': []
        }
        if self._step_count % self.cfg.prune_period_s == 0:
            if self.cfg.enable_instability_growth:
                events['grown_relay'] = self._instability_growth()
            if self.cfg.enable_random_growth:
                events['grown_random'] = self._random_growth()
            if self.cfg.enable_pruning:
                events['pruned'] = self._prune()
            # Always safe to call: a no-op whenever _relay_ids is empty,
            # which it always is if instability growth was never enabled.
            events['orphans_removed'] = self._remove_orphans()
            if self.cfg.enable_synaptic_normalization:
                self._synaptic_normalization()
        return events

    def _apply_connection_stdp(self, output_spikes: np.ndarray) -> None:
        """
        Applies STDP (LTP/LTD) plus passive weight decay to every active
        internal connection, once per circuit step -- the "fast clock" base
        learning rule for the +connect ladder rung (module docstring,
        adaptation #2). Decay mirrors BiologicalNeuron.stdp()'s own
        weight_decay convention, and is what lets an edge that never gets
        coincident pre/post spikes (e.g. a freshly grown relay hop nothing
        reinforces) drift down toward the prune threshold instead of
        sitting frozen at its init weight forever -- without it, mandatory
        pruning (Section 1) would have nothing to act on for an edge that
        simply never fires.

        No-op entirely when cfg.enable_connection_stdp is False: connection
        weights are then frozen at whatever they were constructed/grown
        with, and no Δw history accumulates, so growth/prune naturally have
        nothing to act on either -- the correct "topology and weights both
        frozen" isolation arm.
        """
        if not self.cfg.enable_connection_stdp:
            return
        cfg = self.cfg
        for source_id in range(self.circuit.num_neurons):
            if not self.circuit.neuron_active[source_id]:
                continue
            pre_trace = self._trace_of(source_id)
            for connection in self.circuit.connections[source_id]:
                if not connection.active or not self.circuit.neuron_active[connection.target_id]:
                    continue
                post_trace = self._trace_of(connection.target_id)

                stdp_delta = 0.0
                if output_spikes[connection.target_id]:        # causal (pre before post): LTP
                    stdp_delta += cfg.conn_a_plus * pre_trace
                if output_spikes[source_id]:                   # acausal (post before pre): LTD
                    stdp_delta -= cfg.conn_a_minus * post_trace

                old_weight = connection.weight
                new_weight = (old_weight + stdp_delta) * (1.0 - cfg.conn_weight_decay)
                connection.weight = float(np.clip(new_weight, cfg.conn_weight_min, cfg.conn_weight_max))

                history = self._edge_history.get(connection.conn_id)
                if history is None:
                    history = deque(maxlen=cfg.window_T)
                    self._edge_history[connection.conn_id] = history
                history.append(connection.weight - old_weight)  # total Δw this step: STDP + decay, post-clip

    def _trace_of(self, neuron_id: int) -> float:
        trace = self.circuit.neurons[neuron_id].post_trace
        return float(trace.item()) if hasattr(trace, 'item') else float(trace)

    # ---- slow clock: structural edits ----

    def _instability_growth(self) -> List[Tuple[int, int, int]]:
        """SMGrNN instability rule: |mu_k| < 0.5*sigma_k AND sigma_k^2 > lambda_edge*|mu_k|."""
        cfg = self.cfg
        grown = []
        for source_id in range(self.circuit.num_neurons):
            if not self.circuit.neuron_active[source_id]:
                continue
            for connection in list(self.circuit.connections[source_id]):
                if not connection.active:
                    continue
                history = self._edge_history.get(connection.conn_id)
                if history is None or len(history) < cfg.window_T:
                    continue  # not enough samples yet for a meaningful window

                mu = float(np.mean(history))
                sigma = float(np.std(history))
                if abs(mu) < 0.5 * sigma and sigma ** 2 > cfg.lambda_edge * abs(mu):
                    relay_id = self._insert_relay(source_id, connection.target_id, connection.delay)
                    if relay_id is not None:
                        grown.append((source_id, relay_id, connection.target_id))
        return grown

    def _insert_relay(self, source_id: int, target_id: int, base_delay: int) -> Optional[int]:
        """Insert a relay node on a parallel two-step path i -> relay -> j. The original edge is kept."""
        relay_id = self.circuit.add_neuron()
        if relay_id is None:
            return None  # pool at capacity: skip this growth event rather than force it

        # See module docstring, adaptation #4: start the relay driven by the
        # relayed signal, not an independently-learned, unrelated input tuning.
        relay_neuron = self.circuit.neurons[relay_id]
        relay_neuron.set_weights(np.zeros(self.circuit.input_channels, dtype=np.float32))

        init_w = self.cfg.relay_init_scale * self.cfg.conn_weight_max
        self.circuit.connect(source_id, relay_id, weight=init_w, delay=base_delay)
        self.circuit.connect(relay_id, target_id, weight=init_w, delay=base_delay)

        self._relay_ids.add(relay_id)
        self._afferent_targets[relay_id] = init_w
        return relay_id

    def _random_growth(self) -> List[Tuple[int, int]]:
        """Secondary rule: Bernoulli(p_rand) trigger proposing rho_rand * N new random edges."""
        cfg = self.cfg
        if self._rng.random() >= cfg.p_rand:
            return []

        active_ids = [i for i in range(self.circuit.num_neurons) if self.circuit.neuron_active[i]]
        if len(active_ids) < 2:
            return []

        n_new = max(1, round(cfg.rho_rand * len(active_ids)))
        created = []
        for _ in range(n_new):
            source_id, target_id = self._rng.sample(active_ids, 2)
            weight = float(self._np_rng.uniform(0.0, cfg.relay_init_scale * cfg.conn_weight_max))
            self.circuit.connect(source_id, target_id, weight=weight, delay=1)
            created.append((source_id, target_id))
        return created

    def _prune(self) -> List[Tuple[int, int]]:
        """Mandatory pruning: |w_k| <= tau_w AND |mu_k| <= tau_delta, only eta_prune fraction per cycle."""
        cfg = self.cfg
        flagged: List[Tuple[int, Connection]] = []
        for source_id in range(self.circuit.num_neurons):
            if not self.circuit.neuron_active[source_id]:
                continue
            for connection in self.circuit.connections[source_id]:
                if not connection.active:
                    continue
                history = self._edge_history.get(connection.conn_id)
                if history is None or len(history) < cfg.window_T:
                    continue
                mu = float(np.mean(history))
                if abs(connection.weight) <= cfg.tau_w and abs(mu) <= cfg.tau_delta:
                    flagged.append((source_id, connection))

        n_to_prune = min(round(len(flagged) * cfg.prune_fraction_eta), len(flagged))
        to_prune = self._rng.sample(flagged, n_to_prune) if n_to_prune else []

        pruned = []
        for source_id, connection in to_prune:
            connection.active = False
            self._edge_history.pop(connection.conn_id, None)
            pruned.append((source_id, connection.target_id))
        return pruned

    def _remove_orphans(self) -> List[int]:
        """
        Remove (deactivate) orphaned relay nodes -- zero in- or out-degree
        among active connections. Scoped to relay neurons only; see module
        docstring adaptation #5 for why this can't apply to the circuit's
        original neurons without misfiring on intentionally input-only or
        output-only nodes.
        """
        if not self._relay_ids:
            return []

        in_degree = {i: 0 for i in self._relay_ids}
        out_degree = {i: 0 for i in self._relay_ids}
        for source_id in range(self.circuit.num_neurons):
            for connection in self.circuit.connections[source_id]:
                if not connection.active:
                    continue
                if connection.target_id in in_degree:
                    in_degree[connection.target_id] += 1
                if source_id in out_degree:
                    out_degree[source_id] += 1

        removed = []
        for relay_id in list(self._relay_ids):
            if not self.circuit.neuron_active[relay_id]:
                continue
            if in_degree[relay_id] == 0 or out_degree[relay_id] == 0:
                self.circuit.deactivate_neuron(relay_id)
                self._afferent_targets.pop(relay_id, None)
                self._relay_ids.discard(relay_id)
                removed.append(relay_id)
        return removed

    # ---- SORN homeostasis ----
    #
    # Spec Section 1 requires synaptic normalization + intrinsic plasticity
    # "if the structure pathway is Hebbian-driven" (it is here). Intrinsic
    # plasticity -- adjusting firing threshold to hold a target average
    # activity -- already exists in neuron.py (theta / tau_theta /
    # theta_increment: an activity-dependent dynamic threshold) and is
    # reused as-is, unmodified. Only synaptic normalization is new.

    def _afferent_sum(self, target_id: int) -> float:
        total = 0.0
        for source_id in range(self.circuit.num_neurons):
            for connection in self.circuit.connections[source_id]:
                if connection.active and connection.target_id == target_id:
                    total += connection.weight
        return total

    def _synaptic_normalization(self) -> None:
        """Rescale each neuron's active incoming weights back toward its own target sum."""
        for target_id in range(self.circuit.num_neurons):
            if not self.circuit.neuron_active[target_id]:
                continue
            target = self._afferent_targets.get(target_id)
            if not target:  # None or 0.0: nothing to normalize against
                continue
            current = self._afferent_sum(target_id)
            if current == 0.0:
                continue
            scale = target / current
            for source_id in range(self.circuit.num_neurons):
                for connection in self.circuit.connections[source_id]:
                    if connection.active and connection.target_id == target_id:
                        connection.weight = float(np.clip(
                            connection.weight * scale,
                            self.cfg.conn_weight_min, self.cfg.conn_weight_max
                        ))
