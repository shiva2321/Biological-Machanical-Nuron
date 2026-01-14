"""
BiologicalNeuron: A biologically plausible neuron model with LIF dynamics,
adaptation, and STDP learning.
"""

import numpy as np


class BiologicalNeuron:
    """
    Leaky Integrate-and-Fire neuron with adaptation and Spike-Timing-Dependent Plasticity.

    This neuron model implements:
    - LIF dynamics with membrane potential decay
    - Adaptive threshold and adaptation current
    - STDP-based synaptic plasticity
    """

    def __init__(
        self,
        n_inputs: int,
        tau_m: float = 20.0,      # Membrane time constant (ms)
        tau_u: float = 100.0,     # Adaptation time constant (ms)
        tau_theta: float = 1000.0,  # Threshold adaptation time constant (ms)
        tau_trace: float = 20.0,  # STDP trace time constant (ms)
        dt: float = 1.0,          # Time step (ms)
        v_rest: float = -70.0,    # Resting potential (mV)
        v_reset: float = -75.0,   # Reset potential after spike (mV)
        theta_base: float = -50.0,  # Base threshold (mV)
        u_increment: float = 5.0,   # Adaptation increment on spike
        theta_increment: float = 2.0,  # Threshold increment on spike
        a_plus: float = 0.01,     # STDP potentiation learning rate
        a_minus: float = 0.01,    # STDP depression learning rate
        weight_min: float = 0.0,  # Minimum synaptic weight
        weight_max: float = 1.0   # Maximum synaptic weight
    ):
        """
        Initialize the biological neuron with LIF dynamics.

        Args:
            n_inputs: Number of input synaptic connections
            tau_m: Membrane potential decay time constant
            tau_u: Adaptation current decay time constant
            tau_theta: Dynamic threshold decay time constant
            tau_trace: Eligibility trace decay time constant
            dt: Integration time step
            v_rest: Resting membrane potential
            v_reset: Reset potential after spike
            theta_base: Base firing threshold
            u_increment: Adaptation increase per spike
            theta_increment: Threshold increase per spike
            a_plus: Learning rate for potentiation (pre before post)
            a_minus: Learning rate for depression (post before pre)
            weight_min: Minimum weight value (clipping)
            weight_max: Maximum weight value (clipping)
        """
        self.n_inputs = n_inputs
        self.dt = dt

        # Time constants
        self.tau_m = tau_m
        self.tau_u = tau_u
        self.tau_theta = tau_theta
        self.tau_trace = tau_trace

        # Voltage parameters
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.theta_base = theta_base

        # Adaptation parameters
        self.u_increment = u_increment
        self.theta_increment = theta_increment

        # STDP parameters
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.weight_min = weight_min
        self.weight_max = weight_max

        # State variables
        self.v: float = v_rest  # Membrane potential
        self.u: float = 0.0     # Adaptation current
        self.theta: float = 0.0  # Dynamic threshold (relative to theta_base)

        # Synaptic weights and traces
        self.weights: np.ndarray = np.random.uniform(
            0.3, 0.7, size=n_inputs
        ).astype(np.float64)
        self.trace: np.ndarray = np.zeros(n_inputs, dtype=np.float64)

        # Post-synaptic trace for STDP
        self.post_trace: float = 0.0

    def update(self, input_spikes: np.ndarray, I_ext: float = 0.0) -> bool:
        """
        Update neuron state using Euler integration and check for spike.

        Implements the differential equations:
        dv/dt = (-v + v_rest + I_syn + I_ext - u) / tau_m
        du/dt = -u / tau_u
        d(theta)/dt = -theta / tau_theta

        Args:
            input_spikes: Binary array of input spikes (shape: n_inputs)
            I_ext: External current injection

        Returns:
            bool: True if neuron spiked, False otherwise
        """
        # Compute synaptic current: weighted sum of input spikes
        I_syn = np.dot(self.weights, input_spikes)

        # Euler integration for membrane potential
        # dv/dt = (-v + v_rest + I_syn + I_ext - u) / tau_m
        dv = ((-self.v + self.v_rest + I_syn + I_ext - self.u) / self.tau_m) * self.dt
        self.v += dv

        # Euler integration for adaptation current
        # du/dt = -u / tau_u
        du = (-self.u / self.tau_u) * self.dt
        self.u += du

        # Euler integration for dynamic threshold
        # d(theta)/dt = -theta / tau_theta
        dtheta = (-self.theta / self.tau_theta) * self.dt
        self.theta += dtheta

        # Update pre-synaptic traces for STDP
        # d(trace)/dt = -trace / tau_trace + spike
        self.trace *= np.exp(-self.dt / self.tau_trace)
        self.trace += input_spikes  # Increment trace when input spikes

        # Update post-synaptic trace
        self.post_trace *= np.exp(-self.dt / self.tau_trace)

        # Check for spike: v > theta_base + theta + u
        effective_threshold = self.theta_base + self.theta
        if self.v > effective_threshold:
            # Spike occurred!
            self.v = self.v_reset  # Reset membrane potential
            self.u += self.u_increment  # Increase adaptation
            self.theta += self.theta_increment  # Increase threshold
            self.post_trace += 1.0  # Increment post-synaptic trace
            return True

        return False

    def stdp(self, input_spikes: np.ndarray, output_spike: bool) -> None:
        """
        Apply Spike-Timing-Dependent Plasticity (STDP) to synaptic weights.
        FIXED: Causal (Pre->Post) strengthens, Acausal (Post->Pre) weakens.

        Hebbian learning: "Cells that fire together, wire together"
        - Potentiation (LTP): pre-synaptic spike BEFORE post-synaptic spike -> increase weight
        - Depression (LTD): pre-synaptic spike AFTER post-synaptic spike -> decrease weight

        Standard STDP rule applied at spike times:
        - When POST spikes: w += A+ * trace_pre  (potentiation if pre was recently active)
        - When PRE spikes:  w += -A- * trace_post (depression if post was recently active)

        NOTE: This method assumes update() has already been called and traces are current.
        For correct STDP, use step() which handles trace timing properly.

        Args:
            input_spikes: Binary array of input spikes (shape: n_inputs)
            output_spike: Whether the output neuron spiked
        """
        # For backward compatibility, try to reconstruct old traces
        # This is approximate since we don't know exact previous values
        decay_factor = np.exp(-self.dt / self.tau_trace)

        # Estimate old pre-trace (before current input and decay)
        old_trace = (self.trace - input_spikes) / decay_factor if decay_factor > 0 else self.trace - input_spikes
        old_trace = np.maximum(old_trace, 0)  # Ensure non-negative

        # Estimate old post-trace
        spike_addition = 1.0 if output_spike else 0.0
        old_post_trace = (self.post_trace - spike_addition) / decay_factor if decay_factor > 0 else self.post_trace - spike_addition
        old_post_trace = max(old_post_trace, 0)  # Ensure non-negative

        # Apply STDP with estimated old traces
        self._stdp_with_old_traces(input_spikes, output_spike, old_trace, old_post_trace)

    def step(self, input_spikes: np.ndarray, I_ext: float = 0.0,
             learning: bool = True) -> bool:
        """
        Perform a full time step: update dynamics and apply STDP.

        Args:
            input_spikes: Binary array of input spikes (shape: n_inputs)
            I_ext: External current injection
            learning: Whether to apply STDP learning

        Returns:
            bool: True if neuron spiked, False otherwise
        """
        # Save traces BEFORE update for STDP
        if learning:
            old_trace = self.trace.copy()
            old_post_trace = self.post_trace

        # Update neuron state
        output_spike = self.update(input_spikes, I_ext)

        # Apply STDP if learning is enabled, using OLD traces
        if learning:
            self._stdp_with_old_traces(input_spikes, output_spike, old_trace, old_post_trace)

        return output_spike

    def _stdp_with_old_traces(self, input_spikes: np.ndarray, output_spike: bool,
                               old_trace: np.ndarray, old_post_trace: float) -> None:
        """
        Apply STDP using traces from before the current spike.

        Args:
            input_spikes: Current input spikes
            output_spike: Whether output spiked this timestep
            old_trace: Pre-synaptic trace before this timestep
            old_post_trace: Post-synaptic trace before this timestep
        """
        # Potentiation: Output spike occurs, check if inputs were recently active
        # Use OLD pre-trace (before current input was added)
        if output_spike:
            dw_plus = self.a_plus * old_trace
            self.weights += dw_plus

        # Depression: Input spike occurs, check if output was recently active
        # Use OLD post-trace (before current output spike)
        if np.any(input_spikes > 0):
            dw_minus = -self.a_minus * old_post_trace * input_spikes
            self.weights += dw_minus

        # CRITICAL: Clip weights to prevent explosion
        self.weights = np.clip(self.weights, self.weight_min, self.weight_max)

    def reset_state(self) -> None:
        """Reset all state variables to their initial values."""
        self.v = self.v_rest
        self.u = 0.0
        self.theta = 0.0
        self.trace = np.zeros(self.n_inputs, dtype=np.float64)
        self.post_trace = 0.0

    def get_state(self) -> dict:
        """
        Get current state of the neuron.

        Returns:
            dict: Dictionary containing all state variables
        """
        return {
            'v': self.v,
            'u': self.u,
            'theta': self.theta,
            'effective_threshold': self.theta_base + self.theta,
            'weights': self.weights.copy(),
            'trace': self.trace.copy(),
            'post_trace': self.post_trace
        }

