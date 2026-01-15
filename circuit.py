"""
NeuralCircuit: Infrastructure for managing networks of BiologicalNeurons

Provides a container class for populations of neurons with:
- Inter-neuron connectivity with axonal delays
- Spike buffering and routing
- Lateral inhibition
- Network-level step function
"""

import numpy as np
from typing import List, Optional, Dict
from neuron import BiologicalNeuron

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Connection:
    """
    Represents a synaptic connection between two neurons.

    Attributes:
        source_id: Index of source (pre-synaptic) neuron
        target_id: Index of target (post-synaptic) neuron
        weight: Synaptic strength
        delay: Axonal delay in time steps
    """

    def __init__(self, source_id: int, target_id: int, weight: float, delay: int):
        """
        Initialize a synaptic connection.

        Args:
            source_id: Pre-synaptic neuron index
            target_id: Post-synaptic neuron index
            weight: Connection strength
            delay: Transmission delay in time steps (ms if dt=1.0)
        """
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.delay = delay


class SpikeBuffer:
    """
    Circular buffer for storing spikes with delays.

    Handles axonal transmission delays by storing spikes that will
    arrive at future time steps.
    """

    def __init__(self, max_delay: int, num_neurons: int):
        """
        Initialize spike buffer.

        Args:
            max_delay: Maximum axonal delay in time steps
            num_neurons: Number of neurons in the circuit
        """
        self.max_delay = max_delay
        self.num_neurons = num_neurons

        # Buffer stores spikes for each future time step
        # buffer[i] contains spikes arriving i steps in the future
        self.buffer = [np.zeros(num_neurons, dtype=np.float64)
                       for _ in range(max_delay + 1)]

        self.current_index = 0

    def add_spike(self, neuron_id: int, weight: float, delay: int) -> None:
        """
        Add a spike to the buffer that will arrive after 'delay' steps.

        Args:
            neuron_id: Target neuron index
            weight: Synaptic weight (spike amplitude)
            delay: Number of time steps until arrival
        """
        if delay < 0 or delay > self.max_delay:
            raise ValueError(f"Delay {delay} outside valid range [0, {self.max_delay}]")

        # Calculate buffer index for this delay
        target_index = (self.current_index + delay) % (self.max_delay + 1)
        self.buffer[target_index][neuron_id] += weight

    def get_current_spikes(self) -> np.ndarray:
        """
        Get spikes arriving at the current time step.

        Returns:
            Array of spike inputs for each neuron
        """
        return self.buffer[self.current_index].copy()

    def advance(self) -> None:
        """
        Advance to next time step.

        Clears current buffer slot and moves index forward.
        """
        # Clear current slot (these spikes have been delivered)
        self.buffer[self.current_index].fill(0.0)

        # Move to next time step
        self.current_index = (self.current_index + 1) % (self.max_delay + 1)


class NeuralCircuit:
    """
    Container for managing a network of BiologicalNeurons.

    Provides:
    - Population of neurons
    - Inter-neuron connectivity with delays
    - Lateral inhibition
    - Network-level dynamics
    """

    def __init__(
        self,
        num_neurons: int,
        input_channels: int,
        dt: float = 1.0,
        max_delay: int = 10,
        neuron_params: Optional[Dict] = None
    ):
        """
        Initialize neural circuit.

        Args:
            num_neurons: Number of neurons in the circuit
            input_channels: Number of external input channels
            dt: Time step (ms)
            max_delay: Maximum axonal delay (time steps)
            neuron_params: Optional dict of parameters to pass to all neurons
        """
        self.num_neurons = num_neurons
        self.input_channels = input_channels
        self.dt = dt
        self.max_delay = max_delay

        # Default neuron parameters
        default_params = {
            'n_inputs': input_channels,
            'dt': dt,
            'tau_m': 20.0,
            'tau_trace': 20.0,
            'a_plus': 0.1,  # Increased from 0.01 for better learning
            'a_minus': 0.1,  # Increased from 0.01 for better learning
            'weight_min': 0.0,
            'weight_max': 10.0,  # Increased from 1.0 for more capacity
        }

        # Override with user-provided parameters
        if neuron_params is not None:
            default_params.update(neuron_params)

        # Create population of neurons
        self.neurons: List[BiologicalNeuron] = []
        for i in range(num_neurons):
            neuron = BiologicalNeuron(**default_params)
            self.neurons.append(neuron)

        # Connection matrix: connections[i] = list of connections FROM neuron i
        self.connections: List[List[Connection]] = [[] for _ in range(num_neurons)]

        # Spike buffer for handling axonal delays
        self.spike_buffer = SpikeBuffer(max_delay, num_neurons)

        # Lateral inhibition parameters
        self.inhibition_strength = 0.0  # Default: no inhibition

        # State tracking
        self.current_output_spikes = np.zeros(num_neurons, dtype=bool)
        self.time_step = 0

    def connect(
        self,
        source_id: int,
        target_id: int,
        weight: float,
        delay: int = 1
    ) -> None:
        """
        Create a synaptic connection between two neurons.

        Args:
            source_id: Index of pre-synaptic neuron (0 to num_neurons-1)
            target_id: Index of post-synaptic neuron (0 to num_neurons-1)
            weight: Synaptic strength (can be positive or negative)
            delay: Axonal transmission delay in time steps (default: 1)

        Raises:
            ValueError: If neuron indices are out of range
        """
        if source_id < 0 or source_id >= self.num_neurons:
            raise ValueError(f"Source neuron {source_id} out of range [0, {self.num_neurons})")
        if target_id < 0 or target_id >= self.num_neurons:
            raise ValueError(f"Target neuron {target_id} out of range [0, {self.num_neurons})")
        if delay < 0 or delay > self.max_delay:
            raise ValueError(f"Delay {delay} outside valid range [0, {self.max_delay}]")

        # Create and store connection
        connection = Connection(source_id, target_id, weight, delay)
        self.connections[source_id].append(connection)

    def set_inhibition(self, strength: float) -> None:
        """
        Set lateral inhibition strength.

        When a neuron fires, it will decrease the membrane potential
        of all other neurons by this amount.

        Args:
            strength: Inhibition strength (mV). Positive values inhibit.
                     Typical range: 0.0 (no inhibition) to 5.0 (strong)
        """
        self.inhibition_strength = strength

    def connect_all_to_all(
        self,
        weight: float,
        delay: int = 1,
        include_self: bool = False
    ) -> None:
        """
        Create all-to-all connections between neurons.

        Args:
            weight: Synaptic weight for all connections
            delay: Axonal delay for all connections
            include_self: Whether to include self-connections (recurrent)
        """
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j or include_self:
                    self.connect(i, j, weight, delay)

    def connect_lateral_inhibition(
        self,
        weight: float = -1.0,
        delay: int = 1
    ) -> None:
        """
        Create lateral inhibition connections (all-to-all inhibitory).

        Args:
            weight: Inhibitory weight (should be negative)
            delay: Axonal delay
        """
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    self.connect(i, j, weight, delay)

    def connect_chain(
        self,
        weight: float,
        delay: int = 1,
        bidirectional: bool = False
    ) -> None:
        """
        Connect neurons in a chain: 0->1->2->...->N.

        Args:
            weight: Synaptic weight
            delay: Axonal delay
            bidirectional: If True, also create reverse connections
        """
        for i in range(self.num_neurons - 1):
            self.connect(i, i + 1, weight, delay)
            if bidirectional:
                self.connect(i + 1, i, weight, delay)

    def step(
        self,
        input_spikes: np.ndarray,
        I_ext: Optional[np.ndarray] = None,
        learning: bool = True
    ) -> np.ndarray:
        """
        Execute one time step of the circuit.

        Process:
        1. Get delayed spikes from buffer (inter-neuron inputs)
        2. Apply lateral inhibition from previous step
        3. Update each neuron with external + internal inputs
        4. Route output spikes to connections (with delays)
        5. Advance spike buffer

        Args:
            input_spikes: External input spikes (shape: input_channels)
            I_ext: Optional external current for each neuron (shape: num_neurons)
            learning: Whether to enable STDP learning

        Returns:
            Boolean array of output spikes (shape: num_neurons)
        """
        if input_spikes.shape[0] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, "
                           f"got {input_spikes.shape[0]}")

        # Prepare external current (default to 0)
        if I_ext is None:
            I_ext = np.zeros(self.num_neurons)
        elif I_ext.shape[0] != self.num_neurons:
            raise ValueError(f"Expected {self.num_neurons} external currents, "
                           f"got {I_ext.shape[0]}")

        # 1. Get delayed spikes arriving at this time step
        internal_spikes = self.spike_buffer.get_current_spikes()

        # 2. Apply lateral inhibition from neurons that fired last step
        # (direct membrane potential modification)
        if self.inhibition_strength > 0.0:
            num_spiked_last_step = np.sum(self.current_output_spikes)
            if num_spiked_last_step > 0:
                # Each neuron that fired inhibits all others
                for i, neuron in enumerate(self.neurons):
                    if self.current_output_spikes[i]:
                        # This neuron fired - it inhibits others
                        for j, other_neuron in enumerate(self.neurons):
                            if i != j:
                                # Decrease membrane potential (inhibition)
                                # Handle both GPU and CPU tensors
                                if isinstance(other_neuron.v, torch.Tensor):
                                    other_neuron.v = other_neuron.v - self.inhibition_strength
                                else:
                                    other_neuron.v -= self.inhibition_strength

        # 3. Update each neuron
        output_spikes = np.zeros(self.num_neurons, dtype=bool)

        # Convert input_spikes to torch tensor once if any neuron uses GPU
        # Check if we need GPU conversion
        use_gpu = False
        if len(self.neurons) > 0 and hasattr(self.neurons[0], 'device'):
            use_gpu = self.neurons[0].device.type == 'cuda'

        if use_gpu and not isinstance(input_spikes, torch.Tensor):
            input_spikes_tensor = torch.tensor(
                input_spikes,
                dtype=torch.float32,
                device=self.neurons[0].device
            )
        else:
            input_spikes_tensor = input_spikes

        for i, neuron in enumerate(self.neurons):
            # Combine external inputs and internal (delayed) inputs
            # External inputs go through the neuron's learned weights
            # Internal inputs are direct current injections (already weighted)

            # Update neuron with external input and internal current
            # I_ext is already a scalar, so no conversion needed
            spike = neuron.step(
                input_spikes=input_spikes_tensor,
                I_ext=float(I_ext[i] + internal_spikes[i]),
                learning=learning
            )

            # Spike is already a bool from neuron.step()
            output_spikes[i] = bool(spike)

        # 4. Route output spikes through connections (with delays)
        for i, spiked in enumerate(output_spikes):
            if spiked:
                # This neuron fired - send spike to all its targets
                for connection in self.connections[i]:
                    self.spike_buffer.add_spike(
                        neuron_id=connection.target_id,
                        weight=connection.weight,
                        delay=connection.delay
                    )

        # 5. Advance spike buffer to next time step
        self.spike_buffer.advance()

        # Store current output for lateral inhibition next step
        self.current_output_spikes = output_spikes.copy()

        # Increment time
        self.time_step += 1

        return output_spikes

    def reset_state(self) -> None:
        """
        Reset all neurons and circuit state to initial conditions.

        Clears:
        - All neuron internal states (v, u, theta, traces)
        - Spike buffer
        - Output spike history
        - Time counter

        Preserves:
        - Learned weights
        - Connections
        - Circuit structure
        """
        # Reset all neurons
        for neuron in self.neurons:
            neuron.reset_state()

        # Clear spike buffer
        for i in range(len(self.spike_buffer.buffer)):
            self.spike_buffer.buffer[i].fill(0.0)
        self.spike_buffer.current_index = 0

        # Reset state tracking
        self.current_output_spikes = np.zeros(self.num_neurons, dtype=bool)
        self.time_step = 0

    def get_states(self) -> List[Dict]:
        """
        Get current state of all neurons.

        Returns:
            List of state dictionaries, one per neuron
        """
        return [neuron.get_state() for neuron in self.neurons]

    def get_weights(self, neuron_id: int) -> np.ndarray:
        """
        Get input weights for a specific neuron.

        Args:
            neuron_id: Index of neuron

        Returns:
            Array of input weights
        """
        if neuron_id < 0 or neuron_id >= self.num_neurons:
            raise ValueError(f"Neuron {neuron_id} out of range [0, {self.num_neurons})")

        # Handle GPU weights if necessary
        weights = self.neurons[neuron_id].weights
        if TORCH_AVAILABLE and isinstance(weights, torch.Tensor):
            return weights.cpu().numpy()
        return weights.copy()

    def set_weights(self, neuron_id: int, weights: np.ndarray) -> None:
        """
        Set input weights for a specific neuron.

        Args:
            neuron_id: Index of neuron
            weights: New weight values
        """
        if neuron_id < 0 or neuron_id >= self.num_neurons:
            raise ValueError(f"Neuron {neuron_id} out of range [0, {self.num_neurons})")

        if weights.shape[0] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} weights, got {weights.shape[0]}")

        # Handle GPU weights if necessary
        neuron = self.neurons[neuron_id]
        if hasattr(neuron, 'device') and neuron.device.type == 'cuda':
            neuron.weights = torch.from_numpy(weights).float().to(neuron.device)
        else:
            neuron.weights = weights.copy()

    def get_connection_matrix(self) -> np.ndarray:
        """
        Get connection weight matrix (without delays).

        Returns:
            Matrix where element [i,j] is the weight from neuron i to neuron j.
            Multiple connections are summed.
        """
        matrix = np.zeros((self.num_neurons, self.num_neurons))

        for source_id in range(self.num_neurons):
            for connection in self.connections[source_id]:
                matrix[source_id, connection.target_id] += connection.weight

        return matrix

    def get_num_connections(self) -> int:
        """
        Get total number of connections in the circuit.

        Returns:
            Total connection count
        """
        return sum(len(conn_list) for conn_list in self.connections)

    def __repr__(self) -> str:
        """String representation of the circuit."""
        num_connections = self.get_num_connections()
        return (f"NeuralCircuit(neurons={self.num_neurons}, "
                f"inputs={self.input_channels}, "
                f"connections={num_connections}, "
                f"max_delay={self.max_delay}ms, "
                f"inhibition={self.inhibition_strength})")

    def summary(self) -> str:
        """
        Get detailed summary of circuit configuration.

        Returns:
            Multi-line string describing the circuit
        """
        num_connections = self.get_num_connections()

        # Count connection statistics
        delays = []
        weights = []
        for source_connections in self.connections:
            for conn in source_connections:
                delays.append(conn.delay)
                weights.append(conn.weight)

        summary_lines = [
            "="*60,
            "NEURAL CIRCUIT SUMMARY",
            "="*60,
            f"Neurons: {self.num_neurons}",
            f"Input Channels: {self.input_channels}",
            f"Time Step (dt): {self.dt}ms",
            f"Max Delay: {self.max_delay}ms",
            f"",
            f"Connections: {num_connections}",
        ]

        if num_connections > 0:
            summary_lines.extend([
                f"  Avg Weight: {np.mean(weights):.4f}",
                f"  Weight Range: [{np.min(weights):.4f}, {np.max(weights):.4f}]",
                f"  Avg Delay: {np.mean(delays):.2f}ms",
                f"  Delay Range: [{np.min(delays)}, {np.max(delays)}]ms",
            ])

        summary_lines.extend([
            f"",
            f"Lateral Inhibition: {self.inhibition_strength:.3f}mV",
            f"Current Time Step: {self.time_step}",
            "="*60,
        ])

        return "\n".join(summary_lines)

