"""
Biological Neuron: PyTorch + CUDA Implementation

High-performance GPU-accelerated implementation of a Leaky Integrate-and-Fire (LIF)
neuron with Spike-Timing-Dependent Plasticity (STDP) learning.

Optimized for NVIDIA GPUs (RTX 3060, 3090, etc.) using PyTorch CUDA operations.
All computation happens on GPU - tensors only move to CPU when explicitly requested.

Key Features:
- LIF dynamics with adaptive threshold
- STDP learning (Hebbian plasticity)
- Full GPU acceleration
- torch.float32 for maximum performance
- Automatic CUDA device detection
"""

import torch
import numpy as np
from typing import Optional, Tuple


class BiologicalNeuron:
    """
    GPU-accelerated Leaky Integrate-and-Fire neuron with STDP.

    Uses PyTorch tensors on CUDA for all operations. Automatically detects
    and uses GPU if available, otherwise falls back to CPU.

    Dynamics:
        dv/dt = (-v + v_rest + I_syn + I_ext - u) / tau_m
        du/dt = -u / tau_u
        dθ/dt = -θ / tau_theta

    Spike condition: v > (theta_base + u + theta)

    STDP:
        - Pre before Post (causal): LTP (strengthen synapse)
        - Post before Pre (acausal): LTD (weaken synapse)

    Example:
        >>> # Automatically uses GPU if available
        >>> neuron = BiologicalNeuron(n_inputs=64)
        >>> print(f"Using device: {neuron.device}")  # cuda or cpu
        >>>
        >>> # All operations on GPU
        >>> input_spikes = torch.randn(64).cuda()
        >>> spike = neuron.update(I_ext=50.0)
        >>>
        >>> # Get state as numpy (moves to CPU)
        >>> state = neuron.get_state()
    """

    def __init__(
        self,
        n_inputs: int,
        tau_m: float = 20.0,
        tau_u: float = 30.0,
        tau_theta: float = 50.0,
        tau_trace: float = 20.0,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        theta_base: float = -55.0,
        u_increment: float = 2.0,
        theta_increment: float = 1.0,
        dt: float = 1.0,
        a_plus: float = 0.05,
        a_minus: float = 0.05,
        weight_min: float = 0.0,
        weight_max: float = 10.0,
        device: Optional[str] = None
    ):
        """
        Initialize GPU-accelerated biological neuron.

        Args:
            n_inputs: Number of input synapses
            tau_m: Membrane time constant (ms)
            tau_u: Adaptation time constant (ms)
            tau_theta: Threshold adaptation time constant (ms)
            tau_trace: STDP trace time constant (ms)
            v_rest: Resting potential (mV)
            v_reset: Reset potential after spike (mV)
            theta_base: Base threshold (mV)
            u_increment: Adaptation increase per spike
            theta_increment: Threshold increase per spike
            dt: Time step (ms)
            a_plus: LTP learning rate
            a_minus: LTD learning rate
            weight_min: Minimum synaptic weight
            weight_max: Maximum synaptic weight
            device: Force device ('cuda' or 'cpu'), None for auto-detect
        """
        # Automatic device detection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Print device info (useful for debugging)
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[GPU] Using CUDA device: {gpu_name}")
        else:
            print(f"[CPU] CUDA not available, using CPU")

        # Store parameters
        self.n_inputs = n_inputs
        self.tau_m = tau_m
        self.tau_u = tau_u
        self.tau_theta = tau_theta
        self.tau_trace = tau_trace
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.theta_base = theta_base
        self.u_increment = u_increment
        self.theta_increment = theta_increment
        self.dt = dt
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.weight_min = weight_min
        self.weight_max = weight_max

        # Learning and retention parameters
        self.learning_rate_decay = 0.9999  # Decay learning rate over time
        self.current_a_plus = a_plus
        self.current_a_minus = a_minus
        self.weight_decay = 0.0001  # L2 regularization to prevent overfitting
        self.homeostatic_scale = 1.0  # Homeostatic plasticity scaling
        self.target_firing_rate = 0.1  # Target firing rate for homeostasis
        self.firing_rate_history = []  # Track firing rate for homeostasis
        self.update_count = 0  # Track number of updates for adaptive learning

        # Initialize state variables as GPU tensors (torch.float32 for speed)
        self.v = torch.tensor(v_rest, dtype=torch.float32, device=self.device)
        self.u = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.theta = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Initialize synaptic weights with Xavier/Glorot initialization (better than random)
        # Xavier init: weights ~ U(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))
        # For single neuron: n_in = n_inputs, n_out = 1
        xavier_bound = np.sqrt(6.0 / (n_inputs + 1))
        self.weights = torch.empty(n_inputs, dtype=torch.float32, device=self.device)
        torch.nn.init.uniform_(self.weights, -xavier_bound * 0.5, xavier_bound * 0.5)
        # Ensure weights are positive and within bounds
        self.weights = torch.clamp(self.weights, weight_min, weight_max * 0.3)  # Start smaller
        
        # Initialize learning and retention parameters (with defaults for backward compatibility)
        self._init_learning_params()

        # Initialize STDP traces
        self.trace = torch.zeros(n_inputs, dtype=torch.float32, device=self.device)
        self.post_trace = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Precompute decay factors (for efficiency)
        self.decay_v = torch.exp(torch.tensor(-dt / tau_m, dtype=torch.float32, device=self.device))
        self.decay_u = torch.exp(torch.tensor(-dt / tau_u, dtype=torch.float32, device=self.device))
        self.decay_theta = torch.exp(torch.tensor(-dt / tau_theta, dtype=torch.float32, device=self.device))
        self.decay_trace = torch.exp(torch.tensor(-dt / tau_trace, dtype=torch.float32, device=self.device))

    def _init_learning_params(self):
        """
        Initialize learning and retention parameters.
        Called after weights are set to ensure backward compatibility with saved brains.
        """
        # Learning and retention parameters
        if not hasattr(self, 'learning_rate_decay'):
            self.learning_rate_decay = 0.9999  # Decay learning rate over time
        if not hasattr(self, 'current_a_plus'):
            self.current_a_plus = getattr(self, 'a_plus', 0.1)
        if not hasattr(self, 'current_a_minus'):
            self.current_a_minus = getattr(self, 'a_minus', 0.1)
        if not hasattr(self, 'weight_decay'):
            self.weight_decay = 0.0001  # L2 regularization to prevent overfitting
        if not hasattr(self, 'homeostatic_scale'):
            self.homeostatic_scale = 1.0  # Homeostatic plasticity scaling
        if not hasattr(self, 'target_firing_rate'):
            self.target_firing_rate = 0.1  # Target firing rate for homeostasis
        if not hasattr(self, 'firing_rate_history'):
            self.firing_rate_history = []  # Track firing rate for homeostasis
        if not hasattr(self, 'update_count'):
            self.update_count = 0  # Track number of updates for adaptive learning
        
        # Weight consolidation (for preventing catastrophic forgetting)
        # Get device from existing weights if available
        if hasattr(self, 'weights'):
            device = self.weights.device if isinstance(self.weights, torch.Tensor) else self.device
            if not hasattr(self, 'consolidated_weights'):
                self.consolidated_weights = self.weights.clone()  # Store consolidated weights
            if not hasattr(self, 'importance_weights'):
                self.importance_weights = torch.ones(
                    self.n_inputs, 
                    dtype=torch.float32, 
                    device=device
                )  # Importance for each weight
        else:
            # Fallback if weights don't exist yet
            if not hasattr(self, 'consolidated_weights'):
                self.consolidated_weights = None
            if not hasattr(self, 'importance_weights'):
                self.importance_weights = torch.ones(
                    self.n_inputs, 
                    dtype=torch.float32, 
                    device=self.device
                )

    def update(self, I_ext: float = 0.0) -> bool:
        """
        Update neuron state for one time step (GPU operation).

        Implements LIF dynamics with adaptation and dynamic threshold.

        Args:
            I_ext: External input current (mV)

        Returns:
            bool: True if neuron spiked, False otherwise
        """
        # Ensure learning parameters are initialized (for backward compatibility with saved brains)
        if not hasattr(self, 'homeostatic_scale'):
            self._init_learning_params()
        
        # Use no_grad context to reduce overhead (we're not backpropagating)
        with torch.no_grad():
            # Convert I_ext to scalar float efficiently
            # Avoid creating tensors - use scalar directly
            if isinstance(I_ext, torch.Tensor):
                I_ext_scalar = float(I_ext.item() if I_ext.numel() == 1 else I_ext[0].item())
            elif isinstance(I_ext, np.ndarray):
                I_ext_scalar = float(I_ext.item() if I_ext.size == 1 else I_ext[0])
            else:
                I_ext_scalar = float(I_ext)

            # Compute synaptic input (I_syn = weights · trace)
            # Apply homeostatic scaling to weights for stability
            scaled_weights = self.weights * self.homeostatic_scale
            I_syn = torch.matmul(scaled_weights, self.trace)

            # Update membrane potential (Euler integration)
            # dv/dt = (-v + v_rest + I_syn + I_ext - u) / tau_m
            # Use scalar I_ext directly (PyTorch broadcasts)
            dv = ((-self.v + self.v_rest + I_syn + I_ext_scalar - self.u) / self.tau_m) * self.dt
            self.v = self.v + dv

            # Update adaptation current (exponential decay)
            # du/dt = -u / tau_u
            self.u = self.u * self.decay_u

            # Update dynamic threshold (exponential decay)
            # dθ/dt = -θ / tau_theta
            self.theta = self.theta * self.decay_theta

            # Check spike condition
            # Note: .item() forces sync, but it's necessary for Python bool
            # For small networks, this overhead is acceptable
            spike_threshold = self.theta_base + self.u + self.theta
            spike_occurred = (self.v > spike_threshold).item()
            
            if spike_occurred:
                # Reset membrane potential (use in-place operation to avoid new tensor)
                self.v.fill_(self.v_reset)

                # Increase adaptation (spike-frequency adaptation)
                self.u = self.u + self.u_increment

                # Increase dynamic threshold (homeostatic regulation)
                self.theta = self.theta + self.theta_increment

                # Update post-synaptic trace
                self.post_trace = self.post_trace + 1.0

                return True

            return False

    def step(
        self,
        input_spikes,
        I_ext: float = 0.0,
        learning: bool = True
    ) -> bool:
        """
        Complete neuron update step with input processing and STDP learning.

        This is the main interface method that handles:
        1. Converting input_spikes to torch tensor
        2. Updating neuron state
        3. Applying STDP learning if enabled

        Args:
            input_spikes: Input spike vector (numpy array or torch tensor)
            I_ext: External current (float, numpy, or torch tensor)
            learning: Whether to apply STDP learning

        Returns:
            bool: True if neuron spiked, False otherwise
        """
        # Update neuron state (this computes I_syn from trace)
        spike = self.update(I_ext=I_ext)

        # Apply STDP learning if enabled
        if learning:
            self.stdp(input_spikes, spike)

        return spike

    def stdp(self, input_spikes: torch.Tensor, output_spike: bool) -> None:
        """
        Apply optimized Spike-Timing-Dependent Plasticity (STDP).

        OPTIMIZATIONS:
        - Reduced homeostatic update frequency (every 500 steps instead of every step)
        - Faster learning rate adaptation (every 200 steps)
        - Simplified importance weight updates
        - Removed redundant device checks
        - Consolidated weight updates

        Args:
            input_spikes: Binary spike vector (0 or 1) for each input
        """
        # Ensure learning parameters are initialized
        if not hasattr(self, 'homeostatic_scale'):
            self._init_learning_params()
        
        with torch.no_grad():
            # Convert input_spikes to tensor (fast path for common case)
            if not isinstance(input_spikes, torch.Tensor):
                input_spikes = torch.tensor(input_spikes, dtype=torch.float32, device=self.device)
            elif input_spikes.device != self.device:
                input_spikes = input_spikes.to(self.device)

            # Update traces (exponential decay + spike)
            self.trace = self.trace * self.decay_trace + input_spikes
            self.post_trace = self.post_trace * self.decay_trace

            # Adaptive learning rate (update less frequently - every 200 steps)
            self.update_count += 1
            if self.update_count % 200 == 0:
                decay_factor = self.learning_rate_decay ** (self.update_count / 200)
                self.current_a_plus = max(self.a_plus * decay_factor, self.a_plus * 0.1)
                self.current_a_minus = max(self.a_minus * decay_factor, self.a_minus * 0.1)

            # Homeostatic plasticity (update less frequently - every 500 steps)
            if output_spike:
                self.firing_rate_history.append(1.0)
            else:
                self.firing_rate_history.append(0.0)
            
            # Keep only recent history (last 500 updates - reduced from 1000)
            if len(self.firing_rate_history) > 500:
                self.firing_rate_history.pop(0)
            
            # Update homeostatic scaling less frequently (every 500 steps)
            if len(self.firing_rate_history) >= 100 and self.update_count % 500 == 0:
                recent_firing_rate = np.mean(self.firing_rate_history[-100:])
                if recent_firing_rate > self.target_firing_rate * 1.5:
                    self.homeostatic_scale *= 0.98  # Faster adjustment
                elif recent_firing_rate < self.target_firing_rate * 0.5:
                    self.homeostatic_scale *= 1.02  # Faster adjustment
                self.homeostatic_scale = np.clip(self.homeostatic_scale, 0.5, 2.0)

            # === OPTIMIZED STDP WEIGHT UPDATES ===

            # Combined weight update (reduce separate operations)
            weight_delta = torch.zeros_like(self.weights)

            # LTD: Depression when input spikes AFTER output spike
            if torch.any(input_spikes > 0):
                # Simplified importance factor
                importance_factor = 1.0 / (1.0 + self.importance_weights * 0.5)  # Reduced impact
                dw_minus = -self.current_a_minus * self.post_trace * input_spikes * self.homeostatic_scale * importance_factor
                weight_delta += dw_minus

            # LTP: Potentiation when output spikes AFTER input spike
            if output_spike:
                importance_factor = 1.0 / (1.0 + self.importance_weights * 0.5)
                dw_plus = self.current_a_plus * self.trace * self.homeostatic_scale * importance_factor
                weight_delta += dw_plus

                # Update importance weights (simplified)
                self.importance_weights += 0.005 * self.trace  # Reduced from 0.01

            # Apply weight decay and update in one operation
            self.weights = (self.weights + weight_delta) * (1.0 - self.weight_decay)

            # Clip weights
            self.weights = torch.clamp(self.weights, self.weight_min, self.weight_max)
            
            # Update consolidated weights less frequently (every 100 steps)
            if self.update_count % 100 == 0 and self.consolidated_weights is not None:
                consolidation_rate = 0.001  # Faster consolidation
                self.consolidated_weights = (1.0 - consolidation_rate) * self.consolidated_weights + consolidation_rate * self.weights

    def reset(self) -> None:
        """
        Reset neuron state to resting values (keeps weights).

        Useful for starting new trials or episodes.
        """
        # Use in-place operations to avoid creating new tensors
        self.v.fill_(self.v_rest)
        self.u.fill_(0.0)
        self.theta.fill_(0.0)
        self.trace.fill_(0.0)
        self.post_trace.fill_(0.0)
    
    def reset_state(self) -> None:
        """
        Alias for reset() for compatibility with circuit interface.
        """
        self.reset()

    def get_state(self) -> dict:
        """
        Get current state as CPU numpy arrays (for visualization/logging).

        This is the ONLY method that moves data from GPU to CPU.

        Returns:
            dict: State dictionary with numpy arrays
        """
        return {
            'v': self.v.cpu().numpy(),
            'u': self.u.cpu().numpy(),
            'theta': self.theta.cpu().numpy(),
            'weights': self.weights.cpu().numpy(),
            'trace': self.trace.cpu().numpy(),
            'post_trace': self.post_trace.cpu().numpy(),
            'device': str(self.device)
        }

    def get_weights(self) -> np.ndarray:
        """
        Get synaptic weights as numpy array (moves to CPU).

        Returns:
            numpy array of weights
        """
        return self.weights.cpu().numpy()

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set synaptic weights from numpy array (moves to GPU).

        Args:
            weights: Numpy array of weights
        """
        self.weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device
        )
        # Ensure valid range
        self.weights = torch.clamp(self.weights, self.weight_min, self.weight_max)

    def to(self, device: str) -> 'BiologicalNeuron':
        """
        Move neuron to different device (cuda/cpu).

        Args:
            device: Target device ('cuda' or 'cpu')

        Returns:
            self (for chaining)
        """
        self.device = torch.device(device)

        # Move all tensors to new device
        self.v = self.v.to(self.device)
        self.u = self.u.to(self.device)
        self.theta = self.theta.to(self.device)
        self.weights = self.weights.to(self.device)
        self.trace = self.trace.to(self.device)
        self.post_trace = self.post_trace.to(self.device)

        # Move decay factors
        self.decay_v = self.decay_v.to(self.device)
        self.decay_u = self.decay_u.to(self.device)
        self.decay_theta = self.decay_theta.to(self.device)
        self.decay_trace = self.decay_trace.to(self.device)
        
        # Move learning-related tensors if they exist
        if hasattr(self, 'importance_weights') and isinstance(self.importance_weights, torch.Tensor):
            self.importance_weights = self.importance_weights.to(self.device)
        if hasattr(self, 'consolidated_weights') and isinstance(self.consolidated_weights, torch.Tensor):
            self.consolidated_weights = self.consolidated_weights.to(self.device)

        return self


# ============================================================================
# Helper Functions
# ============================================================================

def check_gpu_available() -> Tuple[bool, str]:
    """
    Check if GPU is available and return info.

    Returns:
        Tuple of (is_available, device_name)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    else:
        return False, "CPU only"


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory usage information.

    Returns:
        dict with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {'available': False}

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        'available': True,
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - allocated
    }


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_gpu_neuron():
    """
    Demonstration of GPU-accelerated neuron.
    """
    print("\n" + "="*70)
    print("GPU-ACCELERATED BIOLOGICAL NEURON DEMO")
    print("="*70)

    # Check GPU availability
    gpu_available, device_name = check_gpu_available()
    print(f"\nGPU Available: {gpu_available}")
    print(f"Device: {device_name}")

    if gpu_available:
        mem_info = get_gpu_memory_info()
        print(f"GPU Memory: {mem_info['free_gb']:.2f} GB free / {mem_info['total_gb']:.2f} GB total")

    print("\n" + "="*70)
    print("Creating neuron...")

    # Create neuron (automatically uses GPU if available)
    neuron = BiologicalNeuron(n_inputs=64)

    print(f"Neuron device: {neuron.device}")
    print(f"Number of inputs: {neuron.n_inputs}")

    # Test basic operation
    print("\n" + "="*70)
    print("Testing neuron dynamics...")

    # Create input spikes on GPU
    input_spikes = torch.zeros(64, dtype=torch.float32, device=neuron.device)
    input_spikes[0] = 1.0  # Spike on first input
    input_spikes[10] = 1.0  # Spike on tenth input

    spike_count = 0
    for step in range(100):
        # Apply STDP learning
        output_spike = neuron.update(I_ext=50.0)
        neuron.stdp(input_spikes, output_spike)

        if output_spike:
            spike_count += 1

    print(f"Total spikes: {spike_count}")

    # Get state (moves to CPU)
    state = neuron.get_state()
    print(f"\nFinal state:")
    print(f"  Voltage: {state['v']:.2f} mV")
    print(f"  Adaptation: {state['u']:.2f}")
    print(f"  Threshold: {state['theta']:.2f}")
    print(f"  Mean weight: {state['weights'].mean():.3f}")
    print(f"  Max weight: {state['weights'].max():.3f}")

    # Show memory usage
    if gpu_available:
        mem_info = get_gpu_memory_info()
        print(f"\nGPU Memory after simulation:")
        print(f"  Allocated: {mem_info['allocated_gb']:.4f} GB")
        print(f"  Reserved: {mem_info['reserved_gb']:.4f} GB")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    # Run demo
    demo_gpu_neuron()

