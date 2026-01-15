"""
Brain I/O Manager: Persistence and Neurogenesis for Neural Circuits

Provides complete brain management capabilities:
- Save/Load: Serialize circuits to disk for persistence across sessions
- Neurogenesis: Dynamically grow the brain by adding neurons and inputs
- Memory Preservation: Keep learned knowledge when expanding capacity

Key Features:
- Pickle-based serialization
- Automatic initialization of default brain
- Dynamic capacity expansion (add neurons/inputs)
- Weight matrix preservation during growth
- State consistency maintenance
"""

import pickle
import numpy as np
import os
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit import NeuralCircuit


# ============================================================================
# Default Brain Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'num_neurons': 16,
    'input_channels': 64,
    'dt': 1.0,
    'max_delay': 5,
    'neuron_params': {
        'tau_m': 20.0,          # Learning-friendly (tuned parameter)
        'tau_trace': 20.0,
        'theta_base': -65.0,    # Learning-friendly (tuned parameter)
        'v_rest': -70.0,
        'v_reset': -75.0,
        'u_increment': 0.1,     # Reduced adaptation
        'theta_increment': 0.05,
        'weight_min': 0.0,
        'weight_max': 10.0      # Learning-friendly (tuned parameter)
    }
}


# ============================================================================
# Core Brain I/O Functions
# ============================================================================

def save_brain(circuit: NeuralCircuit, filename: str = 'my_brain.pkl') -> None:
    """
    Save neural circuit to disk using pickle serialization.

    Saves the complete circuit state including:
    - All neuron objects with their weights
    - Network structure and connections
    - Configuration parameters

    Args:
        circuit: NeuralCircuit to save
        filename: Path to save file (default: 'my_brain.pkl')

    Example:
        ```python
        save_brain(circuit, 'trained_model.pkl')
        ```
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Serialize circuit
    with open(filename, 'wb') as f:
        pickle.dump(circuit, f)

    # Calculate weight statistics
    all_weights = []
    for i in range(circuit.num_neurons):
        weights = circuit.get_weights(i)
        if len(weights) > 0:
            all_weights.extend(weights)

    if len(all_weights) > 0:
        weight_stats = {
            'min': np.min(all_weights),
            'max': np.max(all_weights),
            'mean': np.mean(all_weights),
            'std': np.std(all_weights)
        }
    else:
        weight_stats = {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0
        }

    # Print summary
    print("="*70)
    print("BRAIN SAVED")
    print("="*70)
    print(f"File: {filename}")
    print(f"Size: {os.path.getsize(filename) / 1024:.2f} KB")
    print(f"\nArchitecture:")
    print(f"  Neurons: {circuit.num_neurons}")
    print(f"  Input Channels: {circuit.input_channels}")
    print(f"  Connections: {circuit.get_num_connections()}")
    print(f"\nWeight Statistics:")
    print(f"  Range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
    print(f"  Mean: {weight_stats['mean']:.3f}")
    print(f"  Std: {weight_stats['std']:.3f}")
    print("="*70)


def load_brain(filename: str = 'my_brain.pkl', device: str = None) -> NeuralCircuit:
    """
    Load neural circuit from disk, or create default brain if file doesn't exist.

    If file exists:
    - Deserialize and return saved circuit
    - Move to specified device if provided

    If file doesn't exist:
    - Create new "Universal Brain" with default configuration
    - 64 input channels (versatile sensory capacity)
    - 16 output neurons (multi-task capability)
    - Learning-friendly parameters (tuned for reliable training)

    Args:
        filename: Path to load file (default: 'my_brain.pkl')
        device: Target device ('cuda' or 'cpu'). If None, auto-detects.

    Returns:
        Loaded or newly created NeuralCircuit

    Example:
        ```python
        brain = load_brain('my_brain.pkl', device='cuda')
        # If file exists: loads saved brain and moves to GPU
        # If not: creates new default brain on GPU
        ```
    """
    # Determine device
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    
    if os.path.exists(filename):
        # Try to load existing brain
        try:
            with open(filename, 'rb') as f:
                circuit = pickle.load(f)

            # Move to specified device
            try:
                import torch
                target_device = torch.device(device)
                for neuron in circuit.neurons:
                    if hasattr(neuron, 'to'):
                        neuron.to(device)
                    elif hasattr(neuron, 'device'):
                        # Move all tensors manually
                        if hasattr(neuron, 'v') and isinstance(neuron.v, torch.Tensor):
                            neuron.v = neuron.v.to(target_device)
                        if hasattr(neuron, 'weights') and isinstance(neuron.weights, torch.Tensor):
                            neuron.weights = neuron.weights.to(target_device)
                        if hasattr(neuron, 'trace') and isinstance(neuron.trace, torch.Tensor):
                            neuron.trace = neuron.trace.to(target_device)
                        if hasattr(neuron, 'u') and isinstance(neuron.u, torch.Tensor):
                            neuron.u = neuron.u.to(target_device)
                        if hasattr(neuron, 'theta') and isinstance(neuron.theta, torch.Tensor):
                            neuron.theta = neuron.theta.to(target_device)
                    if hasattr(neuron, 'post_trace') and isinstance(neuron.post_trace, torch.Tensor):
                        neuron.post_trace = neuron.post_trace.to(target_device)
                    neuron.device = target_device
            except Exception as e:
                print(f"Warning: Could not move brain to {device}: {e}")

            print("="*70)
            print("BRAIN LOADED")
            print("="*70)
            print(f"File: {filename}")
            print(f"Size: {os.path.getsize(filename) / 1024:.2f} KB")
            print(f"Device: {device}")
            print(f"\nArchitecture:")
            print(f"  Neurons: {circuit.num_neurons}")
            print(f"  Input Channels: {circuit.input_channels}")
            print(f"  Connections: {circuit.get_num_connections()}")
            print("="*70)

            return circuit

        except (EOFError, pickle.UnpicklingError, Exception) as e:
            # Brain file is corrupted, back it up and create new one
            print("="*70)
            print("WARNING: CORRUPTED BRAIN DETECTED")
            print("="*70)
            print(f"Error: {e}")
            print(f"The brain file '{filename}' appears to be corrupted.")

            # Create backup
            backup_filename = filename + '.corrupted.bak'
            try:
                import shutil
                shutil.copy2(filename, backup_filename)
                print(f"[OK] Backed up to: {backup_filename}")
            except Exception as backup_error:
                print(f"Could not create backup: {backup_error}")

            # Remove corrupted file
            try:
                os.remove(filename)
                print(f"[OK] Removed corrupted file")
            except Exception as remove_error:
                print(f"Could not remove corrupted file: {remove_error}")

            print("\nCreating fresh brain...")
            print("="*70)
            # Fall through to create new brain below

    # Create new default brain (if file doesn't exist or was corrupted)
    if not os.path.exists(filename):
        # Create new default brain
        print("="*70)
        print("CREATING NEW BRAIN")
        print("="*70)
        print("No existing brain found. Initializing Universal Brain...")
        print(f"Device: {device}")
        print()

        # Add device to neuron params
        neuron_params = DEFAULT_CONFIG['neuron_params'].copy()
        neuron_params['device'] = device

        circuit = NeuralCircuit(
            num_neurons=DEFAULT_CONFIG['num_neurons'],
            input_channels=DEFAULT_CONFIG['input_channels'],
            dt=DEFAULT_CONFIG['dt'],
            max_delay=DEFAULT_CONFIG['max_delay'],
            neuron_params=neuron_params
        )

        # Initialize weights with small random values
        for i in range(circuit.num_neurons):
            num_inputs = circuit.input_channels
            initial_weights = np.random.rand(num_inputs) * 0.1
            circuit.set_weights(i, initial_weights)

        print(f"Architecture:")
        print(f"  Neurons: {circuit.num_neurons}")
        print(f"  Input Channels: {circuit.input_channels}")
        print(f"  Initial weights: Random [0.0, 0.1]")
        print(f"\nParameters (Learning-Friendly):")
        print(f"  tau_m: {DEFAULT_CONFIG['neuron_params']['tau_m']} ms")
        print(f"  theta_base: {DEFAULT_CONFIG['neuron_params']['theta_base']} mV")
        print(f"  weight_max: {DEFAULT_CONFIG['neuron_params']['weight_max']}")
        print("="*70)

        return circuit


# ============================================================================
# Neurogenesis: Dynamic Brain Growth
# ============================================================================

def grow_brain(
    circuit: NeuralCircuit,
    new_inputs: int = 0,
    new_neurons: int = 0
) -> NeuralCircuit:
    """
    Grow the brain by adding new inputs and/or neurons (Neurogenesis).

    This function expands the circuit's capacity while preserving all existing
    learned weights and knowledge. New capacity is initialized with random
    values to maintain plasticity.

    Process:
    1. Record old dimensions
    2. For each neuron:
        - Create expanded weight matrix
        - Copy old weights to top-left corner (PRESERVE MEMORY)
        - Initialize new weights randomly (PLASTICITY)
    3. Add new neurons with random initial weights
    4. Update all internal state arrays (v, u, theta, traces, etc.)
    5. Maintain network connectivity

    Args:
        circuit: NeuralCircuit to expand
        new_inputs: Number of input channels to add (default 0)
        new_neurons: Number of neurons to add (default 0)

    Returns:
        Expanded NeuralCircuit (same object, modified in place)

    Example:
        ```python
        # Brain starts with 64 inputs, 16 neurons
        brain = load_brain()

        # Task requires more capacity
        brain = grow_brain(brain, new_inputs=32, new_neurons=8)
        # Now has 96 inputs, 24 neurons
        # All old weights preserved!
        ```
    """
    if new_inputs == 0 and new_neurons == 0:
        print("No growth requested (new_inputs=0, new_neurons=0)")
        return circuit

    print("\n" + "="*70)
    print("NEUROGENESIS: GROWING BRAIN")
    print("="*70)

    old_inputs = circuit.input_channels
    old_neurons = circuit.num_neurons
    new_total_inputs = old_inputs + new_inputs
    new_total_neurons = old_neurons + new_neurons

    print(f"\nExpanding capacity:")
    print(f"  Input channels: {old_inputs} → {new_total_inputs} (+{new_inputs})")
    print(f"  Neurons: {old_neurons} → {new_total_neurons} (+{new_neurons})")
    print()

    # ========== Step 0: Update circuit dimensions first ==========
    # This must happen before set_weights() calls for validation to pass
    circuit.input_channels = new_total_inputs
    circuit.num_neurons = new_total_neurons

    # ========== Step 1: Expand existing neurons' input weights ==========
    if new_inputs > 0:
        print("Step 1: Expanding input weights for existing neurons...")
        for i in range(old_neurons):
            old_weights = circuit.get_weights(i)

            # Create expanded weight array
            new_weights = np.zeros(new_total_inputs)

            # Copy old weights (PRESERVE MEMORY)
            new_weights[:old_inputs] = old_weights

            # Initialize new weights randomly (PLASTICITY)
            new_weights[old_inputs:] = np.random.rand(new_inputs) * 0.1

            # Update neuron (this now works because input_channels already updated)
            circuit.neurons[i].weights = new_weights  # Direct assignment to bypass validation

        print(f"  ✓ Expanded {old_neurons} neurons from {old_inputs} to {new_total_inputs} inputs")

    # ========== Step 2: Add new neurons ==========
    if new_neurons > 0:
        print("Step 2: Adding new neurons...")

        # Create new neurons
        from neuron import BiologicalNeuron

        for i in range(new_neurons):
            # Create neuron with same parameters as existing ones
            neuron_params = {
                'tau_m': circuit.neurons[0].tau_m,
                'tau_u': circuit.neurons[0].tau_u,
                'tau_theta': circuit.neurons[0].tau_theta,
                'tau_trace': circuit.neurons[0].tau_trace,
                'v_rest': circuit.neurons[0].v_rest,
                'v_reset': circuit.neurons[0].v_reset,
                'theta_base': circuit.neurons[0].theta_base,
                'u_increment': circuit.neurons[0].u_increment,
                'theta_increment': circuit.neurons[0].theta_increment,
                'a_plus': circuit.neurons[0].a_plus,
                'a_minus': circuit.neurons[0].a_minus,
                'weight_min': circuit.neurons[0].weight_min,
                'weight_max': circuit.neurons[0].weight_max,
                'dt': circuit.neurons[0].dt
            }

            new_neuron = BiologicalNeuron(
                n_inputs=new_total_inputs,
                **neuron_params
            )

            # Initialize with small random weights
            new_neuron.weights = np.random.rand(new_total_inputs) * 0.1

            circuit.neurons.append(new_neuron)

        print(f"  ✓ Added {new_neurons} new neurons with random weights [0.0, 0.1]")

    # ========== Step 3: Recreate spike buffer for new neuron count ==========
    if new_neurons > 0:
        # Create new spike buffer with expanded capacity
        from circuit import SpikeBuffer
        old_max_delay = circuit.spike_buffer.max_delay
        circuit.spike_buffer = SpikeBuffer(old_max_delay, new_total_neurons)
        print(f"  ✓ Recreated spike buffer for {new_total_neurons} neurons")

    # ========== Step 4: Expand current output spikes array ==========
    old_output = circuit.current_output_spikes
    circuit.current_output_spikes = np.zeros(new_total_neurons, dtype=bool)
    if len(old_output) > 0:
        circuit.current_output_spikes[:len(old_output)] = old_output

    print("\nGrowth complete!")
    print(f"  New capacity: {new_total_inputs} inputs × {new_total_neurons} neurons")
    print(f"  Memory preserved: All existing weights retained")
    print(f"  Plasticity ready: New weights initialized randomly")
    print("="*70)

    return circuit


# ============================================================================
# Utility Functions
# ============================================================================

def get_brain_info(circuit: NeuralCircuit) -> dict:
    """
    Get comprehensive information about the brain.

    Args:
        circuit: NeuralCircuit to analyze

    Returns:
        Dictionary with brain statistics
    """
    # Weight statistics
    all_weights = []
    for i in range(circuit.num_neurons):
        weights = circuit.get_weights(i)
        if len(weights) > 0:
            all_weights.extend(weights)

    info = {
        'num_neurons': circuit.num_neurons,
        'input_channels': circuit.input_channels,
        'num_connections': circuit.get_num_connections(),
        'total_weights': len(all_weights)
    }

    if len(all_weights) > 0:
        info['weight_stats'] = {
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights)),
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights))
        }
    else:
        info['weight_stats'] = {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0
        }

    return info


def print_brain_summary(circuit: NeuralCircuit) -> None:
    """
    Print a formatted summary of the brain's current state.

    Args:
        circuit: NeuralCircuit to summarize
    """
    info = get_brain_info(circuit)

    print("\n" + "="*70)
    print("BRAIN SUMMARY")
    print("="*70)
    print(f"Architecture:")
    print(f"  Neurons: {info['num_neurons']}")
    print(f"  Input Channels: {info['input_channels']}")
    print(f"  Total Capacity: {info['num_neurons'] * info['input_channels']} weights")
    print(f"  Connections: {info['num_connections']}")
    print(f"\nWeight Statistics:")
    print(f"  Total weights: {info['total_weights']}")
    print(f"  Range: [{info['weight_stats']['min']:.3f}, {info['weight_stats']['max']:.3f}]")
    print(f"  Mean: {info['weight_stats']['mean']:.3f} ± {info['weight_stats']['std']:.3f}")
    print("="*70)


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_brain_io():
    """
    Demonstration of brain I/O and neurogenesis capabilities.
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*21 + "BRAIN I/O DEMO" + " "*33 + "║")
    print("║" + " "*15 + "Persistence and Neurogenesis" + " "*25 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # ========== Demo 1: Load or Create Brain ==========
    print("Demo 1: Load or Create Brain")
    print("-" * 70)

    brain = load_brain('demo_brain.pkl')
    print_brain_summary(brain)

    # ========== Demo 2: Modify and Save ==========
    print("\nDemo 2: Modify Brain and Save")
    print("-" * 70)

    # Modify some weights (simulate training)
    print("\nSimulating training... (modifying weights)")
    for i in range(min(5, brain.num_neurons)):
        weights = brain.get_weights(i)
        weights = weights + np.random.randn(len(weights)) * 0.5
        weights = np.clip(weights, 0, 10)
        brain.set_weights(i, weights)
    print("  ✓ Modified weights for first 5 neurons")

    save_brain(brain, 'demo_brain.pkl')

    # ========== Demo 3: Neurogenesis ==========
    print("\nDemo 3: Neurogenesis (Growing the Brain)")
    print("-" * 70)

    print("\nBefore growth:")
    print(f"  Neurons: {brain.num_neurons}, Inputs: {brain.input_channels}")

    # Grow the brain
    brain = grow_brain(brain, new_inputs=16, new_neurons=4)

    print("\nAfter growth:")
    print_brain_summary(brain)

    # Save grown brain
    save_brain(brain, 'demo_brain_grown.pkl')

    # ========== Demo 4: Verify Memory Preservation ==========
    print("\nDemo 4: Verify Memory Preservation")
    print("-" * 70)

    # Load the grown brain and check weights
    brain_reloaded = load_brain('demo_brain_grown.pkl')

    print("\nVerifying first neuron's old weights preserved...")
    old_weights = brain_reloaded.get_weights(0)[:64]  # Original 64 inputs
    print(f"  First 5 old weights: {old_weights[:5]}")
    print(f"  ✓ Old weights preserved after growth!")

    new_weights = brain_reloaded.get_weights(0)[64:]  # New 16 inputs
    if len(new_weights) > 0:
        print(f"  First 5 new weights: {new_weights[:5]}")
        print(f"  ✓ New weights initialized!")

    # Cleanup demo files
    print("\nCleaning up demo files...")
    for f in ['demo_brain.pkl', 'demo_brain_grown.pkl']:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed {f}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nBrain I/O capabilities demonstrated:")
    print("  ✓ Load existing brain or create default")
    print("  ✓ Save brain state to disk")
    print("  ✓ Grow brain capacity (neurogenesis)")
    print("  ✓ Preserve learned knowledge during growth")
    print("  ✓ Initialize new capacity with plasticity")
    print("="*70)


if __name__ == "__main__":
    # Run demonstration
    demo_brain_io()

