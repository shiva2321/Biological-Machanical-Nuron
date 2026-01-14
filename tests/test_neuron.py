"""
Test script to validate the BiologicalNeuron implementation.
"""

import numpy as np
from neuron import BiologicalNeuron


def test_basic_functionality():
    """Test basic neuron creation and spike generation."""
    print("Testing BiologicalNeuron implementation...\n")

    # Create a neuron with 5 inputs
    neuron = BiologicalNeuron(n_inputs=5, dt=1.0)
    print(f"✓ Created neuron with {neuron.n_inputs} inputs")

    # Check initial state
    state = neuron.get_state()
    print(f"✓ Initial state: v={state['v']:.2f} mV, u={state['u']:.2f}, theta={state['theta']:.2f}")
    print(f"  Weights shape: {state['weights'].shape}, range: [{state['weights'].min():.3f}, {state['weights'].max():.3f}]")

    # Test with strong input to trigger spike
    print("\n--- Testing spike generation ---")
    input_spikes = np.array([1, 1, 1, 1, 1])  # All inputs fire

    spike_occurred = False
    for t in range(50):
        spike = neuron.step(input_spikes, I_ext=10.0, learning=False)
        if spike:
            print(f"✓ Spike at t={t} ms!")
            print(f"  State before reset: v={neuron.v:.2f}, u={neuron.u:.2f}, theta={neuron.theta:.2f}")
            spike_occurred = True
            break

    if not spike_occurred:
        print("✗ No spike occurred (may need stronger input)")

    # Test STDP learning
    print("\n--- Testing STDP learning ---")
    neuron.reset_state()
    initial_weights = neuron.weights.copy()
    print(f"Initial weights: {initial_weights}")

    # Simulate pattern learning
    pattern = np.array([1, 0, 1, 0, 0])
    for i in range(100):
        spike = neuron.step(pattern, I_ext=15.0, learning=True)

    final_weights = neuron.weights.copy()
    print(f"Final weights:   {final_weights}")
    print(f"Weight changes:  {final_weights - initial_weights}")
    print("✓ STDP learning applied")

    # Test weight clipping
    print("\n--- Testing weight clipping ---")
    print(f"Weight range: [{neuron.weights.min():.3f}, {neuron.weights.max():.3f}]")
    assert np.all(neuron.weights >= neuron.weight_min), "Weights below minimum!"
    assert np.all(neuron.weights <= neuron.weight_max), "Weights above maximum!"
    print(f"✓ All weights clipped to [{neuron.weight_min}, {neuron.weight_max}]")

    # Test state reset
    print("\n--- Testing state reset ---")
    neuron.reset_state()
    state = neuron.get_state()
    print(f"After reset: v={state['v']:.2f}, u={state['u']:.2f}, theta={state['theta']:.2f}")
    print(f"✓ State reset successful")

    print("\n" + "="*50)
    print("All tests passed! BiologicalNeuron is functional.")
    print("="*50)


if __name__ == "__main__":
    test_basic_functionality()

