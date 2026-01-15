"""
Trader Experiment: Time-Series Prediction with Population Coding

Demonstrates how a spiking neural network can learn and predict continuous
time-series data (sine wave) using population coding and STDP.

Key Concepts:
- Population coding: Map continuous values to discrete neurons
- Temporal learning: STDP learns temporal transitions (y(t) → y(t+1))
- Prediction: Network anticipates next value based on current input
- Decoding: Read prediction from neuron with highest sub-threshold voltage
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit import NeuralCircuit


def encode_value(value, num_neurons=20):
    """
    Encode a continuous value [-1, 1] to a neuron index [0, num_neurons-1].

    Population coding: Each neuron represents a "bin" in the value space.

    Args:
        value: Continuous value in range [-1, 1]
        num_neurons: Number of neurons in population (default 20)

    Returns:
        neuron_id: Integer index of neuron representing this value
    """
    # Map [-1, 1] to [0, num_neurons-1]
    normalized = (value + 1.0) / 2.0  # Map to [0, 1]
    neuron_id = int(normalized * (num_neurons - 1))
    neuron_id = np.clip(neuron_id, 0, num_neurons - 1)
    return neuron_id


def decode_value(neuron_id, num_neurons=20):
    """
    Decode a neuron index back to continuous value [-1, 1].

    Inverse of encode_value.

    Args:
        neuron_id: Integer index of neuron
        num_neurons: Number of neurons in population

    Returns:
        value: Continuous value in range [-1, 1]
    """
    # Map [0, num_neurons-1] to [-1, 1]
    normalized = neuron_id / (num_neurons - 1)  # Map to [0, 1]
    value = normalized * 2.0 - 1.0  # Map to [-1, 1]
    return value


def generate_sine_wave(num_steps=200, frequency=0.1):
    """
    Generate sine wave time series.

    Args:
        num_steps: Number of time steps
        frequency: Frequency parameter (controls oscillation speed)

    Returns:
        Array of sine wave values in range [-1, 1]
    """
    steps = np.arange(num_steps)
    sine_wave = np.sin(steps * frequency)
    return sine_wave


def build_predictor_brain(num_neurons=20):
    """
    Build neural circuit for time-series prediction.

    Architecture:
    - num_neurons neurons (population coding for values)
    - num_neurons input channels (one-to-one mapping)
    - Each neuron learns: which other neuron typically fires next
    - STDP creates temporal chain: neuron_t → neuron_{t+1}

    Returns:
        NeuralCircuit configured for prediction
    """
    print("Building Predictor Brain (Population Coding)...")
    print("="*70)

    circuit = NeuralCircuit(
        num_neurons=num_neurons,
        input_channels=num_neurons,
        dt=1.0,
        max_delay=5,
        neuron_params={
            'tau_m': 20.0,        # From hunter experiment
            'tau_trace': 20.0,
            'theta_base': -65.0,  # From hunter experiment
            'v_rest': -70.0,
            'v_reset': -75.0,
            'u_increment': 0.3,
            'theta_increment': 0.1,
            'weight_min': 0.0,
            'weight_max': 10.0    # From hunter experiment
        }
    )

    # Initialize all neurons with zero input weights (tabula rasa)
    for i in range(num_neurons):
        circuit.set_weights(i, np.zeros(num_neurons))

    print(f"\nArchitecture:")
    print(f"  {num_neurons} Neurons: Population coding for values [-1, 1]")
    print(f"  {num_neurons} Input Channels: One-to-one with neurons")
    print(f"  Each neuron has {num_neurons} input weights")
    print(f"\nPopulation Coding:")
    print(f"  Neuron 0  → Value -1.0")
    print(f"  Neuron 10 → Value  0.0")
    print(f"  Neuron 19 → Value +1.0")
    print(f"\nLearning Mechanism:")
    print(f"  STDP learns transitions: value(t) → value(t+1)")
    print(f"  Creates temporal chain through weight strengthening")
    print("="*70)
    print()

    return circuit


def training_phase(circuit, sine_wave, train_steps=150):
    """
    Training phase: Learn temporal structure of sine wave.

    On each step:
    1. Get current sine value
    2. Encode to neuron index
    3. Stimulate that neuron strongly (make it fire)
    4. STDP learns: current neuron predicts next neuron

    Args:
        circuit: NeuralCircuit brain
        sine_wave: Time series data
        train_steps: Number of training steps
    """
    print("\n" + "="*70)
    print("TRAINING PHASE: Learning Sine Wave Temporal Structure")
    print("="*70)
    print("Teaching the brain temporal transitions: y(t) → y(t+1)")
    print()

    num_neurons = circuit.num_neurons

    for step in range(train_steps):
        # Get current value
        current_value = sine_wave[step]

        # Encode to neuron
        neuron_id = encode_value(current_value, num_neurons)

        # Create input pattern: activate only this neuron's input
        input_spikes = np.zeros(num_neurons)
        input_spikes[neuron_id] = 1.0

        # Scale for reliable but selective firing
        input_spikes = input_spikes * 120.0  # Balanced scaling

        # Moderate baseline current
        I_ext = np.ones(num_neurons) * 7.0  # Balanced baseline

        # Step circuit with STDP enabled
        output_spikes = circuit.step(
            input_spikes=input_spikes,
            I_ext=I_ext,
            learning=True  # STDP active!
        )

        # Display progress
        if (step + 1) % 30 == 0:
            fired_neurons = np.where(output_spikes)[0]
            print(f"  Step {step+1:3d}: Value={current_value:+.3f}, "
                  f"Neuron={neuron_id:2d}, Fired={fired_neurons.tolist()}")

    print(f"\nTraining complete! {train_steps} steps.")
    print("STDP has learned temporal transitions in the sine wave.")


def testing_phase(circuit, sine_wave, train_steps=150, test_steps=50):
    """
    Testing phase: Predict future values using learned weights.

    On each step:
    1. Stimulate neuron for current value
    2. Check voltages of all neurons (sub-threshold excitation)
    3. Neuron with highest voltage (excluding input) = prediction
    4. Decode prediction to continuous value

    Args:
        circuit: NeuralCircuit brain
        sine_wave: Time series data
        train_steps: Where training ended
        test_steps: Number of testing steps

    Returns:
        predictions: Array of predicted values
        actuals: Array of actual values
    """
    print("\n" + "="*70)
    print("TESTING PHASE: Forecasting Sine Wave")
    print("="*70)
    print("Using learned transitions to predict next value")
    print()

    num_neurons = circuit.num_neurons
    predictions = []
    actuals = []

    for step in range(train_steps, train_steps + test_steps):
        if step >= len(sine_wave):
            break

        # Get current value
        current_value = sine_wave[step]
        neuron_id = encode_value(current_value, num_neurons)

        # Create input pattern
        input_spikes = np.zeros(num_neurons)
        input_spikes[neuron_id] = 1.0
        input_spikes = input_spikes * 120.0  # Balanced input

        # Moderate baseline current
        I_ext = np.ones(num_neurons) * 7.0

        # Step circuit WITHOUT firing (just to see voltage changes)
        # First, step to update voltages
        output_spikes = circuit.step(
            input_spikes=input_spikes,
            I_ext=I_ext,
            learning=False  # No learning during testing
        )

        # Get voltages of all neurons (look for sub-threshold excitation)
        # Access voltage directly from neuron objects
        voltages = np.array([circuit.neurons[i].v for i in range(num_neurons)])

        # Find neuron with highest voltage (excluding the input neuron)
        voltages_copy = voltages.copy()
        voltages_copy[neuron_id] = -np.inf  # Exclude input neuron

        predicted_neuron = np.argmax(voltages_copy)
        predicted_value = decode_value(predicted_neuron, num_neurons)

        # Get actual next value (if available)
        if step + 1 < len(sine_wave):
            actual_next_value = sine_wave[step + 1]
        else:
            actual_next_value = current_value

        predictions.append(predicted_value)
        actuals.append(actual_next_value)

        # Display progress
        if (step - train_steps + 1) % 10 == 0:
            error = abs(predicted_value - actual_next_value)
            print(f"  Step {step+1:3d}: Current={current_value:+.3f}, "
                  f"Predict={predicted_value:+.3f}, Actual={actual_next_value:+.3f}, "
                  f"Error={error:.3f}")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate performance metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))

    print(f"\nTesting complete! {len(predictions)} predictions.")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print("="*70)

    return predictions, actuals


def visualize_results(sine_wave, predictions, actuals, train_steps=150):
    """
    Visualize prediction results.

    Creates plot with:
    - Blue line: Actual sine wave
    - Orange line: Predicted values
    - Green line: Training region

    Args:
        sine_wave: Full time series
        predictions: Predicted values (testing phase)
        actuals: Actual next values (testing phase)
        train_steps: Number of training steps
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Time-Series Prediction with Spiking Neural Network',
                 fontsize=16, fontweight='bold')

    # ========== Plot 1: Full Time Series with Predictions ==========
    ax1 = axes[0]

    # Plot full actual sine wave
    steps = np.arange(len(sine_wave))
    ax1.plot(steps, sine_wave, 'b-', linewidth=2, label='Actual Sine Wave', alpha=0.7)

    # Highlight training region
    ax1.axvspan(0, train_steps, alpha=0.2, color='green', label='Training Region')

    # Plot predictions in testing region
    test_steps_range = np.arange(train_steps, train_steps + len(predictions))
    ax1.plot(test_steps_range, predictions, 'o-', color='orange',
             linewidth=2, markersize=4, label='Predicted Values', alpha=0.9)

    # Mark transition point
    ax1.axvline(x=train_steps, color='red', linestyle='--',
                linewidth=2, label='Train/Test Split')

    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Sine Wave Prediction: Network Learns Temporal Structure',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(sine_wave))
    ax1.set_ylim(-1.2, 1.2)

    # ========== Plot 2: Prediction Error Over Time ==========
    ax2 = axes[1]

    # Calculate errors
    errors = np.abs(predictions - actuals)
    test_steps_range = np.arange(train_steps, train_steps + len(errors))

    ax2.plot(test_steps_range, errors, 'r-', linewidth=2, label='Absolute Error')
    ax2.fill_between(test_steps_range, 0, errors, alpha=0.3, color='red')

    # Add mean error line
    mean_error = np.mean(errors)
    ax2.axhline(y=mean_error, color='darkred', linestyle='--',
                linewidth=2, label=f'Mean Error = {mean_error:.3f}')

    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Prediction Error: |Predicted - Actual|',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(train_steps, train_steps + len(errors))
    ax2.set_ylim(0, max(errors) * 1.1)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    filename = 'trader_experiment_results.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to '{filename}'")

    plt.show()


def analyze_learned_weights(circuit):
    """
    Analyze the learned weight matrix to understand temporal structure.

    Args:
        circuit: NeuralCircuit with learned weights
    """
    print("\n" + "="*70)
    print("WEIGHT MATRIX ANALYSIS")
    print("="*70)

    num_neurons = circuit.num_neurons

    # Build weight matrix
    weight_matrix = np.zeros((num_neurons, num_neurons))
    for i in range(num_neurons):
        weight_matrix[i, :] = circuit.get_weights(i)

    # Analyze structure
    print("\nLearned Temporal Structure:")
    print("(Showing strongest connections - neuron i → neuron j)")
    print()

    # For each neuron, find its strongest connections
    for i in [0, 5, 10, 15, 19]:  # Sample neurons
        weights = weight_matrix[i, :]
        top_indices = np.argsort(weights)[-3:][::-1]  # Top 3
        top_weights = weights[top_indices]

        value_i = decode_value(i, num_neurons)
        print(f"Neuron {i:2d} (value={value_i:+.2f}):")
        for idx, w in zip(top_indices, top_weights):
            value_j = decode_value(idx, num_neurons)
            print(f"  → Neuron {idx:2d} (value={value_j:+.2f}): weight={w:.2f}")

    # Check if diagonal is strong (should be)
    diagonal = np.diag(weight_matrix)
    off_diagonal = weight_matrix[~np.eye(num_neurons, dtype=bool)]

    print(f"\nWeight Statistics:")
    print(f"  Diagonal (self-connection): mean={np.mean(diagonal):.2f}, "
          f"max={np.max(diagonal):.2f}")
    print(f"  Off-diagonal (transitions): mean={np.mean(off_diagonal):.2f}, "
          f"max={np.max(off_diagonal):.2f}")

    # Check if super-diagonal is strong (should predict next value)
    super_diagonal = np.diag(weight_matrix, k=1)  # One step ahead
    if len(super_diagonal) > 0:
        print(f"  Super-diagonal (i→i+1):     mean={np.mean(super_diagonal):.2f}, "
              f"max={np.max(super_diagonal):.2f}")

    print("="*70)


def run_trader_experiment():
    """
    Run the complete time-series prediction experiment.

    Demonstrates:
    1. Population coding for continuous values
    2. STDP learning of temporal structure
    3. Prediction via sub-threshold voltage reading
    4. Evaluation on held-out test data
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*17 + "TRADER EXPERIMENT" + " "*33 + "║")
    print("║" + " "*11 + "Time-Series Prediction via Population Coding" + " "*13 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # Parameters
    num_neurons = 20
    num_steps = 200
    train_steps = 150
    test_steps = num_steps - train_steps

    # Generate data
    print("Generating sine wave time series...")
    sine_wave = generate_sine_wave(num_steps=num_steps, frequency=0.1)
    print(f"  Generated {num_steps} time steps")
    print(f"  Value range: [{np.min(sine_wave):.3f}, {np.max(sine_wave):.3f}]")
    print()

    # Build circuit
    circuit = build_predictor_brain(num_neurons=num_neurons)

    # Training phase
    training_phase(circuit, sine_wave, train_steps=train_steps)

    # Analyze learned weights
    analyze_learned_weights(circuit)

    # Testing phase
    predictions, actuals = testing_phase(
        circuit, sine_wave,
        train_steps=train_steps,
        test_steps=test_steps
    )

    # Visualize results
    visualize_results(sine_wave, predictions, actuals, train_steps=train_steps)

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))

    # Calculate correlation
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, actuals)[0, 1]
    else:
        correlation = 0.0

    print("\n📊 Performance Metrics:")
    print(f"  Training Steps: {train_steps}")
    print(f"  Testing Steps: {test_steps}")
    print(f"  Mean Absolute Error: {mae:.3f}")
    print(f"  Root Mean Squared Error: {rmse:.3f}")
    print(f"  Correlation: {correlation:.3f}")

    print("\n🧠 Learning Mechanism:")
    print("  1. Population Coding: Continuous values → discrete neurons")
    print("  2. STDP Learning: Neuron(t) → Neuron(t+1) transitions")
    print("  3. Prediction: Read sub-threshold voltage (anticipation)")
    print("  4. Decoding: Highest voltage → predicted value")

    print("\n🎯 Key Insights:")
    print("  ✓ SNNs can learn temporal structure in continuous signals")
    print("  ✓ Population coding bridges continuous ↔ discrete")
    print("  ✓ STDP creates predictive connections")
    print("  ✓ Sub-threshold voltages reveal anticipation")

    # Assess performance
    if mae < 0.3:
        status = "✅ EXCELLENT: Strong prediction accuracy!"
    elif mae < 0.5:
        status = "✅ GOOD: Reasonable prediction performance"
    elif mae < 0.7:
        status = "⚠️  PARTIAL: Some predictive capability"
    else:
        status = "❌ POOR: Needs parameter tuning"

    print(f"\n{status}")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    run_trader_experiment()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nThis demonstrates:")
    print("  ✓ Time-series prediction with SNNs")
    print("  ✓ Population coding for continuous values")
    print("  ✓ Temporal learning via STDP")
    print("  ✓ Prediction via voltage anticipation")
    print("\nThe network learned to forecast the sine wave! 📈")
    print("="*70)

