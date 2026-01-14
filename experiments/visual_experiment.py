"""
Visual Experiment: Hidden Pattern Discovery

Demonstrates how a BiologicalNeuron can learn to identify a repeating signal
hidden inside random Poisson noise through STDP learning.

Experimental Setup:
- 20 input channels with 2% Poisson noise
- Hidden pattern: inputs [0, 5, 10, 15] fire together every 100ms
- 2000ms simulation
- Visualization shows learning process
"""

import numpy as np
import matplotlib.pyplot as plt
from neuron import BiologicalNeuron


def run_hidden_pattern_experiment():
    """
    Run the hidden pattern discovery experiment and visualize results.
    """
    # ====================
    # Experiment Parameters
    # ====================
    n_inputs = 20
    duration_ms = 4000  # 4 seconds for clear learning trajectory
    dt = 1.0
    n_steps = int(duration_ms / dt)

    # Pattern configuration
    pattern_inputs = [0, 5, 10, 15]  # The "hidden pattern"
    pattern_interval = 100  # Pattern repeats every 100ms
    noise_prob = 0.02  # 2% Poisson noise per input per timestep

    # Signal strengths
    noise_amplitude = 1.0  # Noise spikes have normal amplitude
    pattern_amplitude = 4.0  # Pattern spikes are 4x stronger - very clear signal

    # ====================
    # Initialize Neuron
    # ====================
    neuron = BiologicalNeuron(
        n_inputs=n_inputs,
        tau_m=15.0,
        tau_trace=20.0,
        dt=dt,
        a_plus=0.06,  # Moderate potentiation rate
        a_minus=0.04,  # Slightly lower depression rate
        weight_min=0.0,
        weight_max=1.0,
        v_rest=-70.0,
        v_reset=-75.0,
        theta_base=-55.0,  # Lower threshold to make neuron more excitable
        u_increment=2.0,  # Reduced adaptation for more consistent firing
        theta_increment=0.5,  # Reduced threshold adaptation
    )

    # Set initial weights randomly between 0.2 and 0.4 (lower start for clearer learning)
    neuron.weights = np.random.uniform(0.2, 0.4, size=n_inputs)

    # Add baseline external current to help neuron reach threshold
    # This represents tonic background activity
    I_baseline = 18.0  # Moderate baseline - pattern should push it over threshold

    # Scale input spikes to have more effect (biological spikes have ~1mV PSP)
    input_scale = 6.0  # Increased for stronger pattern signal

    # ====================
    # Data Recording
    # ====================
    # History arrays
    time_history = []
    v_history = []
    theta_history = []
    weight_history = []
    output_spike_times = []

    # For raster plot: store (time, input_index) for each input spike
    input_spike_times = []
    input_spike_indices = []

    # ====================
    # Run Simulation
    # ====================
    print(f"Running {duration_ms}ms simulation...")
    print(f"Pattern inputs: {pattern_inputs}")
    print(f"Pattern repeats every {pattern_interval}ms")
    print(f"Background noise: {noise_prob*100}% per input per ms")
    print()

    for step in range(n_steps):
        t = step * dt

        # Generate input spikes
        input_spikes = np.zeros(n_inputs)

        # 1. Add Poisson noise (2% chance per input)
        noise_mask = np.random.rand(n_inputs) < noise_prob
        input_spikes[noise_mask] = noise_amplitude

        # 2. Inject hidden pattern every 100ms (OVERRIDE noise for pattern inputs)
        if t % pattern_interval == 0 and t > 0:
            for idx in pattern_inputs:
                input_spikes[idx] = pattern_amplitude  # Pattern is stronger

        # Scale all inputs
        input_spikes_scaled = input_spikes * input_scale

        # Record input spikes for raster plot (use binary for visualization)
        spike_indices = np.where(input_spikes > 0)[0]
        for idx in spike_indices:
            input_spike_times.append(t)
            input_spike_indices.append(idx)

        # Update neuron (with learning enabled, using scaled inputs)
        output_spike = neuron.step(input_spikes_scaled, I_ext=I_baseline, learning=True)

        # Record output spike
        if output_spike:
            output_spike_times.append(t)

        # Record state
        time_history.append(t)
        v_history.append(neuron.v)
        theta_history.append(neuron.theta_base + neuron.theta)
        weight_history.append(neuron.weights.copy())

    print(f"Simulation complete!")
    print(f"Total output spikes: {len(output_spike_times)}")
    print()

    # ====================
    # Visualization
    # ====================
    print("Generating visualization...")

    # Convert histories to arrays
    time_history = np.array(time_history)
    v_history = np.array(v_history)
    theta_history = np.array(theta_history)
    weight_history = np.array(weight_history)  # Shape: (n_steps, n_inputs)

    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Hidden Pattern Discovery Experiment', fontsize=16, fontweight='bold')

    # ====================
    # Top Plot: Input Raster
    # ====================
    ax1 = axes[0]

    # Separate pattern inputs from distractors
    pattern_times = []
    pattern_indices = []
    distractor_times = []
    distractor_indices = []

    for t, idx in zip(input_spike_times, input_spike_indices):
        if idx in pattern_inputs:
            pattern_times.append(t)
            pattern_indices.append(idx)
        else:
            distractor_times.append(t)
            distractor_indices.append(idx)

    # Plot distractors in black (behind)
    if distractor_times:
        ax1.scatter(distractor_times, distractor_indices, c='black', s=1, alpha=0.6, label='Noise')

    # Plot pattern inputs in red (on top)
    if pattern_times:
        ax1.scatter(pattern_times, pattern_indices, c='red', s=3, alpha=0.8, label='Pattern')

    ax1.set_ylabel('Input Channel', fontsize=11)
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_title('Input Spike Raster', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.5, n_inputs - 0.5)
    ax1.set_xlim(0, duration_ms)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ====================
    # Middle Plot: Membrane Voltage and Threshold
    # ====================
    ax2 = axes[1]

    # Plot membrane voltage
    ax2.plot(time_history, v_history, color='blue', linewidth=1, label='Membrane Voltage (v)')

    # Plot threshold
    ax2.plot(time_history, theta_history, color='orange', linestyle='--', linewidth=1.5,
             label='Threshold (θ)')

    # Mark output spikes with vertical lines
    for spike_time in output_spike_times:
        ax2.axvline(spike_time, color='green', alpha=0.3, linewidth=0.8)

    ax2.set_ylabel('Voltage (mV)', fontsize=11)
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_title('Internal State (Output spikes shown as green lines)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, duration_ms)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ====================
    # Bottom Plot: Synaptic Weight Evolution
    # ====================
    ax3 = axes[2]

    # Plot distractor weights in gray with low opacity
    distractor_inputs = [i for i in range(n_inputs) if i not in pattern_inputs]
    for idx in distractor_inputs:
        ax3.plot(time_history, weight_history[:, idx], color='gray', linewidth=0.8,
                alpha=0.3)

    # Plot pattern weights in green (on top)
    for idx in pattern_inputs:
        ax3.plot(time_history, weight_history[:, idx], color='green', linewidth=2,
                alpha=0.8, label=f'Pattern Input {idx}')

    # Add a single legend entry for distractors
    ax3.plot([], [], color='gray', linewidth=1, alpha=0.5, label='Distractor Inputs')

    ax3.set_ylabel('Synaptic Weight', fontsize=11)
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_title('Learning: Synaptic Weight Evolution', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, duration_ms)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ====================
    # Print Final Statistics
    # ====================
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)

    final_weights = weight_history[-1]
    pattern_weights = final_weights[pattern_inputs]
    distractor_weights = final_weights[distractor_inputs]

    print(f"\nFinal Pattern Weights (inputs {pattern_inputs}):")
    for idx, w in zip(pattern_inputs, pattern_weights):
        print(f"  Input {idx:2d}: {w:.4f}")

    print(f"\nPattern Weight Statistics:")
    print(f"  Mean: {np.mean(pattern_weights):.4f}")
    print(f"  Std:  {np.std(pattern_weights):.4f}")
    print(f"  Min:  {np.min(pattern_weights):.4f}")
    print(f"  Max:  {np.max(pattern_weights):.4f}")

    print(f"\nDistractor Weight Statistics:")
    print(f"  Mean: {np.mean(distractor_weights):.4f}")
    print(f"  Std:  {np.std(distractor_weights):.4f}")
    print(f"  Min:  {np.min(distractor_weights):.4f}")
    print(f"  Max:  {np.max(distractor_weights):.4f}")

    separation = np.mean(pattern_weights) - np.mean(distractor_weights)
    print(f"\nWeight Separation (Pattern - Distractor): {separation:.4f}")

    if separation > 0.02:
        print(f"\n✓ SUCCESS: Pattern weights ({np.mean(pattern_weights):.3f}) are separated from noise ({np.mean(distractor_weights):.3f})!")
        print(f"  Pattern inputs have learned the hidden pattern!")
    else:
        print("\n⚠ Pattern weights have not fully separated yet. Consider longer training.")

    print("="*60)

    # Show the plot
    plt.show()

    return {
        'time': time_history,
        'weights': weight_history,
        'v': v_history,
        'theta': theta_history,
        'output_spikes': output_spike_times,
        'pattern_weights_final': pattern_weights,
        'distractor_weights_final': distractor_weights
    }


if __name__ == "__main__":
    print("="*60)
    print("HIDDEN PATTERN DISCOVERY EXPERIMENT")
    print("="*60)
    print()
    print("This experiment demonstrates how a BiologicalNeuron with STDP")
    print("can learn to identify a repeating pattern hidden in noise.")
    print()

    # Run the experiment
    results = run_hidden_pattern_experiment()

    print("\nExperiment complete! Close the plot window to exit.")

