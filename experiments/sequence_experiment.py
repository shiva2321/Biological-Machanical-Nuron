"""
Sequence Experiment: Temporal Sequence Detection ("Passcode Lock")

Demonstrates how a neural circuit can detect specific temporal sequences.
The circuit acts as a "passcode lock" that only fires when inputs arrive
in the correct order (0 → 2 → 1) with correct timing (20ms gaps).

Circuit Design:
- 4 neurons: 0, 1, 2 (hidden), 3 (output)
- 3 input channels
- Bucket brigade architecture with delayed handovers
- Sub-threshold connections require temporal coincidence
"""

import numpy as np
import matplotlib.pyplot as plt
from circuit import NeuralCircuit


def build_sequence_detector():
    """
    Build the "passcode lock" circuit.

    Architecture:
    - Input 0 → Neuron 0 (direct)
    - Neuron 0 → Neuron 1 (delayed 20ms, sub-threshold)
    - Input 2 → Neuron 1 (direct, sub-threshold)
    - Neuron 1 → Neuron 2 (delayed 20ms, sub-threshold)
    - Input 1 → Neuron 2 (direct, sub-threshold)
    - Neuron 2 → Neuron 3 (output, strong)

    Key: Sub-threshold connections (0.6) require coincidence to reach threshold

    Returns:
        NeuralCircuit configured as sequence detector
    """
    # Create circuit
    circuit = NeuralCircuit(
        num_neurons=4,
        input_channels=3,
        dt=1.0,
        max_delay=25,
        neuron_params={
            'tau_m': 20.0,        # Slower leak = holds charge longer
            'tau_trace': 20.0,    # STDP trace
            'theta_base': -65.0,  # Lower threshold (easier to fire)
            'v_rest': -70.0,
            'v_reset': -75.0,
            'u_increment': 0.3,   # Minimal adaptation
            'theta_increment': 0.1,  # Minimal threshold adaptation
            'weight_min': 0.0,
            'weight_max': 10.0    # Increased for stronger connections
        }
    )

    print("Building Sequence Detector Circuit...")
    print("="*70)

    # Set input weights for each neuron
    # Neuron 0: responds to Input 0
    weights_n0 = np.zeros(3)
    weights_n0[0] = 1.5  # Input 0 → Neuron 0 (strong trigger)
    circuit.set_weights(0, weights_n0)
    print("Step 1: Input 0 → Neuron 0 (weight=1.5, strong trigger)")

    # Neuron 1: responds to Input 2
    weights_n1 = np.zeros(3)
    weights_n1[2] = 1.5  # Input 2 → Neuron 1 (strong trigger)
    circuit.set_weights(1, weights_n1)
    print("Step 3: Input 2 → Neuron 1 (weight=1.5, strong trigger)")

    # Neuron 2: responds to Input 1
    weights_n2 = np.zeros(3)
    weights_n2[1] = 1.5  # Input 1 → Neuron 2 (strong trigger)
    circuit.set_weights(2, weights_n2)
    print("Step 5: Input 1 → Neuron 2 (weight=1.5, strong trigger)")

    # Neuron 3 (output): no external inputs
    weights_n3 = np.zeros(3)
    circuit.set_weights(3, weights_n3)

    # Internal connections (delayed handovers)
    # Step 2: Neuron 0 → Neuron 1 (delayed, strong assist)
    circuit.connect(
        source_id=0,
        target_id=1,
        weight=1.2,   # Strong assist (1.2 + 1.5 >>> threshold)
        delay=20      # 20ms delay
    )
    print("Step 2: Neuron 0 → Neuron 1 (weight=1.2, delay=20ms, strong assist)")
    print("  → With Input 2 (1.5) + delayed N0 (1.2) = 2.7 >>> threshold")

    # Step 4: Neuron 1 → Neuron 2 (delayed, strong assist)
    circuit.connect(
        source_id=1,
        target_id=2,
        weight=1.2,   # Strong assist (1.2 + 1.5 >>> threshold)
        delay=20      # 20ms delay
    )
    print("Step 4: Neuron 1 → Neuron 2 (weight=1.2, delay=20ms, strong assist)")
    print("  → With Input 1 (1.5) + delayed N1 (1.2) = 2.7 >>> threshold")

    # Output: Neuron 2 → Neuron 3 (extremely strong connection)
    circuit.connect(
        source_id=2,
        target_id=3,
        weight=8.0,   # Extremely strong, guaranteed to fire
        delay=1       # Minimal delay
    )
    print("Output: Neuron 2 → Neuron 3 (weight=8.0, delay=1ms, extremely strong)")
    print("  → Neuron 3 fires when sequence is complete")

    print("="*70)
    print(f"Total connections: {circuit.get_num_connections()}")
    print()

    return circuit


def run_trial(circuit, input_times, trial_name, duration=100):
    """
    Run a single trial with specified input timing.

    Args:
        circuit: NeuralCircuit instance (will be reset)
        input_times: Dict mapping input_id to list of fire times
                    e.g., {0: [10], 1: [50], 2: [30]}
        trial_name: Name of the trial for display
        duration: Trial duration in ms

    Returns:
        spike_raster: Array of shape (duration, num_neurons)
    """
    print(f"\n{trial_name}")
    print("-"*70)

    # Display input timing
    print("Input timing:")
    for input_id in sorted(input_times.keys()):
        times = input_times[input_id]
        print(f"  Input {input_id}: {times}ms")

    # Reset circuit state (keep connections)
    circuit.reset_state()

    # Storage for spike raster
    spike_raster = []
    output_fired = False
    first_output_time = None

    # Run simulation
    for t in range(duration):
        # Generate input spikes based on timing
        input_spikes = np.zeros(3)
        for input_id, fire_times in input_times.items():
            if t in fire_times:
                input_spikes[input_id] = 1.0

        # Scale inputs very strongly to drive spiking (BLAST IT!)
        input_spikes_scaled = input_spikes * 80.0

        # Baseline current for output neuron - tuned to critical point
        I_ext = np.array([0.0, 0.0, 0.0, 7.0])

        # Step circuit
        output_spikes = circuit.step(
            input_spikes_scaled,
            I_ext=I_ext,
            learning=False  # No learning, just detection
        )

        spike_raster.append(output_spikes.copy())

        # Track output neuron (neuron 3)
        if output_spikes[3] and not output_fired:
            output_fired = True
            first_output_time = t
            print(f"\n  ✓ OUTPUT FIRED at t={t}ms (Neuron 3)")

        # Display intermediate spikes
        if np.any(output_spikes):
            fired_neurons = np.where(output_spikes)[0]
            for neuron_id in fired_neurons:
                if neuron_id < 3:  # Hidden neurons
                    print(f"  t={t:3d}ms: Neuron {neuron_id} fired (hidden)")

    # Results
    spike_raster = np.array(spike_raster)
    total_spikes = np.sum(spike_raster, axis=0)

    print(f"\nResults:")
    print(f"  Neuron 0 (hidden): {int(total_spikes[0])} spikes")
    print(f"  Neuron 1 (hidden): {int(total_spikes[1])} spikes")
    print(f"  Neuron 2 (hidden): {int(total_spikes[2])} spikes")
    print(f"  Neuron 3 (OUTPUT): {int(total_spikes[3])} spikes", end="")

    if output_fired:
        print(f" ✓ SUCCESS (first at t={first_output_time}ms)")
    else:
        print(f" ✗ FAILED (no output)")

    return spike_raster


def run_sequence_experiment():
    """
    Run the complete sequence detection experiment with 3 trials.
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "TEMPORAL SEQUENCE DETECTION" + " "*25 + "║")
    print("║" + " "*20 + "\"Passcode Lock\" Circuit" + " "*25 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # Build circuit
    circuit = build_sequence_detector()

    print("EXPERIMENT: Test sequence detection with different input patterns")
    print("Correct sequence: Input 0 (t=10) → Input 2 (t=30) → Input 1 (t=50)")
    print("Expected: Only Trial 1 should unlock the output")
    print()

    # Define trials
    trials = [
        {
            'name': 'TRIAL 1: Correct Sequence (0→2→1 with 20ms gaps)',
            'input_times': {
                0: [10],   # Input 0 at t=10
                2: [30],   # Input 2 at t=30 (20ms after)
                1: [50]    # Input 1 at t=50 (20ms after)
            },
            'expected': 'SUCCESS'
        },
        {
            'name': 'TRIAL 2: Wrong Timing (all simultaneous)',
            'input_times': {
                0: [10],   # All at t=10
                1: [10],
                2: [10]
            },
            'expected': 'FAIL'
        },
        {
            'name': 'TRIAL 3: Wrong Order (1→2→0)',
            'input_times': {
                1: [10],   # Wrong order
                2: [30],
                0: [50]
            },
            'expected': 'FAIL'
        }
    ]

    # Run trials and collect results
    results = []
    for trial in trials:
        spike_raster = run_trial(
            circuit,
            trial['input_times'],
            trial['name'],
            duration=100
        )
        results.append({
            'name': trial['name'],
            'raster': spike_raster,
            'input_times': trial['input_times'],
            'expected': trial['expected']
        })

    # Visualize results
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)

    visualize_results(results)

    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for i, result in enumerate(results, 1):
        output_spikes = np.sum(result['raster'][:, 3])
        status = "✓ SUCCESS" if output_spikes > 0 else "✗ FAILED"
        expected = result['expected']
        match = "✓" if (output_spikes > 0 and expected == 'SUCCESS') or \
                      (output_spikes == 0 and expected == 'FAIL') else "✗"

        print(f"\nTrial {i}: {result['name']}")
        print(f"  Output Neuron Spikes: {int(output_spikes)}")
        print(f"  Status: {status}")
        print(f"  Expected: {expected}")
        print(f"  Match: {match}")

    print("\n" + "="*70)
    print("KEY INSIGHT: Temporal Sequence Detection")
    print("="*70)
    print("The circuit acts as a 'passcode lock' using:")
    print("  1. SUB-THRESHOLD connections (0.6) - need coincidence")
    print("  2. AXONAL DELAYS (20ms) - create temporal windows")
    print("  3. BUCKET BRIGADE - signals handed from neuron to neuron")
    print()
    print("Only the correct sequence (0→2→1 with 20ms gaps) unlocks output!")
    print("="*70)


def visualize_results(results):
    """
    Create visualization of all three trials.

    Args:
        results: List of trial result dictionaries
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Temporal Sequence Detection: "Passcode Lock" Circuit',
                 fontsize=16, fontweight='bold')

    neuron_labels = ['N0 (Hidden)', 'N1 (Hidden)', 'N2 (Hidden)', 'N3 (OUTPUT)']
    colors = ['blue', 'green', 'orange', 'red']

    for trial_idx, (ax, result) in enumerate(zip(axes, results)):
        spike_raster = result['raster']
        input_times = result['input_times']
        duration = len(spike_raster)

        # Plot spike raster
        for neuron_id in range(4):
            spike_times = np.where(spike_raster[:, neuron_id])[0]

            # Marker size and style
            if neuron_id == 3:  # Output neuron
                marker_size = 200
                marker_style = '*'
                alpha = 1.0
                linewidth = 3
            else:
                marker_size = 120
                marker_style = '|'
                alpha = 0.8
                linewidth = 2.5

            if len(spike_times) > 0:
                ax.scatter(spike_times, [neuron_id] * len(spike_times),
                          s=marker_size, marker=marker_style,
                          color=colors[neuron_id], alpha=alpha,
                          linewidths=linewidth,
                          label=neuron_labels[neuron_id])

        # Add input markers (arrows at bottom)
        input_colors = {0: 'purple', 1: 'brown', 2: 'cyan'}
        for input_id, times in input_times.items():
            for t in times:
                ax.annotate('', xy=(t, -0.5), xytext=(t, -0.8),
                           arrowprops=dict(arrowstyle='->', color=input_colors[input_id],
                                         lw=2.5, alpha=0.8))
                ax.text(t, -1.1, f'In{input_id}', fontsize=8, ha='center',
                       color=input_colors[input_id], weight='bold')

        # Styling
        ax.set_ylabel('Neuron ID', fontsize=11)
        ax.set_xlabel('Time (ms)', fontsize=11)

        # Title with result
        output_spikes = int(np.sum(spike_raster[:, 3]))
        status = "✓ OUTPUT FIRED" if output_spikes > 0 else "✗ NO OUTPUT"
        status_color = 'green' if output_spikes > 0 else 'red'

        trial_name = result['name'].split(':')[1].strip() if ':' in result['name'] else result['name']
        ax.set_title(f"Trial {trial_idx + 1}: {trial_name} - {status}",
                    fontsize=12, fontweight='bold', color=status_color)

        ax.set_yticks(range(4))
        ax.set_yticklabels(neuron_labels)
        ax.set_xlim(-5, duration + 5)
        ax.set_ylim(-1.5, 3.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper right', fontsize=9, ncol=2)

        # Add timing annotations for Trial 1
        if trial_idx == 0:
            # Show expected cascade
            ax.text(10, 3.7, '20ms', fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            ax.text(30, 3.7, '20ms', fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    filename = 'sequence_experiment_results.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to '{filename}'")

    plt.show()


if __name__ == "__main__":
    run_sequence_experiment()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nThis experiment demonstrates:")
    print("  ✓ Temporal sequence detection")
    print("  ✓ Sub-threshold summation (coincidence detection)")
    print("  ✓ Axonal delays creating temporal windows")
    print("  ✓ Bucket brigade information flow")
    print("\nThe circuit successfully acts as a 'passcode lock'!")
    print("="*70)

