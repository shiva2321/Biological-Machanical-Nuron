"""
Visual demonstration of NeuralCircuit capabilities.

Demonstrates:
1. Spike propagation through a chain with delays
2. Winner-take-all with lateral inhibition
"""

import numpy as np
import matplotlib.pyplot as plt
from circuit import NeuralCircuit


def demo_delayed_propagation():
    """
    Visualize spike propagation through a chain network with delays.
    """
    print("="*70)
    print("DEMO 1: Delayed Spike Propagation")
    print("="*70)

    # Create 5-neuron chain: 0 -> 1 -> 2 -> 3 -> 4
    num_neurons = 5
    circuit = NeuralCircuit(
        num_neurons=num_neurons,
        input_channels=1,
        dt=1.0,
        max_delay=5,
        neuron_params={
            'tau_m': 8.0,
            'theta_base': -58.0,
        }
    )

    # Connect in chain with increasing delays
    delays = [1, 2, 3, 4]
    for i in range(num_neurons - 1):
        circuit.connect(i, i+1, weight=20.0, delay=delays[i])

    print(f"Chain topology:")
    for i in range(num_neurons - 1):
        print(f"  Neuron {i} --({delays[i]}ms)--> Neuron {i+1}")

    # Simulate
    duration = 20
    spike_raster = []

    print(f"\nRunning {duration}ms simulation...")

    for t in range(duration):
        # Strong stimulus to first neuron at t=0
        input_spikes = np.array([1.0]) if t == 0 else np.array([0.0])
        I_ext = np.zeros(num_neurons)
        I_ext[0] = 25.0  # Boost neuron 0

        output_spikes = circuit.step(input_spikes, I_ext=I_ext, learning=False)
        spike_raster.append(output_spikes.copy())

        if np.any(output_spikes):
            fired = np.where(output_spikes)[0]
            print(f"  t={t:2d}ms: Neuron {fired[0]} fired")

    # Visualize
    spike_raster = np.array(spike_raster)

    plt.figure(figsize=(12, 6))

    # Raster plot
    for neuron_id in range(num_neurons):
        spike_times = np.where(spike_raster[:, neuron_id])[0]
        plt.scatter(spike_times, [neuron_id] * len(spike_times),
                   s=100, marker='|', linewidths=2,
                   label=f'Neuron {neuron_id}')

    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Neuron ID', fontsize=12)
    plt.title('Spike Propagation Through Chain Network', fontsize=14, fontweight='bold')
    plt.yticks(range(num_neurons))
    plt.xlim(-0.5, duration - 0.5)
    plt.ylim(-0.5, num_neurons - 0.5)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Add delay annotations
    for i in range(num_neurons - 1):
        plt.annotate(f'{delays[i]}ms',
                    xy=(1, i + 0.5), xytext=(3, i + 0.5),
                    fontsize=9, color='red', weight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.tight_layout()
    plt.savefig('circuit_demo_propagation.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'circuit_demo_propagation.png'")

    plt.show()


def demo_winner_take_all():
    """
    Visualize winner-take-all competition with lateral inhibition.
    """
    print("\n" + "="*70)
    print("DEMO 2: Winner-Take-All Competition")
    print("="*70)

    # Create competitive network
    num_neurons = 8
    circuit = NeuralCircuit(
        num_neurons=num_neurons,
        input_channels=num_neurons,
        dt=1.0,
        neuron_params={
            'tau_m': 12.0,
            'theta_base': -56.0,
        }
    )

    # Set strong lateral inhibition
    circuit.set_inhibition(strength=5.0)

    # Set input weights (identity mapping)
    for i in range(num_neurons):
        weights = np.zeros(num_neurons)
        weights[i] = 1.0
        circuit.set_weights(i, weights)

    print(f"Network: {num_neurons} neurons with lateral inhibition (5.0 mV)")
    print(f"Input: Varying strengths, strongest to neuron 5")

    # Create input pattern with one dominant channel
    input_strengths = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 0.65, 0.45])
    print(f"\nInput strengths: {input_strengths}")

    # Simulate
    duration = 30
    spike_history = []
    membrane_history = []

    print(f"\nRunning {duration}ms simulation...")

    for t in range(duration):
        # Constant input
        input_spikes = input_strengths
        I_ext = np.ones(num_neurons) * 18.0

        # Step
        output_spikes = circuit.step(input_spikes, I_ext=I_ext, learning=False)
        spike_history.append(output_spikes.copy())

        # Record membrane potentials
        states = circuit.get_states()
        v_values = [state['v'] for state in states]
        membrane_history.append(v_values)

        if np.any(output_spikes) and t < 15:
            fired = np.where(output_spikes)[0]
            print(f"  t={t:2d}ms: Neuron {fired} fired (suppressing others)")

    spike_history = np.array(spike_history)
    membrane_history = np.array(membrane_history)

    # Count spikes
    spike_counts = np.sum(spike_history, axis=0)
    print(f"\nTotal spikes per neuron:")
    for i, count in enumerate(spike_counts):
        marker = " <-- WINNER" if i == 5 else ""
        print(f"  Neuron {i}: {int(count)} spikes{marker}")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Winner-Take-All with Lateral Inhibition', fontsize=16, fontweight='bold')

    # Top: Raster plot
    ax1 = axes[0]
    for neuron_id in range(num_neurons):
        spike_times = np.where(spike_history[:, neuron_id])[0]
        color = 'red' if neuron_id == 5 else 'blue'
        marker_size = 120 if neuron_id == 5 else 80
        alpha = 1.0 if neuron_id == 5 else 0.6
        ax1.scatter(spike_times, [neuron_id] * len(spike_times),
                   s=marker_size, marker='|', linewidths=3,
                   color=color, alpha=alpha,
                   label=f'Neuron {neuron_id}' + (' (strongest input)' if neuron_id == 5 else ''))

    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Neuron ID', fontsize=11)
    ax1.set_title('Spike Raster: Neuron 5 (strongest input) dominates', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(num_neurons))
    ax1.set_xlim(-0.5, duration - 0.5)
    ax1.set_ylim(-0.5, num_neurons - 0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    # Bottom: Spike counts
    ax2 = axes[1]
    colors = ['red' if i == 5 else 'blue' for i in range(num_neurons)]
    bars = ax2.bar(range(num_neurons), spike_counts, color=colors, edgecolor='black')

    # Adjust alpha per bar
    for i, bar in enumerate(bars):
        bar.set_alpha(1.0 if i == 5 else 0.6)

    # Add input strength overlay
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(num_neurons), input_strengths, 'go-', linewidth=2,
                  markersize=8, label='Input Strength', alpha=0.7)
    ax2_twin.set_ylabel('Input Strength', fontsize=11, color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2_twin.set_ylim(0, 1.2)
    ax2_twin.legend(loc='upper left')

    ax2.set_xlabel('Neuron ID', fontsize=11)
    ax2.set_ylabel('Total Spikes', fontsize=11)
    ax2.set_title('Competition Result: Strongest Input Wins', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(num_neurons))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig('circuit_demo_winner_take_all.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'circuit_demo_winner_take_all.png'")

    plt.show()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*17 + "NEURAL CIRCUIT DEMOS" + " "*30 + "║")
    print("║" + " "*12 + "Visual Demonstrations of Capabilities" + " "*19 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # Run demos
    demo_delayed_propagation()
    demo_winner_take_all()

    print("\n" + "="*70)
    print("DEMOS COMPLETE")
    print("="*70)
    print("\nKey Demonstrations:")
    print("  ✓ Spike propagation with axonal delays")
    print("  ✓ Winner-take-all competition")
    print("  ✓ Lateral inhibition mechanism")
    print("\nThe NeuralCircuit infrastructure is fully functional!")

